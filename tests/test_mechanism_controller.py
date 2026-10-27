"""Single-seed stage controllers must preserve dependency and GPU barriers."""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_moe_mechanisms import Controller, digest, STAGE_MATRIX
from scripts.run_organ_weighted import write_json, exclusive_lock


def seal(value, field='protocol_sha256'):
    return {**value, field: digest(value)}


def setup(tmp_path, stage='q06'):
    root, reference = tmp_path/stage, tmp_path/'q05'
    after = reference if stage == 'q06' else tmp_path/'q06'
    models, arms = STAGE_MATRIX[stage]
    pred = seal(dict(protocol_version='q05_moe_refinement_v1' if stage == 'q06' else 'q06_q07_mechanism_v1',
        stage='q05' if stage == 'q06' else 'q06', config=dict(models=['region','task_id'] if stage == 'q06' else ['region'],
        arms=['control','refined'] if stage == 'q06' else ['control','routing'],
        seeds=[42,43,44] if stage == 'q06' else [42], epochs=80)))
    selection = seal(dict(protocol_sha256=pred['protocol_sha256'], main_loss_arm='organ_exposure_fg_bg'), 'selection_sha256')
    write_json(after/'protocol.json', pred)
    write_json(after/'main_loss_selection.json', selection)
    if stage == 'q07':
        write_json(reference/'protocol.json', {'protocol_sha256':'a'*64})
    protocol = seal(dict(protocol_version='q06_q07_mechanism_v1', stage=stage,
        reference_root=str(reference.resolve()), after_root=str(after.resolve()),
        after_protocol_sha256=pred['protocol_sha256'],
        reference_protocol_sha256=pred['protocol_sha256'] if stage == 'q06' else 'a'*64,
        config=dict(models=list(models), arms=list(arms), seeds=[42], epochs=80)))
    write_json(root/'protocol.json', protocol)
    own = seal(dict(protocol_sha256=protocol['protocol_sha256'], main_loss_arm='organ_exposure_fg_bg'), 'selection_sha256')
    write_json(root/'main_loss_selection.json', own)
    (root/'source_snapshot/unet_moe').mkdir(parents=True)
    (root/'source_snapshot/unet_moe/mechanism_suite.py').write_text('')
    c=Controller(root,reference,after,poll_seconds=.01)
    c.gpu_memory=lambda _:48000
    c.gpu_lock_path=lambda gpu: tmp_path/f'gpu{gpu}.lock'
    c.validate_reference_evidence=lambda: None
    return c, pred, selection


def finish_predecessor(c, pred, selection):
    runs=[]
    cfg=pred['config']
    for kind in cfg['models']:
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                folder=c.after_root/arm/kind/f'seed_{seed}'
                row=dict(kind=kind,seed=seed,arm=arm,protocol_sha256=pred['protocol_sha256'],
                    main_selection_sha256=selection['selection_sha256'],epochs_completed=80,test_evaluated=True)
                runs.append(row)
                write_json(folder/'run_summary.json',row)
                for name in ('best.pt','last.pt','metrics.jsonl','test_best.json','test_last.json'):
                    (folder/name).write_text('evidence')
    write_json(c.after_root/'summary.json',dict(complete=True,protocol_sha256=pred['protocol_sha256'],
        main_selection_sha256=selection['selection_sha256'],runs=runs))
    write_json(c.after_root/'controller_status.json',dict(state='complete',protocol_sha256=pred['protocol_sha256']))


@pytest.mark.parametrize('stage,count',[('q06',2),('q07',3)])
def test_matrix_and_preflight_train_test_barriers(tmp_path,stage,count):
    c,_,_=setup(tmp_path,stage)
    assert len(c.jobs)==len(c.preflights)==count
    assert {j['seed'] for j in c.jobs}=={42}
    assert all(t[3]=='preflight' for t in c.pending_tasks())
    for p in c.preflights.values():p['state']='complete'
    assert len(c.pending_tasks())==count and all(t[3]=='train' for t in c.pending_tasks())
    for j in c.jobs:j['train_state']='complete'
    c.jobs[-1]['train_state']='running'
    assert c.pending_tasks()==[]
    c.jobs[-1]['train_state']='complete'
    assert len(c.pending_tasks())==count and all(t[3]=='evaluate' for t in c.pending_tasks())


def test_no_seed_expansion_or_unknown_arm(tmp_path):
    c,_,_=setup(tmp_path)
    p=c.protocol.copy();p.pop('protocol_sha256');p['config']={**p['config'],'seeds':[42,43,44]}
    write_json(c.root/'protocol.json',seal(p))
    with pytest.raises(ValueError,match='seed42'):
        Controller(c.root,c.reference_root,c.after_root)


def test_training_resume_and_cpu_verify_commands(tmp_path):
    c,_,_=setup(tmp_path,'q07');task=('task_id',42,'experts0','train')
    cmd=c.command(task)
    assert cmd[2]=='unet_moe.mechanism_suite' and '--resume' in cmd
    assert c.command(task,action='verify',device='cpu')[-1]=='verify'
    assert '--resume' not in c.command(task,action='verify')


@pytest.mark.parametrize('stage,count',[('q06',12),('q07',2)])
def test_stage_specific_predecessor_matrix(tmp_path,stage,count):
    c,p,s=setup(tmp_path,stage);finish_predecessor(c,p,s)
    assert c.prerequisites()[0]
    summary=json.loads((c.after_root/'summary.json').read_text())
    assert len(summary['runs'])==count
    summary['runs'].pop();write_json(c.after_root/'summary.json',summary)
    assert c.prerequisites()[1]=='blocked_predecessor'


def test_q07_waits_for_q06_and_never_probes_gpu(tmp_path):
    c,p,_=setup(tmp_path,'q07')
    c.gpu_memory=lambda _:pytest.fail('waiting dependency touched GPU')
    write_json(c.after_root/'controller_status.json',dict(state='training',protocol_sha256=p['protocol_sha256']))
    assert c.prerequisites()[1]=='waiting_q06'


def test_reference_trigger_failure_blocks_both_stages(tmp_path):
    c,p,s=setup(tmp_path,'q07');finish_predecessor(c,p,s)
    def fail():raise ValueError('trigger digest changed')
    c.validate_reference_evidence=fail
    assert c.prerequisites()[1]=='blocked_predecessor'


def test_missing_predecessor_checkpoint_blocks_completion_claim(tmp_path):
    c,p,s=setup(tmp_path);finish_predecessor(c,p,s)
    (c.after_root/'refined/task_id/seed_44/last.pt').unlink()
    assert c.prerequisites()[1]=='blocked_predecessor'


def test_failed_predecessor_tasks_override_complete_state(tmp_path):
    c,p,s=setup(tmp_path);finish_predecessor(c,p,s)
    write_json(c.after_root/'controller_status.json',dict(state='complete',protocol_sha256=p['protocol_sha256'],
        jobs=[dict(train_state='failed',evaluate_state='pending')]))
    assert c.prerequisites()[1]=='blocked_predecessor'


def test_missing_dependencies_never_launch_in_once(tmp_path):
    c,_,_=setup(tmp_path,'q07')
    c.admit=lambda *a:pytest.fail('once admitted GPU')
    c.verify_predecessor=lambda:pytest.fail('once ran heavy verification')
    assert c.run(once=True)==0
    assert json.loads((c.root/'controller_status.json').read_text())['state']=='waiting_q06'


def test_dry_run_has_no_filesystem_side_effects(tmp_path,capsys):
    c,_,_=setup(tmp_path)
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    c.gpu_memory=lambda _:pytest.fail('dry run touched GPU')
    assert c.run(dry_run=True)==0
    assert len(json.loads(capsys.readouterr().out)['jobs'])==2
    after={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after


def test_failure_requires_explicit_retry(tmp_path):
    c,_,_=setup(tmp_path);c.jobs[0]['train_state']='failed';c.update('blocked_failed_tasks')
    restored=Controller(c.root,c.reference_root,c.after_root);restored.recover()
    assert restored.failed() and not restored.pending_tasks()
    retried=Controller(c.root,c.reference_root,c.after_root,retry_failed=True);retried.recover()
    assert not retried.failed()


def test_nonzero_exit_is_durable_and_stops_dispatch(tmp_path):
    c,_,_=setup(tmp_path);task=('region',42,'routing','train');job=c._job(*task[:3])
    job.update(train_state='running',log='fixture.log',pid=99)
    c.active[0]=dict(process=SimpleNamespace(poll=lambda:17),task=task,
        lock=(c.root/'fake.lock').open('a+'),log=io.StringIO())
    c.reap()
    assert job['train_state']=='failed' and not c.pending_tasks()
    assert json.loads((c.root/'failures/region_42_routing_train.json').read_text())['returncode']==17


def test_gpu_memory_and_shared_lock_required(tmp_path):
    c,_,_=setup(tmp_path);task=('region',42,'control','preflight')
    with exclusive_lock(c.gpu_lock_path(0)):assert c.admit(0,[task]) is None
    c.gpu_memory=lambda _:40000
    assert c.admit(0,[task]) is None
    c.gpu_memory=lambda _:48000
    actual,lock,_=c.admit(0,[task]);assert actual==task;lock.close()


def test_cannot_launch_test_before_all_training(tmp_path):
    c,_,_=setup(tmp_path)
    with pytest.raises(ValueError,match='all training'):
        c.launch(0,('region',42,'control','evaluate'),None,48000)


def test_orphan_worker_holds_root_and_gpu_lock(tmp_path):
    c,_,_=setup(tmp_path);c.read_selection();task=('region',42,'control','preflight')
    c.suite_prefix=[sys.executable,'-c','import time; time.sleep(20)']
    try:
        with exclusive_lock(c.root/'.controller.lock') as lock:
            c.root_lock=lock
            task,gpu_lock,free=c.admit(0,[task]);c.launch(0,task,gpu_lock,free);gpu_lock.close()
        for p in (c.root/'.controller.lock',c.gpu_lock_path(0)):
            with pytest.raises(RuntimeError,match='already owns'):
                with exclusive_lock(p):pass
        assert c.active[0]['process'].poll() is None
    finally:c.stop()


def test_preflight_environment_is_not_inherited_by_training(tmp_path,monkeypatch):
    c,_,_=setup(tmp_path);c.read_selection();seen=[]
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    def fake_popen(*args,**kwargs):
        seen.append(kwargs);return SimpleNamespace(pid=100,poll=lambda:None)
    monkeypatch.setattr('scripts.run_moe_mechanisms.subprocess.Popen',fake_popen)
    with exclusive_lock(c.root/'.controller.lock') as lock:
        c.root_lock=lock
        task=('region',42,'control','train');c.required_mib['region_control']=10000
        _,gpu_lock,free=c.admit(0,[task]);c.launch(0,task,gpu_lock,free)
    assert 'CUBLAS_WORKSPACE_CONFIG' not in seen[0]['env']
    c.active[0]['lock'].close();c.active[0]['log'].close()


def test_retry_preserves_preflight_failure_history(tmp_path):
    c,_,_=setup(tmp_path)
    c.preflights['region_control'].update(state='failed',failures=[{'error':'admission rejected'}])
    c.update('blocked_failed_tasks')
    retried=Controller(c.root,c.reference_root,c.after_root,retry_failed=True);retried.recover()
    assert retried.preflights['region_control']['state']=='pending'
    assert retried.preflights['region_control']['failures']==[{'error':'admission rejected'}]


def test_recovery_verifies_entire_train_matrix_before_any_test(tmp_path):
    c,_,_=setup(tmp_path,'q07');seen=[]
    for job in c.jobs:
        write_json(c.folder(job)/'run_summary.json',dict(epochs_completed=80,test_evaluated=True))
    c.validate_task=lambda task:seen.append(task)
    c.recover()
    assert [task[-1] for task in seen]==['train']*3+['evaluate']*3
    assert all(j['evaluate_state']=='complete' for j in c.jobs)


def test_recovery_refuses_test_when_other_arm_untrained(tmp_path):
    c,_,_=setup(tmp_path);seen=[]
    write_json(c.folder(c.jobs[0])/'run_summary.json',dict(epochs_completed=80,test_evaluated=True))
    c.validate_task=lambda task:seen.append(task)
    with pytest.raises(ValueError,match='all-training barrier'):c.recover()
    assert [task[-1] for task in seen]==['train']


def test_cli_reexecs_snapshot_before_instantiating_controller(tmp_path,monkeypatch):
    from scripts import run_moe_mechanisms as module
    c,_,_=setup(tmp_path)
    entry=c.code_root/'scripts/run_moe_mechanisms.py';entry.parent.mkdir();entry.write_text('frozen')
    argv=['--output',str(c.root),'--reference-root',str(c.reference_root),
          '--after-root',str(c.after_root),'--dry-run']
    seen=[]
    def reexec(executable,args):seen.append((executable,args));raise RuntimeError('reexec frozen')
    monkeypatch.setattr(module.os,'execv',reexec)
    monkeypatch.setattr(module,'Controller',lambda *a,**kw:pytest.fail('mutable controller instantiated'))
    with pytest.raises(RuntimeError,match='reexec frozen'):module.main(argv)
    assert seen==[(sys.executable,[sys.executable,str(entry),*argv])]


@pytest.mark.parametrize('change',[
    {},{'numerical_scope':{}},{'update_deterministic_modes':[False]*4},
    {'update_cudnn_deterministic':[False]*4},
    {'gpu_training_admissible':False},{'required_free_mib':True},
])
def test_real_suite_preflight_admission_enforces_scope_and_memory(tmp_path,monkeypatch,change):
    from unet_moe import mechanism_suite as suite
    c,_,_=setup(tmp_path);selection=c.read_selection()
    monkeypatch.setattr(suite,'with_selection',lambda root,p:{**p,'_selection':selection})
    report=dict(protocol_sha256=c.protocol_sha256,kind='region',arm='control',passed=True,
        main_selection_sha256=selection['selection_sha256'],main_loss_arm=selection['main_loss_arm'],
        real_optimizer_updates=2,effective_batch=16,micro_batch=4,size=512,real_query_images=32,
        strict_model_optimizer_rng_reload=True,identical_reload_prediction=True,identical_resume_update=True,
        continuation_verification_updates=2,continuation_sample_count=32,gpu_training_admissible=True,
        numerical_scope=suite.NUMERICAL_SCOPE,update_deterministic_modes=[False,False,True,True],
        update_cudnn_deterministic=[True]*4,
        required_free_mib=18000)
    report.update(change);write_json(c.root/'preflight/region_control.json',report)
    if change:
        with pytest.raises(ValueError):c.read_preflight('region','control')
        assert c.preflights['region_control']['state']=='pending'
        assert c.required_mib=={}
    else:
        assert c.read_preflight('region','control')==report
        assert c.required_mib=={'region_control':18000}
