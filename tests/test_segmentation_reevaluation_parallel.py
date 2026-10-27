"""Scheduling isolation over the original frozen evaluator contract."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_segmentation_reevaluation as original
from scripts import run_segmentation_reevaluation_parallel as parallel


class Worker:
    next_pid = 1000000

    def __init__(self, command, **kwargs):
        Worker.next_pid += 1
        self.pid = Worker.next_pid
        self.command, self.kwargs, self.returncode = command, kwargs, None

    def poll(self):
        return self.returncode

    def wait(self, *args, **kwargs):
        pytest.fail('Scheduler must not wait synchronously for an active worker')


@pytest.fixture
def environment(tmp_path, monkeypatch):
    project = tmp_path / 'checkout'
    for folder in ('scripts', 'unet_moe'):
        (project / folder).mkdir(parents=True)
        (project / folder / '__init__.py').write_text('')
    runs = tmp_path / 'runs'
    runs.mkdir()
    checkpoint = runs / 'best.pt'
    checkpoint.write_bytes(b'immutable checkpoint')
    jobs = [dict(job_id=f'job-{i}', family='fair', experiment='completed', kind='shared', arm=None,
                 seed=42, selection='best', split='val', size=512, status='ready',
                 checkpoint=str(checkpoint), records_path=str(runs / 'records.json'), config={'size': 512})
            for i in range(4)]
    inventory = dict(jobs=jobs, counts={'ready': len(jobs)})
    monkeypatch.setattr(original, 'CODE_ROOT', project)
    monkeypatch.setattr(original, 'build_inventory', lambda *args, **kwargs: deepcopy(inventory))
    output = tmp_path / 'output'
    protocol = original.initialize(output, runs, metric_workers=1)
    monkeypatch.setattr(original, 'CODE_ROOT', output / 'source_snapshot')
    # Reporting itself is covered by the unchanged evaluator tests. Keep its
    # invocation visible while using tiny provenance-only synthetic results.
    reports = []
    monkeypatch.setattr(original.Controller, 'reports', lambda self, inv: reports.append(deepcopy(inv)))
    monkeypatch.setattr(original, 'training_needs_gpu', lambda *args: False)
    manifest = dict(gpus=[0, 1], scheduler_file_sha256='scheduler-digest',
                    evaluation_protocol_sha256=protocol['protocol_sha256'])
    controller = parallel.Controller(output, original, manifest,
        output / 'source_snapshot/scripts/run_segmentation_reevaluation.py')
    monkeypatch.setattr(controller, 'gpu_lock_path', lambda gpu: output / f'gpu{gpu}.lock')
    monkeypatch.setattr(controller, 'gpu_memory', lambda gpu: 40000)
    spawned = []

    def popen(command, **kwargs):
        child = Worker(command, **kwargs)
        spawned.append(child)
        return child

    monkeypatch.setattr(parallel.subprocess, 'Popen', popen)
    with parallel.durable_lock(output / '.controller.lock') as root_lock:
        controller.root_lock = root_lock
        env = SimpleNamespace(controller=controller, output=output, inventory=inventory, jobs=jobs,
                              protocol=protocol, spawned=spawned, reports=reports, monkeypatch=monkeypatch)
        yield env
        for item in controller.active.values():
            item['log'].close()
            item['lock'].close()
        controller.active.clear()


def complete(env, gpu, returncode=0):
    item = env.controller.active[gpu]
    item['process'].returncode = returncode
    if returncode:
        return
    folder = env.output / 'jobs' / item['job']['job_id']
    (folder / 'per_image.jsonl').write_text('{}\n')
    result = dict(complete=True, evaluation_protocol_sha256=env.protocol['protocol_sha256'],
                  job=item['job'], checkpoint_sha256=item['job']['checkpoint_sha256'],
                  per_image_sha256=original.file_sha(folder / 'per_image.jsonl'))
    result['result_sha256'] = original.digest(result)
    original.atomic_write_json(folder / 'result.json', result)


def test_two_distinct_workers_are_dispatched_without_waiting(environment):
    env, controller = environment, environment.controller
    assert controller.tick() is False
    assert set(controller.active) == {0, 1}
    assert len(env.spawned) == 2
    assert {item['job']['job_id'] for item in controller.active.values()} == {'job-0', 'job-1'}
    for gpu, item in controller.active.items():
        child = item['process']
        assert child.command[1] == str(controller.worker)
        assert child.command[-2:] == ['--device', 'cuda:0']
        assert child.kwargs['env']['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert child.kwargs['env']['PYTHONPATH'] == str(original.CODE_ROOT)
        assert set(child.kwargs['pass_fds']) == {item['lock'].fileno(), controller.root_lock.fileno()}
        assert child.kwargs['start_new_session'] is True
    controller.tick()
    assert len(env.spawned) == 2
    status = original.read(env.output / 'status.json')
    assert status['counts'] == {'running': 2, 'ready': 2}
    assert len(status['active_jobs']) == 2


def test_locked_gpu_zero_does_not_block_gpu_one(environment):
    env = environment
    with parallel.durable_lock(env.controller.gpu_lock_path(0)):
        env.controller.tick()
    assert set(env.controller.active) == {1}
    assert env.controller.observations['0']['state'] == 'waiting_gpu_lock'


def test_small_memory_rejected_without_using_same_card(environment):
    env = environment
    env.monkeypatch.setattr(env.controller, 'gpu_memory', lambda gpu: 17999 if gpu == 0 else 40000)
    env.controller.tick()
    assert set(env.controller.active) == {1}
    assert env.controller.observations['0']['state'] == 'waiting_gpu_memory'


def test_training_priority_does_not_interrupt_active_jobs(environment):
    env, controller = environment, environment.controller
    controller.tick()
    env.monkeypatch.setattr(original, 'training_needs_gpu', lambda *args: True)
    complete(env, 0)
    controller.tick()
    assert set(controller.active) == {1}
    assert len(env.spawned) == 2
    assert 'job-0' in controller.base.completed
    complete(env, 1)
    controller.tick()
    assert not controller.active
    assert original.read(env.output / 'status.json')['state'] == 'yielding_to_training'


def test_training_priority_rechecked_inside_gpu_lock(environment):
    env = environment
    decisions = iter([False, True, True])
    env.monkeypatch.setattr(original, 'training_needs_gpu', lambda *args: next(decisions))
    env.controller.tick()
    assert not env.spawned
    assert all(observation['state'] == 'yielding_to_training' for observation in env.controller.observations.values())


def test_completed_result_reused_and_freed_gpu_immediately_gets_next_job(environment):
    env, controller = environment, environment.controller
    controller.tick()
    complete(env, 0)
    result_path = env.output / 'jobs/job-0/result.json'
    before = result_path.read_bytes()
    controller.tick()
    assert controller.active[0]['job']['job_id'] == 'job-2'
    assert len(env.spawned) == 3
    assert result_path.read_bytes() == before
    assert 'job-0' in controller.base.completed
    controller.tick()
    assert len(env.spawned) == 3


def test_failure_persists_and_other_jobs_continue(environment):
    env, controller = environment, environment.controller
    controller.tick()
    complete(env, 0, returncode=7)
    controller.tick()
    assert controller.active[0]['job']['job_id'] == 'job-2'
    assert controller.base.failed['job-0']['exit_code'] == 7
    assert original.read(env.output / 'jobs/job-0/failure.json')['exit_code'] == 7
    complete(env, 1)
    controller.tick()
    complete(env, 0)
    complete(env, 1)
    assert controller.tick() is True
    assert original.read(env.output / 'status.json')['state'] == 'complete_with_failures'
    assert len(env.spawned) == 4


def test_success_without_result_persists_failure(environment):
    env, controller = environment, environment.controller
    controller.tick()
    controller.active[0]['process'].returncode = 0
    controller.tick()
    assert 'without a completed result' in controller.base.failed['job-0']['error']


def test_tampered_result_rejected_before_success(environment):
    env, controller = environment, environment.controller
    controller.tick()
    complete(env, 0)
    path = env.output / 'jobs/job-0/per_image.jsonl'
    path.write_text('tampered\n')
    with pytest.raises(ValueError, match='Unverified prior evaluation'):
        controller.reap()
    assert 'job-0' not in controller.base.completed
    assert 0 not in controller.active


def test_read_only_preview_never_writes_existing_outputs(environment):
    env = environment
    before = {p: p.read_bytes() for p in (env.output / 'status.json', env.output / 'inventory.json')}
    preview = env.controller.inspect()
    assert preview['read_only'] is True
    assert not env.spawned
    assert all(path.read_bytes() == content for path, content in before.items())


def test_once_cannot_overwrite_a_living_controller_status(environment):
    env = environment
    before = (env.output / 'status.json').read_bytes()
    with pytest.raises(BlockingIOError):
        env.controller.run(once=True)
    assert (env.output / 'status.json').read_bytes() == before
    assert not env.spawned


def test_once_refreshes_without_launching_after_lock_release(environment):
    env = environment
    env.controller.root_lock.close()
    assert env.controller.run(once=True) == 0
    assert not env.spawned
    assert env.reports


def test_deferred_jobs_do_not_launch_before_original_barrier(environment):
    env = environment
    for job in env.inventory['jobs']:
        job['status'] = 'deferred'
    assert env.controller.tick() is False
    assert not env.spawned
    assert original.read(env.output / 'status.json')['state'] == 'waiting_training_barriers'


def test_durable_parent_close_preserves_actual_child_lock(tmp_path):
    lock_path = tmp_path / 'inherited.lock'
    with parallel.durable_lock(lock_path) as handle:
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read(1)'],
                                 stdin=subprocess.PIPE, pass_fds=(handle.fileno(),))
    try:
        with pytest.raises(BlockingIOError):
            with parallel.durable_lock(lock_path):
                pass
    finally:
        child.communicate(b'x', timeout=10)
    with parallel.durable_lock(lock_path):
        pass


@pytest.fixture
def manifest_files(tmp_path, monkeypatch):
    script = tmp_path / 'scheduler/run_segmentation_reevaluation_parallel.py'
    script.parent.mkdir()
    script.write_text('frozen scheduler fixture')
    monkeypatch.setattr(parallel, '__file__', str(script))
    root = tmp_path / 'root'
    worker = root / 'source_snapshot/scripts/run_segmentation_reevaluation.py'
    worker.parent.mkdir(parents=True)
    worker.write_text('def load_protocol(root):\n    return {}\n')
    (root / 'protocol.json').write_text(json.dumps({'protocol_sha256': 'original-protocol'}))
    manifest = dict(scheduler_file_sha256=parallel.file_sha(script), worker_file_sha256=parallel.file_sha(worker),
                    evaluation_protocol_sha256='original-protocol', gpus=[0, 1])
    path = script.parent / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return SimpleNamespace(root=root, script=script, worker=worker, manifest=manifest, path=path)


def test_manifest_accepts_frozen_scheduler_and_original_worker(manifest_files):
    fixture = manifest_files
    module, manifest, worker = parallel.load_original(fixture.root, fixture.path)
    assert callable(module.load_protocol)
    assert worker == fixture.worker
    assert manifest == fixture.manifest


@pytest.mark.parametrize('target', ['script', 'worker', 'protocol'])
def test_manifest_tampering_is_rejected(manifest_files, target):
    fixture = manifest_files
    if target == 'protocol':
        (fixture.root / 'protocol.json').write_text(json.dumps({'protocol_sha256': 'changed'}))
    else:
        getattr(fixture, target).write_text('changed')
    with pytest.raises(ValueError):
        parallel.load_original(fixture.root, fixture.path)


def test_unfrozen_scheduler_path_rejected(manifest_files, tmp_path):
    fixture = manifest_files
    path = tmp_path / 'wrong_manifest.json'
    path.write_text(fixture.path.read_text())
    with pytest.raises(ValueError, match='beside its manifest'):
        parallel.load_original(fixture.root, path)


@pytest.mark.parametrize('gpus', [[], [0, 0], [-1], [1, 0]])
def test_gpu_configuration_must_match_frozen_manifest(environment, gpus):
    env = environment
    with pytest.raises(ValueError):
        parallel.Controller(env.output, original, env.controller.manifest, env.controller.worker, gpus=gpus)
