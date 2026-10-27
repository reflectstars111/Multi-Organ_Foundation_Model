"""Retrospective evaluation isolation, durable recovery and GPU admission."""

from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_segmentation_reevaluation as runner
from scripts.medical_priority_process import atomic_write_json, exclusive_lock
from unet_moe import reevaluation_inference as inference


@pytest.fixture
def environment(tmp_path, monkeypatch):
    project = tmp_path / 'checkout'
    for folder in ('scripts', 'unet_moe'):
        (project / folder).mkdir(parents=True)
        (project / folder / '__init__.py').write_text('"""Frozen fixture package."""\n')
    (project / 'scripts/evaluator.py').write_text('FROZEN_VALUE = 1\n')
    runs = tmp_path / 'runs'
    runs.mkdir()
    checkpoint = runs / 'best.pt'
    checkpoint.write_bytes(b'original immutable checkpoint fixture')
    records = runs / 'records.json'
    atomic_write_json(records, [{'id': 'image-1', 'split': 'val'}])
    job = dict(job_id='job-a', family='fair', experiment='completed_fair', kind='shared', arm=None,
               seed=42, selection='best', split='val', size=512, status='ready',
               checkpoint=str(checkpoint), records_path=str(records), config={'size': 512})
    inventory = {'jobs': [job], 'counts': {'ready': 1}}
    monkeypatch.setattr(runner, 'CODE_ROOT', project)
    monkeypatch.setattr(runner, 'build_inventory', lambda *args, **kwargs: deepcopy(inventory))
    output = tmp_path / 'evaluation'
    protocol = runner.initialize(output, runs, metric_workers=2)
    frozen = output / 'source_snapshot'
    monkeypatch.setattr(runner, 'CODE_ROOT', frozen)
    folder = output / 'jobs/job-a'
    folder.mkdir()
    manifest = folder / 'job.json'
    dispatched = dict(job, checkpoint_sha256=runner.file_sha(checkpoint),
                      evaluation_protocol_sha256=protocol['protocol_sha256'])
    dispatched['job_manifest_sha256'] = runner.digest(dispatched)
    atomic_write_json(manifest, dispatched)
    return SimpleNamespace(project=project, runs=runs, checkpoint=checkpoint, records=records,
                           output=output, protocol=protocol, folder=folder, manifest=manifest,
                           inventory=inventory, job=job, frozen=frozen)


def sample(index=1):
    target = np.zeros((9, 9), dtype=np.int8)
    target[3:6, 3:6] = 1
    return inference.EvaluationSample(image_id=f'image-{index}', domain='BUSI',
                                     structure='breast_lesion', task_id=0,
                                     target=target, prediction=target.astype(bool), view='structure')


def fake_session(monkeypatch, samples):
    monkeypatch.setattr(inference, 'load_evaluation', lambda *args, **kwargs:
                        inference.EvaluationSession(metadata={'fixture': True}, samples=iter(samples)))


def complete_job(environment, monkeypatch):
    fake_session(monkeypatch, [sample(1), sample(2)])
    return runner.evaluate_job(environment.output, environment.manifest)


def test_initialization_freezes_source_and_rejects_overwrite(environment):
    copied = environment.frozen / 'scripts/evaluator.py'
    assert copied.read_text() == 'FROZEN_VALUE = 1\n'
    (environment.project / 'scripts/evaluator.py').write_text('FROZEN_VALUE = 2\n')
    assert runner.load_protocol(environment.output)['source_sha256'] == environment.protocol['source_sha256']
    assert copied.read_text() == 'FROZEN_VALUE = 1\n'
    with pytest.raises(FileExistsError, match='already exists'):
        runner.initialize(environment.output, environment.runs)


def test_changed_frozen_source_is_rejected(environment):
    (environment.frozen / 'scripts/evaluator.py').write_text('FROZEN_VALUE = 9\n')
    with pytest.raises(ValueError, match='source changed'):
        runner.load_protocol(environment.output)


def test_mutable_checkout_cannot_run_the_frozen_protocol(environment, monkeypatch):
    monkeypatch.setattr(runner, 'CODE_ROOT', environment.project)
    with pytest.raises(ValueError, match='frozen evaluator'):
        runner.load_protocol(environment.output)


def test_protocol_tampering_is_rejected(environment):
    protocol = runner.read(environment.output / 'protocol.json')
    protocol['threshold'] = 0.9
    atomic_write_json(environment.output / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='protocol changed'):
        runner.load_protocol(environment.output)


def test_initial_inventory_hash_is_verified_on_resume(environment):
    inventory = runner.read(environment.output / 'inventory_initial.json')
    inventory['jobs'][0]['split'] = 'test'
    atomic_write_json(environment.output / 'inventory_initial.json', inventory)
    with pytest.raises(ValueError, match='[Ii]nventory'):
        runner.load_protocol(environment.output)


def test_changed_checkpoint_is_rejected_before_inference(environment, monkeypatch):
    environment.checkpoint.write_bytes(b'changed after dispatch')
    monkeypatch.setattr(inference, 'load_evaluation', lambda *args, **kwargs: pytest.fail('inference must not run'))
    with pytest.raises(ValueError, match='Checkpoint changed'):
        runner.evaluate_job(environment.output, environment.manifest)
    assert not (environment.folder / 'result.json').exists()


def test_changed_checkpoint_during_inference_never_completes(environment, monkeypatch):
    def changing_samples():
        yield sample()
        environment.checkpoint.write_bytes(b'changed during evaluation')

    fake_session(monkeypatch, changing_samples())
    with pytest.raises(ValueError, match='Checkpoint changed'):
        runner.evaluate_job(environment.output, environment.manifest)
    assert not (environment.folder / 'result.json').exists()
    assert not (environment.folder / '.per_image.jsonl.tmp').exists()


def test_current_original_test_barrier_is_checked_before_inference(environment, monkeypatch):
    environment.inventory['jobs'][0]['status'] = 'deferred'
    monkeypatch.setattr(inference, 'load_evaluation', lambda *args, **kwargs: pytest.fail('inference must not run'))
    with pytest.raises(ValueError, match='barrier'):
        runner.evaluate_job(environment.output, environment.manifest)
    assert not (environment.folder / 'result.json').exists()


def test_job_manifest_corruption_is_rejected(environment):
    job = runner.read(environment.manifest)
    job['split'] = 'test'
    atomic_write_json(environment.manifest, job)
    with pytest.raises(ValueError, match='manifest changed'):
        runner.evaluate_job(environment.output, environment.manifest)


def test_bounded_scores_does_not_eagerly_materialize_mask_iterator(monkeypatch):
    release, window_filled = threading.Event(), threading.Event()
    produced, result, errors = [], [], []

    def samples():
        for index in range(20):
            produced.append(index)
            if len(produced) == 4:
                window_filled.set()
            yield index

    def score(index, policy):
        if not release.wait(timeout=5):
            raise TimeoutError('test worker was not released')
        return index

    monkeypatch.setattr(runner, 'score_sample', score)

    def consume():
        try:
            result.extend(runner.bounded_scores(samples(), workers=2, empty_policy='diagonal'))
        except BaseException as error:
            errors.append(error)

    consumer = threading.Thread(target=consume)
    consumer.start()
    try:
        assert window_filled.wait(timeout=3)
        assert produced == [0, 1, 2, 3]
        assert result == []
    finally:
        release.set()
        consumer.join(timeout=5)
    assert not consumer.is_alive()
    assert not errors
    assert result == list(range(20))


@pytest.mark.parametrize('failure', ['iterator', 'duplicate', 'empty', 'sample_limit'])
def test_incomplete_evaluation_never_writes_result(environment, monkeypatch, failure):
    def broken():
        yield sample()
        raise RuntimeError('simulated inference failure')

    samples = {'iterator': broken(), 'duplicate': [sample(), sample()],
               'empty': [], 'sample_limit': [sample(), sample(2)]}[failure]
    fake_session(monkeypatch, samples)
    with pytest.raises((RuntimeError, ValueError)):
        runner.evaluate_job(environment.output, environment.manifest,
                            limit_samples=1 if failure == 'sample_limit' else None)
    assert not (environment.folder / 'result.json').exists()
    assert not (environment.folder / '.per_image.jsonl.tmp').exists()


def test_completed_result_is_atomic_hashed_and_never_overwritten(environment, monkeypatch):
    original = environment.checkpoint.read_bytes()
    result = complete_job(environment, monkeypatch)
    assert result['complete'] is True
    assert result['rows'] == 2
    assert result['per_image_sha256'] == runner.file_sha(environment.folder / 'per_image.jsonl')
    assert runner.read(environment.folder / 'progress.json')['state'] == 'complete'
    assert environment.checkpoint.read_bytes() == original
    persisted = (environment.folder / 'result.json').read_bytes()
    with pytest.raises(FileExistsError, match='refuse to overwrite'):
        runner.evaluate_job(environment.output, environment.manifest)
    assert (environment.folder / 'result.json').read_bytes() == persisted


def test_declared_split_image_count_must_match_actual_unique_images(environment, monkeypatch):
    monkeypatch.setattr(inference, 'load_evaluation', lambda *args, **kwargs:
                        inference.EvaluationSession(metadata={'images': 3}, samples=iter([sample(1), sample(2)])))
    with pytest.raises(ValueError, match='[Ii]mage|split'):
        runner.evaluate_job(environment.output, environment.manifest)
    assert not (environment.folder / 'result.json').exists()
    assert not (environment.folder / '.per_image.jsonl.tmp').exists()


def test_declared_union_view_must_cover_every_split_image(environment, monkeypatch):
    union = sample(1)
    union.view, union.structure, union.task_id = 'union', None, None
    monkeypatch.setattr(inference, 'load_evaluation', lambda *args, **kwargs:
                        inference.EvaluationSession(metadata={'images': 2, 'views': ['structure', 'union']},
                                                    samples=iter([sample(1), sample(2), union])))
    with pytest.raises(ValueError, match='[Uu]nion'):
        runner.evaluate_job(environment.output, environment.manifest)
    assert not (environment.folder / 'result.json').exists()
    assert not (environment.folder / '.per_image.jsonl.tmp').exists()


def test_absent_declared_image_count_does_not_compare_union_count_with_none(environment, monkeypatch):
    union = sample(1)
    union.view, union.structure, union.task_id = 'union', None, None
    monkeypatch.setattr(inference, 'load_evaluation', lambda *args, **kwargs:
                        inference.EvaluationSession(metadata={'views': ['union']}, samples=iter([union])))
    assert runner.evaluate_job(environment.output, environment.manifest)['complete'] is True


def test_resume_verifies_per_image_artifact(environment, monkeypatch):
    complete_job(environment, monkeypatch)
    (environment.folder / 'per_image.jsonl').write_text('{}\n')
    controller = runner.Controller(environment.output)
    with pytest.raises(ValueError, match='Unverified'):
        controller.refresh()


def test_resume_rejects_changed_checkpoint_after_previous_completion(environment, monkeypatch):
    complete_job(environment, monkeypatch)
    environment.checkpoint.write_bytes(b'checkpoint changed after completed evaluation')
    controller = runner.Controller(environment.output)
    with pytest.raises(ValueError, match='[Cc]heckpoint|[Uu]nverified'):
        controller.refresh()


def test_resume_rejects_changed_embedded_job_manifest(environment, monkeypatch):
    complete_job(environment, monkeypatch)
    result = runner.read(environment.folder / 'result.json')
    result['job']['split'] = 'test'
    atomic_write_json(environment.folder / 'result.json', result)
    controller = runner.Controller(environment.output)
    with pytest.raises(ValueError, match='[Mm]anifest|[Uu]nverified'):
        controller.refresh()


def test_resume_rejects_inconsistent_checkpoint_identity(environment, monkeypatch):
    complete_job(environment, monkeypatch)
    result = runner.read(environment.folder / 'result.json')
    result['checkpoint_sha256'] = '0' * 64
    atomic_write_json(environment.folder / 'result.json', result)
    controller = runner.Controller(environment.output)
    with pytest.raises(ValueError, match='[Cc]heckpoint|[Uu]nverified'):
        controller.refresh()


def test_valid_resume_recovers_completed_result_without_inference(environment, monkeypatch):
    result = complete_job(environment, monkeypatch)
    monkeypatch.setattr(inference, 'load_evaluation', lambda *args, **kwargs: pytest.fail('must reuse verified result'))
    controller = runner.Controller(environment.output)
    inventory = controller.refresh()
    assert inventory['evaluation_counts'] == {'complete': 1}
    assert controller.completed['job-a'] == result


@pytest.mark.parametrize('field', ['protocol_sha256', 'records_sha256'])
def test_completed_result_cannot_override_changed_source_provenance(environment, monkeypatch, field):
    complete_job(environment, monkeypatch)
    environment.inventory['jobs'][0][field] = 'changed-live-experiment-digest'
    controller = runner.Controller(environment.output)
    with pytest.raises(ValueError, match='[Uu]nverified|[Pp]rovenance|[Cc]hanged'):
        controller.refresh()


@pytest.mark.parametrize('exit_code', [0, 3])
def test_worker_exit_without_result_cannot_complete_job(environment, monkeypatch, exit_code):
    controller = runner.Controller(environment.output)
    captured = {}

    class Process:
        pid = 111111

        def wait(self):
            return exit_code

    def launch(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Process()

    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    with exclusive_lock(environment.output / '.controller.lock') as root_lock:
        with exclusive_lock(environment.output / 'fixture_gpu.lock') as gpu_lock:
            controller.root_lock = root_lock
            if exit_code == 0:
                with pytest.raises(RuntimeError, match='without a completed result'):
                    controller.dispatch(environment.job, gpu_lock.fileno())
            else:
                controller.dispatch(environment.job, gpu_lock.fileno())
            assert set(captured['pass_fds']) == {root_lock.fileno(), gpu_lock.fileno()}
    assert captured['env']['CUDA_VISIBLE_DEVICES'] == str(controller.gpu)
    assert not (environment.folder / 'result.json').exists()
    if exit_code:
        assert runner.read(environment.folder / 'failure.json')['exit_code'] == exit_code
        assert controller.refresh()['evaluation_counts'] == {'failed': 1}


def controller_one_turn(environment, monkeypatch):
    controller = runner.Controller(environment.output, gpu=12345, poll_seconds=1)
    states = []
    real_update = controller.update

    def update(state, **extra):
        states.append(state)
        return real_update(state, **extra)

    def stop_after_one_iteration(*args):
        raise InterruptedError('test one-turn stop')

    monkeypatch.setattr(controller, 'update', update)
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(runner.time, 'sleep', stop_after_one_iteration)
    monkeypatch.setattr(controller, 'dispatch', lambda *args: pytest.fail('GPU worker must not be dispatched'))
    return controller, states


def test_pending_training_has_priority_before_gpu_admission(environment, monkeypatch):
    controller, states = controller_one_turn(environment, monkeypatch)
    monkeypatch.setattr(runner, 'training_needs_gpu', lambda *args: True)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs:
                        pytest.fail('GPU query must not run while training has priority'))
    with pytest.raises(InterruptedError):
        controller.run()
    assert 'yielding_to_training' in states
    assert 'evaluating' not in states


def test_training_gpu_lock_blocks_evaluation_with_real_file_lock(environment, monkeypatch):
    controller, states = controller_one_turn(environment, monkeypatch)
    gpu_lock = environment.output / 'fixture_gpu.lock'

    def redirected_lock(path):
        return exclusive_lock(gpu_lock if str(path).startswith('/tmp/multi_organ_fair_gpu_') else path)

    monkeypatch.setattr(runner, 'exclusive_lock', redirected_lock)
    monkeypatch.setattr(runner, 'training_needs_gpu', lambda *args: False)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs:
                        pytest.fail('GPU query must not run while training owns its lock'))
    with exclusive_lock(gpu_lock):
        with pytest.raises(InterruptedError):
            controller.run()
    assert 'waiting_gpu_lock' in states
    assert 'evaluating' not in states


def test_training_priority_is_rechecked_after_acquiring_gpu_lock(environment, monkeypatch):
    controller, states = controller_one_turn(environment, monkeypatch)
    decisions = iter([False, True])
    gpu_lock = environment.output / 'fixture_gpu.lock'
    monkeypatch.setattr(runner, 'training_needs_gpu', lambda *args: next(decisions))
    monkeypatch.setattr(runner, 'exclusive_lock', lambda path:
                        exclusive_lock(gpu_lock if str(path).startswith('/tmp/multi_organ_fair_gpu_') else path))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: '30000\n')
    with pytest.raises(InterruptedError):
        controller.run()
    assert 'yielding_to_training' in states
    assert 'evaluating' not in states


def test_failed_job_does_not_turn_into_successfully_complete_queue(environment, monkeypatch):
    atomic_write_json(environment.folder / 'failure.json', {'exit_code': 2, 'message': 'inference failed'})
    controller = runner.Controller(environment.output)
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    controller.run()
    status = runner.read(environment.output / 'status.json')
    assert status['state'] == 'complete_with_failures'
    assert status['failed_jobs'] == 1
    assert status['completed_jobs'] == 0
    assert not (environment.folder / 'result.json').exists()


def test_disappeared_scoped_job_is_preserved_as_unavailable(environment, monkeypatch):
    environment.inventory['jobs'].clear()
    controller = runner.Controller(environment.output)
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    controller.run()
    status = runner.read(environment.output / 'status.json')
    inventory = runner.read(environment.output / 'inventory.json')
    assert status['state'] == 'complete_with_unavailable'
    assert status['completed_jobs'] == 0
    assert inventory['evaluation_counts'] == {'unavailable': 1}
    assert inventory['jobs'][0]['job_id'] == 'job-a'


@pytest.mark.parametrize('state,phase,expected', [
    ('recovering', 'train', True),
    ('preflight', 'train', True),
    ('training', 'train', True),
    ('evaluating_test', 'test', True),
    ('waiting_reference', 'train', False),
    ('complete', 'train', False),
])
def test_live_training_controller_pending_work_priority(environment, monkeypatch, state, phase, expected):
    path = environment.runs / 'q01_organ_weighted_80_512_20261007/controller_status.json'
    path.parent.mkdir()
    atomic_write_json(path, {'controller_pid': 123, 'state': state, 'jobs': [{phase + '_state': 'pending'}]})
    monkeypatch.setattr(runner, 'process_identity', lambda pid: {'pid': pid, 'starttime': 7})
    monkeypatch.setattr(runner, 'process_running', lambda identity: True)
    assert runner.training_needs_gpu(environment.runs) is expected


def test_stale_dead_controller_status_does_not_hold_gpu_priority(environment, monkeypatch):
    path = environment.runs / 'q01_organ_weighted_80_512_20261007/controller_status.json'
    path.parent.mkdir()
    atomic_write_json(path, {'controller_pid': 123, 'state': 'training', 'jobs': [{'train_state': 'pending'}]})
    monkeypatch.setattr(runner, 'process_identity', lambda pid: None)
    assert runner.training_needs_gpu(environment.runs) is False
