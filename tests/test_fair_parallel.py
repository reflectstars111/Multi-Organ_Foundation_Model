"""CPU-only checks for protocol isolation, the cross-GPU barrier, and aggregation."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import fair_parallel as parallel


@pytest.fixture
def plan(tmp_path):
    return parallel.build_plan(tmp_path / 'run', tmp_path / 'data')


def metadata_for(plan, gpu, tested=True):
    config = parallel.shard_config(plan, gpu)
    protocol = dict(config=config, config_sha256=parallel.digest(config),
                    record_sha256=parallel.digest([{'record': 'synthetic'}]),
                    source_files=plan['source_files'], source_sha256=plan['source_sha256'],
                    budget=plan['shards'][gpu]['budget'])
    histories, runs = {}, []
    for seed in parallel.SEEDS:
        for kind in plan['shards'][gpu]['models']:
            histories[kind, seed] = [dict(kind=kind, seed=seed, epoch=epoch, sample_count=4500, annotated_channel_queries=7000,
                                          sample_sha256=parallel.digest([seed, epoch]))
                                     for epoch in range(1, 51)]
            run = dict(kind=kind, seed=seed, epochs_completed=50, best_epoch=20,
                       best_validation_macro_positive_dice=.8 + (seed - 42) * .01,
                       parameter_count=100, elapsed_seconds=100., test_evaluated=tested,
                       checkpoint_directory=str(parallel.shard_path(plan, gpu) / kind / f'seed_{seed}'))
            if tested:
                run['test'] = dict(macro_positive_dice=.7 + (seed - 42) * .01,
                                   macro_empty_false_positive_rate=None, checkpoint_epoch=20,
                                   checkpoint_selection='best_validation_macro_positive_dice',
                                   config_sha256=protocol['config_sha256'],
                                   record_sha256=protocol['record_sha256'])
            runs.append(run)
    summary = dict(protocol_version=plan['common_config']['protocol_version'],
                   config_sha256=protocol['config_sha256'], record_sha256=protocol['record_sha256'],
                   dry_run=False, test_evaluation_requested=tested, runs=runs)
    return dict(protocol=protocol, summary=summary, histories=histories)


@pytest.fixture
def shards(plan):
    return {gpu: metadata_for(plan, gpu) for gpu in parallel.PARTITIONS}


def materialize(plan, gpu, metadata):
    folder = parallel.shard_path(plan, gpu)
    parallel.write_json(folder / 'protocol.json', metadata['protocol'])
    parallel.write_json(folder / 'records.json', [{'record': 'synthetic'}])
    parallel.write_json(folder / 'summary.json', metadata['summary'])
    for (kind, seed), rows in metadata['histories'].items():
        run_folder = folder / kind / f'seed_{seed}'
        run_folder.mkdir(parents=True, exist_ok=True)
        (run_folder / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        import torch
        run = next(run for run in metadata['summary']['runs'] if (run['kind'], run['seed']) == (kind, seed))
        checkpoint = dict(config=metadata['protocol']['config'],
                          config_sha256=metadata['protocol']['config_sha256'],
                          record_sha256=metadata['protocol']['record_sha256'],
                          source_sha256=metadata['protocol']['source_sha256'], kind=kind, seed=seed,
                          best_epoch=run['best_epoch'], best_score=run['best_validation_macro_positive_dice'],
                          history=rows)
        torch.save(dict(checkpoint, epoch=50), run_folder / 'last.pt')
        torch.save(dict(checkpoint, epoch=run['best_epoch']), run_folder / 'best.pt')
    for run in metadata['summary']['runs']:
        if run['test_evaluated']:
            parallel.write_json(folder / run['kind'] / f'seed_{run["seed"]}' / 'test_metrics.json', run['test'])


def test_plan_is_idempotent_and_refuses_changed_source_or_existing_artifacts(tmp_path, monkeypatch):
    output, data = tmp_path / 'run', tmp_path / 'data'
    plan = parallel.prepare_plan(output, data)
    assert parallel.prepare_plan(output, data) == plan
    assert not (output / 'shards').exists()
    changed = deepcopy(plan)
    changed['common_config']['batch_size'] = 2
    parallel.write_json(output / 'parallel_plan.json', changed)
    with pytest.raises(ValueError, match='differs'):
        parallel.prepare_plan(output, data)
    with pytest.raises(ValueError, match='hash mismatch'):
        parallel.load_plan(output, data)
    other = tmp_path / 'old_sequential_run'
    other.mkdir()
    (other / 'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='artifacts without'):
        parallel.prepare_plan(other, data)


def test_aggregation_preserves_fair_suite_shape_and_three_seed_statistics(plan, shards):
    result = parallel.aggregate_shards(plan, shards)
    assert len(result['runs']) == 9
    assert [(run['seed'], run['kind']) for run in result['runs']][:3] == [
        (42, 'shared'), (42, 'task_id'), (42, 'region')]
    for kind in parallel.MODELS:
        assert result['validation_across_seeds'][kind]['mean'] == pytest.approx(.81)
        assert result['validation_across_seeds'][kind]['sample_std'] == pytest.approx(.01)
        assert result['test_across_seeds'][kind]['macro_positive_dice']['seeds'] == 3
        assert result['test_across_seeds'][kind]['macro_positive_dice']['mean'] == pytest.approx(.71)
    assert result['parallel']['training_barrier_verified'] is True


@pytest.mark.parametrize('field', ['record_sha256', 'source_sha256', 'config_sha256', 'budget'])
def test_aggregation_rejects_mismatched_protocols(plan, shards, field):
    shards['1']['protocol'][field] = 'different'
    with pytest.raises(ValueError):
        parallel.aggregate_shards(plan, shards)


def test_aggregation_rejects_self_consistent_but_different_record_hash(plan, shards):
    protocol, summary = shards['1']['protocol'], shards['1']['summary']
    protocol['record_sha256'] = summary['record_sha256'] = parallel.digest(['different'])
    for run in summary['runs']:
        run['test']['record_sha256'] = protocol['record_sha256']
    with pytest.raises(ValueError, match='record hashes differ'):
        parallel.aggregate_shards(plan, shards)


@pytest.mark.parametrize('mutation,match', [
    (lambda data: data['summary']['runs'].pop(), 'exactly'),
    (lambda data: data['summary']['runs'].append(deepcopy(data['summary']['runs'][0])), 'exactly'),
    (lambda data: data['summary']['runs'][0].update(epochs_completed=49), 'all 50'),
    (lambda data: data['histories']['region', 42].pop(), 'all 50'),
    (lambda data: data['histories']['region', 42][0].update(sample_count=4499), 'sample count'),
    (lambda data: data['histories']['region', 42][0].update(sample_sha256='f' * 64), 'sample hashes'),
    (lambda data: data['histories']['region', 42][0].update(annotated_channel_queries=7001), 'annotated-channel'),
    (lambda data: data['summary']['runs'][0].update(test_evaluated=False), 'test evaluation'),
    (lambda data: data['summary']['runs'][0]['test'].update(checkpoint_epoch=49), 'selected checkpoint'),
])
def test_aggregation_rejects_incomplete_or_unfair_runs(plan, shards, mutation, match):
    mutation(shards['1'])
    with pytest.raises(ValueError, match=match):
        parallel.aggregate_shards(plan, shards)


def test_training_gate_rejects_tampered_completion_marker_and_uses_immutable_snapshot(plan):
    metadata = metadata_for(plan, '0', tested=False)
    materialize(plan, '0', metadata)
    folder = parallel.shard_path(plan, '0')
    parallel.write_json(folder / 'training_summary.json', metadata['summary'])
    marker = parallel.completion_marker(plan, '0', 'training', metadata)
    parallel.write_json(parallel.marker_path(plan, '0', 'training'), marker)
    # fair_suite progressively replaces summary during test; training evidence is stable.
    parallel.write_json(folder / 'summary.json', {'runs': []})
    assert parallel.validate_marker(plan, '0', 'training')['summary'] == metadata['summary']
    marker['plan_sha256'] = 'stale'
    parallel.write_json(parallel.marker_path(plan, '0', 'training'), marker)
    with pytest.raises(ValueError, match='stale'):
        parallel.validate_marker(plan, '0', 'training')
    marker['plan_sha256'] = plan['plan_sha256']
    parallel.write_json(parallel.marker_path(plan, '0', 'training'), marker)
    metadata['histories']['shared', 42][0]['sample_count'] = 1
    materialize(plan, '0', metadata)
    with pytest.raises(ValueError, match='sample count'):
        parallel.validate_marker(plan, '0', 'training')


def test_gpu_admission_preserves_existing_jobs_and_requires_22_gib(monkeypatch):
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return ('0, GPU-uuid, RTX A6000, 49140, 4500, 44640\n'
                if '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free' in command
                else '12345, another_project, 4200\n')
    monkeypatch.setattr(parallel.subprocess, 'check_output', query)
    result = parallel.gpu_snapshot(0)
    assert result['shared_with_existing_work']
    assert result['free_mib'] > 22 * 1024
    assert all(command[0] == 'nvidia-smi' for command in calls)
    monkeypatch.setattr(parallel.subprocess, 'check_output', lambda command, **kwargs:
                        '0, GPU-uuid, RTX A6000, 49140, 30000, 19140\n' if '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free' in command else '')
    with pytest.raises(RuntimeError, match='existing jobs were left untouched'):
        parallel.gpu_snapshot(0)


def test_gpu_lock_refuses_duplicate_controller_without_mutating_artifacts(tmp_path):
    path = tmp_path / 'gpu.lock'
    with parallel.exclusive_lock(path):
        with pytest.raises(RuntimeError, match='already owns'):
            with parallel.exclusive_lock(path):
                pytest.fail('Duplicate controller entered the GPU lock')


def test_failure_marker_is_reported_while_waiting(plan, tmp_path):
    with (tmp_path / 'lock').open('a+') as lock:
        worker = parallel.Worker(plan, 0, lock)
        parallel.write_json(worker.failure_path('1'), dict(plan_sha256=plan['plan_sha256'], error='OOM'))
        with pytest.raises(RuntimeError, match='Peer GPU 1 failed: OOM'):
            worker.wait_for_peer('training')


def test_two_workers_wait_for_all_training_before_testing_and_restart_without_retraining(plan, monkeypatch, tmp_path):
    calls, errors = [], []
    monkeypatch.setattr(parallel, 'gpu_snapshot', lambda gpu: dict(free_mib=43000, used_mib=4000))
    actual_sleep = time.sleep
    monkeypatch.setattr(parallel, 'time', SimpleNamespace(sleep=lambda seconds: actual_sleep(.01),
                                                         time_ns=time.time_ns))
    def fake_suite(worker, evaluate_test=False):
        if evaluate_test:
            assert all(parallel.marker_path(plan, gpu, 'training').exists() for gpu in parallel.PARTITIONS)
            assert sum(not tested for _, tested in calls) == 2
        calls.append((worker.gpu, evaluate_test))
        materialize(plan, worker.gpu, metadata_for(plan, worker.gpu, evaluate_test))
    monkeypatch.setattr(parallel.Worker, 'run_suite', fake_suite)
    def run_worker(gpu):
        try:
            with (tmp_path / f'gpu{gpu}.lock').open('a+') as lock:
                parallel.Worker(plan, gpu, lock).run()
        except BaseException as error:
            errors.append(error)
    first = threading.Thread(target=run_worker, args=(0,), daemon=True)
    first.start()
    status_path = Path(plan['output']) / 'worker_gpu0.json'
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if status_path.exists() and parallel.read_json(status_path)['state'] == 'waiting_for_peer':
            break
        actual_sleep(.01)
    else:
        pytest.fail(f'GPU 0 never entered waiting_for_peer: {errors}')
    assert calls == [('0', False)]
    second = threading.Thread(target=run_worker, args=(1,), daemon=True)
    second.start()
    first.join(5)
    second.join(5)
    assert not errors
    assert not first.is_alive() and not second.is_alive()
    summary = parallel.read_json(Path(plan['output']) / 'summary.json')
    assert len(summary['runs']) == 9
    assert all(run['test_evaluated'] for run in summary['runs'])
    original_calls = list(calls)
    run_worker(0)
    assert not errors
    assert calls == original_calls


@pytest.mark.parametrize('filename,change', [
    ('last.pt', {'epoch': 49}),
    ('best.pt', {'epoch': 19}),
    ('last.pt', {'source_sha256': 'different'}),
    ('last.pt', {'seed': 43}),
])
def test_checkpoint_metadata_must_match_completed_summary(plan, filename, change):
    import torch
    metadata = metadata_for(plan, '1', tested=False)
    materialize(plan, '1', metadata)
    checkpoint_path = parallel.shard_path(plan, '1') / 'region' / 'seed_42' / filename
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    checkpoint.update(change)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='checkpoint'):
        parallel.load_shard(plan, '1')


def test_peer_failure_does_not_create_cascading_local_failure(plan, monkeypatch, tmp_path):
    monkeypatch.setattr(parallel, 'load_plan', lambda *args: plan)
    original_lock = parallel.exclusive_lock
    monkeypatch.setattr(parallel, 'exclusive_lock', lambda path: original_lock(tmp_path / 'test_only.lock'))
    def blocked(worker):
        raise parallel.PeerFailedError('Peer GPU 1 failed: recover the peer first')
    monkeypatch.setattr(parallel.Worker, 'run', blocked)
    with pytest.raises(parallel.PeerFailedError, match='recover the peer'):
        parallel.main(['--output', plan['output'], '--gpu', '0'])
    assert not (Path(plan['output']) / 'failed_gpu0.json').exists()
    assert parallel.read_json(Path(plan['output']) / 'worker_gpu0.json')['state'] == 'blocked_by_peer'
