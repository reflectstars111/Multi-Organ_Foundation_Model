"""CPU checks for deferred 80-epoch union scoring and artifact reuse."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import defer_union_evaluation_80 as queue
from scripts import defer_single_channel_80 as single_queue


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _prerequisites(tmp_path, monkeypatch):
    fair = tmp_path / 'fair80'
    single = tmp_path / 'single80'
    status = tmp_path / 'single80-logs/controller_status.json'
    record_hash = queue.evaluator.EXPECTED_RECORD_SHA256
    fair_summary = dict(record_sha256=record_hash, source_sha256='source80')
    _json(fair / 'summary.json', fair_summary)
    _json(fair / 'parallel_plan.json', {'plan_sha256': 'plan80'})
    _json(single / 'protocol.json', dict(config_sha256='single-config',
        record_sha256=record_hash, reference_records_sha256=record_hash,
        config=single_queue.EXPECTED_SINGLE_CONFIG))
    _json(single / 'summary.json', dict(protocol_version='single_channel_paper5_union_v1',
        config_sha256='single-config', record_sha256=record_hash,
        test_evaluation_requested=True,
        runs=[dict(seed=seed, epochs_completed=80, test_evaluated=True) for seed in (42, 43, 44)]))
    for seed in (42, 43, 44):
        _json(single / f'seed_{seed}/test_metrics.json', {'seed': seed})
    _json(status, {'state': 'complete', 'summary': str(single / 'summary.json')})
    monkeypatch.setattr(queue.evaluator, 'load_completed_fair', lambda _: ({}, fair_summary, []))
    return fair, single, status


def _union_output(output, fair):
    protocol = dict(schema=queue.evaluator.SCHEMA, split='test', image_size=512,
                    images=2, fair_root=str(fair.resolve()),
                    fair_record_sha256=queue.evaluator.EXPECTED_RECORD_SHA256,
                    fair_source_sha256='source80', fair_plan_sha256='plan80')
    for key, module in [('evaluator_sha256', queue.evaluator),
                        ('shared_scoring_sha256', queue.evaluator.common),
                        ('target_source_sha256', queue.evaluator.single_channel_suite)]:
        protocol[key] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    overall = dict(mean_dice=.8, positive_dice=.8, empty_false_positive_rate=None,
                   empty_mean_false_positive_area_fraction=None,
                   positive_miss_rate=0., positive_zero_overlap_rate=0., samples=2)
    metrics = dict(overall=overall, per_domain={'BUSI': overall}, abdomen_by_ct={},
                   macro_domain_positive_dice=.8)
    runs = []
    for kind, seed in sorted(queue.EXPECTED):
        run = dict(kind=kind, seed=seed, checkpoint='best.pt', checkpoint_epoch=17,
                   metrics=metrics)
        runs.append(run)
        _json(output / f'{kind}_seed_{seed}.json', dict(run, per_image=[{}, {}]))
    _json(output / 'protocol.json', protocol)
    _json(output / 'summary.json', dict(protocol=protocol, runs=runs,
        across_seeds=queue.evaluator.common.summarize_seeds(runs)))


def test_waits_for_both_complete_protocols_and_rejects_unpaired_single(tmp_path, monkeypatch):
    fair, single, status = _prerequisites(tmp_path, monkeypatch)
    research = tmp_path / 'fair80-status.json'
    monkeypatch.setattr(queue, 'fair_state', lambda *_: 'waiting')
    assert queue.prerequisite_state(fair, research, status, single) == 'waiting_fair'
    monkeypatch.setattr(queue, 'fair_state', lambda *_: 'ready')
    _json(status, {'state': 'training'})
    assert queue.prerequisite_state(fair, research, status, single) == 'waiting_single'
    _json(status, {'state': 'complete', 'summary': str(single / 'summary.json')})
    assert queue.prerequisite_state(fair, research, status, single) == 'ready'
    protocol = json.loads((single / 'protocol.json').read_text())
    protocol['reference_records_sha256'] = 'wrong'
    _json(single / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='not paired'):
        queue.prerequisite_state(fair, research, status, single)


def test_failed_prerequisite_is_reported_without_inference(tmp_path, monkeypatch):
    fair, single, status = _prerequisites(tmp_path, monkeypatch)
    research = tmp_path / 'fair80-status.json'
    monkeypatch.setattr(queue, 'fair_state', lambda *_: 'waiting')
    _json(research, {'state': 'failed', 'error': 'GPU preflight failed'})
    with pytest.raises(RuntimeError, match='GPU preflight failed'):
        queue.prerequisite_state(fair, research, status, single)
    _json(research, {'state': 'complete'})
    monkeypatch.setattr(queue, 'fair_state', lambda *_: 'ready')
    _json(status, {'state': 'failed', 'error': 'training failed'})
    with pytest.raises(RuntimeError, match='single-channel controller failed'):
        queue.prerequisite_state(fair, research, status, single)


def test_reuse_requires_matching_code_and_complete_per_image_results(tmp_path, monkeypatch):
    fair, _, _ = _prerequisites(tmp_path, monkeypatch)
    output = tmp_path / 'union80'
    assert not queue.completed_output(output, fair)
    _union_output(output, fair)
    assert queue.completed_output(output, fair)
    detail_path = output / 'region_seed_44.json'
    detail = json.loads(detail_path.read_text())
    detail['per_image'].pop()
    _json(detail_path, detail)
    with pytest.raises(ValueError, match='differs from its summary'):
        queue.completed_output(output, fair)
    _union_output(output, fair)
    protocol_path = output / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['fair_plan_sha256'] = 'old50-plan'
    _json(protocol_path, protocol)
    with pytest.raises(ValueError, match='mismatched'):
        queue.completed_output(output, fair)


def test_controller_obeys_memory_gate_lock_and_inference_only_command(tmp_path, monkeypatch):
    fair, single, status = _prerequisites(tmp_path, monkeypatch)
    output = tmp_path / 'union80'
    monkeypatch.setattr(queue, 'prerequisite_state', lambda *_: 'ready')
    monkeypatch.setattr(queue, 'gpu_lock_path', lambda _: tmp_path / 'gpu0.lock')
    memory = iter((queue.MINIMUM_FREE_MIB - 1, queue.MINIMUM_FREE_MIB))
    monkeypatch.setattr(queue, 'gpu_free_mib', lambda _: next(memory))
    sleeps = []
    monkeypatch.setattr(queue.time, 'sleep', lambda value: sleeps.append(value))
    calls = []

    def command(command, log, environment, lock):
        calls.append(command)
        assert command[:3] == [queue.sys.executable, '-m', 'unet_moe.union_evaluate_80']
        assert environment['CUDA_VISIBLE_DEVICES'] == '0'
        assert lock.fileno() >= 0
        for flag, expected in [('--split', 'test'), ('--workers', '0'),
                               ('--batch-size', '2'), ('--device', 'cuda:0')]:
            assert command[command.index(flag) + 1] == expected
        _union_output(output, fair)
        return 0

    monkeypatch.setattr(queue, 'run_command', command)
    queue.main(['--fair-root', str(fair), '--research-status', str(tmp_path / 'fair-status'),
                '--single-status', str(status), '--single-output', str(single),
                '--output', str(output), '--poll-seconds', '1'])
    assert sleeps == [1] and len(calls) == 1
    controller = json.loads((tmp_path / 'union80-logs/controller_status.json').read_text())
    assert controller['state'] == 'complete' and controller['gpu'] == 0
