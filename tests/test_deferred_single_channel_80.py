"""CPU-only checks for the 80-epoch union U-Net queue."""

import json

import pytest

from scripts import defer_single_channel_80 as queue


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _fair(tmp_path):
    fair = tmp_path / 'fair80'
    _json(fair / 'summary.json', {
        'record_sha256': 'paired-records',
        'runs': [{'kind': kind, 'seed': seed} for kind in queue.FAIR_MODELS
                 for seed in queue.SEEDS],
    })
    _json(fair / 'shards/gpu1/protocol.json', {
        'record_sha256': 'paired-records',
        'config': dict(queue.EXPECTED_FAIR_CONFIG, models=['region']),
    })
    _json(fair / 'shards/gpu1/records.json', [])
    return fair


def _single_output(output, fair_hash='paired-records', epochs=80):
    _json(output / 'protocol.json', {
        'config_sha256': 'single-config', 'record_sha256': fair_hash,
        'reference_records_sha256': fair_hash,
        'config': queue.EXPECTED_SINGLE_CONFIG,
    })
    runs = []
    for seed in queue.SEEDS:
        _json(output / f'seed_{seed}/test_metrics.json', {'seed': seed})
        runs.append({'seed': seed, 'epochs_completed': epochs, 'test_evaluated': True})
    _json(output / 'summary.json', {
        'protocol_version': 'single_channel_paper5_union_v1',
        'config_sha256': 'single-config', 'record_sha256': fair_hash,
        'test_evaluation_requested': True, 'runs': runs,
    })


def test_queue_uses_identical_80_epoch_budget_after_fair_barrier(tmp_path, monkeypatch):
    fair = _fair(tmp_path)
    output = tmp_path / 'single80'
    states = iter(('waiting', 'ready'))
    calls = []
    available = iter((queue.MINIMUM_FREE_MIB - 1, queue.MINIMUM_FREE_MIB,
                      queue.MINIMUM_FREE_MIB))
    monkeypatch.setattr(queue, 'fair_state', lambda *_: next(states))
    monkeypatch.setattr(queue, 'gpu_lock_path', lambda _: tmp_path / 'gpu0.lock')
    monkeypatch.setattr(queue, 'gpu_free_mib', lambda _: next(available))
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)

    def fake_run(command, log, environment, lock=None):
        calls.append((command, environment, lock))
        if 'scripts.preflight_single80' in command:
            _json(tmp_path / 'single80-logs/gpu_preflight.json', {
                'schema': 'single80_physical_batch_preflight_v1',
                'record_sha256': 'paired-records', 'status': 'passed',
                'optimizer_step_ok': True, 'optimizer_steps': 2, 'gpu_total_mib': 49140,
                'physical_batch_size': 16,
                'peak_reserved_mib': 30000,
            })
        if '--evaluate-test' in command:
            _single_output(output)
        return 0

    monkeypatch.setattr(queue, 'run_command', fake_run)
    queue.main(['--fair-root', str(fair),
                '--research-status', str(tmp_path / 'fair80-controller.json'),
                '--data-root', str(tmp_path / 'data'), '--output', str(output),
                '--poll-seconds', '1'])

    assert len(calls) == 4
    prepare, probe, train, test = (row[0] for row in calls)
    assert prepare[-1:] == ['--dry-run']
    assert probe[:3] == [queue.sys.executable, '-m', 'scripts.preflight_single80']
    assert train[-1:] == ['--resume']
    assert test[-2:] == ['--resume', '--evaluate-test']
    for command, environment, _ in (calls[0], calls[2], calls[3]):
        assert command[:3] == [queue.sys.executable, '-m', 'unet_moe.single_channel_suite']
        assert command[command.index('--reference-records') + 1] == str(
            fair / 'shards/gpu1/records.json')
        assert environment['CUDA_VISIBLE_DEVICES'] == '0'
        expected = {'--seeds': '42,43,44', '--split-seed': '42', '--size': '512',
                    '--base': '24', '--batch-size': '16', '--epochs': '80',
                    '--quota-per-domain': '500', '--lr': '0.0001',
                    '--weight-decay': '0.0001', '--workers': '0',
                    '--device': 'cuda:0'}
        assert all(command[command.index(flag) + 1] == value
                   for flag, value in expected.items())
    assert calls[0][2] is None
    assert all(row[2] is not None for row in calls[1:])
    status = json.loads((tmp_path / 'single80-logs/controller_status.json').read_text())
    assert status['state'] == 'complete'
    assert status['summary'] == str(output / 'summary.json')


def test_wrong_fair_config_or_incomplete_single_result_is_rejected(tmp_path):
    fair = _fair(tmp_path)
    protocol_path = fair / 'shards/gpu1/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['config']['batch_size'] = 4
    _json(protocol_path, protocol)
    with pytest.raises(ValueError, match='batch_size'):
        queue.check_fair_configuration(fair)

    output = tmp_path / 'single80'
    _single_output(output, epochs=79)
    with pytest.raises(ValueError, match='80-epoch three-seed test barrier'):
        queue.validate_complete(output, 'paired-records')
    _single_output(output, fair_hash='different')
    with pytest.raises(ValueError, match='not paired'):
        queue.validate_complete(output, 'paired-records')


def test_failed_fair_controller_never_prepares_or_trains(tmp_path, monkeypatch):
    fair = _fair(tmp_path)
    controller = tmp_path / 'fair80-controller.json'
    _json(controller, {'state': 'failed'})
    monkeypatch.setattr(queue, 'fair_state', lambda *_: 'waiting')
    monkeypatch.setattr(queue, 'run_command',
                        lambda *_args, **_kwargs: pytest.fail('unexpected single-channel run'))
    with pytest.raises(RuntimeError, match='fair controller failed'):
        queue.main(['--fair-root', str(fair), '--research-status', str(controller),
                    '--data-root', str(tmp_path / 'data'),
                    '--output', str(tmp_path / 'single80'), '--poll-seconds', '1'])
    status = json.loads((tmp_path / 'single80-logs/controller_status.json').read_text())
    assert status['state'] == 'failed'


def test_preflight_report_must_match_reference(tmp_path):
    report = tmp_path / 'preflight.json'
    _json(report, {'schema': 'single80_physical_batch_preflight_v1',
                   'record_sha256': 'wrong', 'status': 'passed',
                   'optimizer_step_ok': True, 'optimizer_steps': 2, 'gpu_total_mib': 49140,
                'physical_batch_size': 16,
                   'peak_reserved_mib': 30000})
    with pytest.raises(ValueError, match='preflight'):
        queue.preflight_peak(report, 'paired-records')


@pytest.mark.parametrize('steps,capacity', [(1, 49140), (None, 49140), (2, 31000)])
def test_preflight_requires_resident_optimizer_state_and_attainable_memory(tmp_path, steps, capacity):
    report = tmp_path / 'preflight.json'
    _json(report, {'schema': 'single80_physical_batch_preflight_v1',
                   'record_sha256': 'paired-records', 'status': 'passed',
                   'optimizer_step_ok': True, 'optimizer_steps': steps,
                   'physical_batch_size': 16, 'gpu_total_mib': capacity,
                   'peak_reserved_mib': 30000})
    with pytest.raises(ValueError, match='preflight'):
        queue.preflight_peak(report, 'paired-records')
