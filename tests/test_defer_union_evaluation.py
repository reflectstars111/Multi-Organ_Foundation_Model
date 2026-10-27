"""CPU-only checks for the deferred common-union evaluation controller."""

import json
from pathlib import Path

import pytest

from scripts import defer_union_evaluation as queue


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _prerequisites(tmp_path):
    fair = tmp_path / 'fair'
    single = tmp_path / 'single'
    single_status = tmp_path / 'single-logs' / 'controller_status.json'
    _json(fair / 'summary.json', {'record_sha256': 'same-records'})
    _json(single / 'summary.json', {
        'protocol_version': 'single_channel_paper5_union_v1',
        'test_evaluation_requested': True,
        'record_sha256': 'same-records',
        'runs': [{'seed': seed, 'epochs_completed': 50, 'test_evaluated': True}
                 for seed in (42, 43, 44)],
    })
    _json(single / 'protocol.json', {'record_sha256': 'same-records',
                                    'reference_records_sha256': 'same-records',
                                    'config': {'size': 512}})
    _json(single_status, {'state': 'complete', 'summary': str(single / 'summary.json')})
    return fair, single, single_status


def _union_output(output, fair):
    protocol = {'schema': 'fair_models_union_evaluation_v1', 'split': 'test',
                'fair_root': str(fair.resolve()), 'fair_record_sha256': 'same-records'}
    runs = []
    for kind, seed in sorted(queue.EXPECTED):
        runs.append({'kind': kind, 'seed': seed, 'metrics': {'overall': {'samples': 2}}})
        _json(output / f'{kind}_seed_{seed}.json', runs[-1])
    _json(output / 'protocol.json', protocol)
    _json(output / 'summary.json', {'protocol': protocol, 'runs': runs})


def test_waits_for_both_barriers_and_rejects_unpaired_single_result(tmp_path, monkeypatch):
    fair, single, status = _prerequisites(tmp_path)
    research = tmp_path / 'research.json'
    monkeypatch.setattr(queue, 'fair_state', lambda *args: 'waiting')
    assert queue.prerequisite_state(fair, research, status, single) == 'waiting_fair'
    monkeypatch.setattr(queue, 'fair_state', lambda *args: 'ready')
    assert queue.prerequisite_state(fair, research, tmp_path / 'missing', single) == 'waiting_single'
    _json(status, {'state': 'training'})
    assert queue.prerequisite_state(fair, research, status, single) == 'waiting_single'
    _json(status, {'state': 'complete', 'summary': str(single / 'summary.json')})
    assert queue.prerequisite_state(fair, research, status, single) == 'ready'
    protocol = json.loads((single / 'protocol.json').read_text())
    protocol['reference_records_sha256'] = 'different'
    _json(single / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='not paired'):
        queue.prerequisite_state(fair, research, status, single)


def test_only_complete_matching_union_output_can_be_reused(tmp_path):
    fair, _, _ = _prerequisites(tmp_path)
    output = tmp_path / 'union'
    assert not queue.completed_output(output, fair)
    _union_output(output, fair)
    assert queue.completed_output(output, fair)
    (output / 'region_seed_44.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        queue.completed_output(output, fair)


def test_controller_uses_gpu0_lock_and_runs_only_test_evaluation(tmp_path, monkeypatch):
    fair, single, single_status = _prerequisites(tmp_path)
    output = tmp_path / 'union'
    gpu_lock = tmp_path / 'gpu0.lock'
    monkeypatch.setattr(queue, 'prerequisite_state', lambda *args: 'ready')
    monkeypatch.setattr(queue, 'gpu_lock_path', lambda gpu: gpu_lock)
    monkeypatch.setattr(queue, 'gpu_free_mib', lambda gpu: queue.MINIMUM_FREE_MIB)

    def fake_command(command, log_path, environment, lock):
        assert command[0:3] == [queue.sys.executable, '-m', 'unet_moe.union_evaluate']
        assert command[command.index('--split') + 1] == 'test'
        assert command[command.index('--device') + 1] == 'cuda:0'
        assert environment['CUDA_VISIBLE_DEVICES'] == '0'
        assert lock is not None and lock.fileno() >= 0
        _union_output(output, fair)
        return 0

    monkeypatch.setattr(queue, 'run_command', fake_command)
    queue.main(['--fair-root', str(fair), '--research-status', str(tmp_path / 'research.json'),
                '--single-status', str(single_status), '--single-output', str(single),
                '--output', str(output), '--poll-seconds', '1'])
    controller = json.loads((tmp_path / 'union-logs/controller_status.json').read_text())
    assert controller['state'] == 'complete'
    assert controller['gpu'] == 0
    assert controller['summary'] == str(output / 'summary.json')
