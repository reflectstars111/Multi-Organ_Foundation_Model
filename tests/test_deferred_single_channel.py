"""The unattended queue must not start training before the fair barrier."""

import json

import pytest

from scripts import defer_single_channel as queue


def test_single_channel_queue_waits_and_stages_fixed_fair_configuration(tmp_path, monkeypatch):
    fair = tmp_path / 'fair'
    research = tmp_path / 'research_status.json'
    data = tmp_path / 'pointed_data'
    output = tmp_path / 'single_channel'
    stages = iter(('waiting', 'ready'))
    commands = []

    monkeypatch.setattr(queue, 'fair_state', lambda *_: next(stages))
    monkeypatch.setattr(queue.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(queue, 'gpu_lock_path', lambda *_: tmp_path / 'gpu.lock')
    monkeypatch.setattr(queue, 'gpu_free_mib', lambda *_: queue.MINIMUM_FREE_MIB)
    def fake_run(command, log, env, lock=None):
        commands.append((command, lock))
        if '--evaluate-test' in command:
            output.mkdir()
            (output / 'summary.json').write_text(json.dumps(dict(
                protocol_version='single_channel_paper5_union_v1',
                test_evaluation_requested=True,
                runs=[dict(seed=seed, epochs_completed=50, test_evaluated=True)
                      for seed in (42, 43, 44)])))
        return 0

    monkeypatch.setattr(queue, 'run_command', fake_run)

    queue.main(['--fair-root', str(fair), '--research-status', str(research),
                '--data-root', str(data), '--output', str(output),
                '--gpu', '0', '--poll-seconds', '1'])

    assert len(commands) == 3
    prepare, train, test = [command for command, _ in commands]
    assert prepare[-1] == '--dry-run'
    assert train[-1] == '--resume'
    assert test[-2:] == ['--resume', '--evaluate-test']
    assert all('--size' in command and command[command.index('--size') + 1] == '512'
               for command in (prepare, train, test))
    assert all(command[command.index('--reference-records') + 1] ==
               str(fair / 'shards/gpu1/records.json') for command in (prepare, train, test))
    assert commands[0][1] is None
    assert commands[1][1] is not None and commands[2][1] is not None
    status = json.loads((tmp_path / 'single_channel-logs/controller_status.json').read_text())
    assert status['state'] == 'complete'


def test_incomplete_single_channel_summary_is_not_accepted(tmp_path):
    (tmp_path / 'summary.json').write_text(json.dumps(dict(
        protocol_version='single_channel_paper5_union_v1',
        test_evaluation_requested=True,
        runs=[dict(seed=42, epochs_completed=50, test_evaluated=True)])))
    with pytest.raises(ValueError, match='three-seed test barrier'):
        queue.validate_complete(tmp_path)
