"""CPU checks for the independent 80-epoch plan and GPU preflight gate."""

from copy import deepcopy
import json
import signal
from types import SimpleNamespace

import pytest

from scripts import defer_fair_80 as queue
from scripts import fair_parallel as original


@pytest.fixture
def plan(tmp_path):
    return queue.build_plan(tmp_path / 'fair80', tmp_path / 'data')


def prepared_protocol(plan, gpu):
    config = original.shard_config(plan, gpu)
    records = [{'record': 'synthetic'}]
    protocol = dict(config=config, config_sha256=original.digest(config),
                    record_sha256=original.digest(records),
                    source_files=plan['source_files'],
                    source_sha256=plan['source_sha256'],
                    budget=plan['shards'][gpu]['budget'])
    folder = original.shard_path(plan, gpu)
    original.write_json(folder / 'protocol.json', protocol)
    original.write_json(folder / 'records.json', records)
    return protocol


def report_for(plan, kind, protocol):
    return dict(schema='fair80_worst_batch_preflight_v1', kind=kind,
                status='passed', record_sha256=protocol['record_sha256'],
                source_sha256=protocol['source_sha256'],
                effective_batch_size=16, physical_micro_batch_size=4,
                selection=dict(images=16, annotated_queries=58,
                               max_microbatch_queries=20),
                peak_reserved_mib=18000, gpu_total_mib=49152,
                optimizer_step_ok=True, optimizer_steps=2)


def test_plan_is_80_epoch_accum_and_original_module_is_restored(plan, tmp_path):
    assert plan['schema'] == queue.SCHEMA
    assert plan['budget']['planned_epochs'] == 80
    assert plan['budget']['optimizer_steps_per_epoch'] == 282
    assert plan['budget']['planned_images_per_run'] == 360000
    assert plan['common_config']['batch_size'] == 16
    assert plan['common_config']['micro_batch_size'] == 4
    assert plan['common_config']['base'] == 24
    assert plan['common_config']['lr'] == .0001
    assert plan['common_config']['workers'] == 0
    assert plan['common_config']['protocol_version'] == 'fair_image_matched_accum_v1'
    assert plan['shards']['0']['models'] == ['shared', 'task_id']
    assert plan['shards']['1']['models'] == ['region']
    assert 'fair_suite_accum.py' in plan['source_files']
    old_epochs, old_validate = original.EPOCHS, original.validate_plan
    with queue.configured_parallel():
        assert original.EPOCHS == 80
        assert original.validate_plan is queue.validate_plan
        output = tmp_path / 'prepared'
        saved = original.prepare_plan(output, tmp_path / 'data')
        assert saved['plan_sha256'] == queue.build_plan(output, tmp_path / 'data')['plan_sha256']
        altered = deepcopy(saved)
        altered['common_config']['micro_batch_size'] = 8
        with pytest.raises(ValueError):
            queue.validate_plan(altered)
    assert original.EPOCHS == old_epochs == 50
    assert original.validate_plan is old_validate


def test_preflight_requires_three_real_steps_and_matching_protocol(plan):
    protocols = {gpu: prepared_protocol(plan, gpu) for gpu in ('0', '1')}
    for kind in original.MODELS:
        gpu = '1' if kind == 'region' else '0'
        original.write_json(queue.report_path(plan, kind),
                            report_for(plan, kind, protocols[gpu]))
    marker = queue.preflight_marker(plan, protocols)
    assert marker['minimum_free_mib_by_gpu'] == {'0': queue.MINIMUM_FREE_MIB,
                                                  '1': queue.MINIMUM_FREE_MIB}
    original.write_json(plan['output'] + '/preflight_complete.json', marker)
    assert queue.validate_preflight(plan) == marker

    task_report = queue.report_path(plan, 'task_id')
    corrupted = json.loads(task_report.read_text())
    corrupted['optimizer_step_ok'] = False
    original.write_json(task_report, corrupted)
    with pytest.raises(ValueError, match='incomplete'):
        queue.validate_preflight(plan)


def test_preflight_memory_gate_is_bound_to_report_peak(plan):
    protocols = {gpu: prepared_protocol(plan, gpu) for gpu in ('0', '1')}
    for kind in original.MODELS:
        gpu = '1' if kind == 'region' else '0'
        report = report_for(plan, kind, protocols[gpu])
        report['peak_reserved_mib'] = 33000 if kind == 'task_id' else 12000
        original.write_json(queue.report_path(plan, kind), report)
    marker = queue.preflight_marker(plan, protocols)
    assert marker['minimum_free_mib_by_gpu']['0'] == 35048
    assert marker['minimum_free_mib_by_gpu']['1'] == queue.MINIMUM_FREE_MIB


def test_prepared_records_must_equal_completed_fifty_epoch_protocol(plan, tmp_path):
    for gpu in ('0', '1'):
        prepared_protocol(plan, gpu)
    old = tmp_path / 'old_fair'
    old.mkdir()
    original.write_json(old / 'summary.json', dict(record_sha256='different'))
    with pytest.raises(ValueError, match='new records differ'):
        queue.prepare_shards(plan, old, tmp_path / 'controller.log')


def test_worker_uses_new_trainer_and_explicit_test_flag(plan, tmp_path, monkeypatch):
    prepared_protocol(plan, '0')
    commands = []
    class Completed:
        pid = 101
        def wait(self):
            return 0
        def poll(self):
            return 0
    def fake_popen(command, **kwargs):
        commands.append((command, kwargs))
        return Completed()
    monkeypatch.setattr(queue.subprocess, 'Popen', fake_popen)
    with (tmp_path / 'gpu.lock').open('a+') as lock:
        worker = queue.Worker80(plan, '0', lock)
        worker.run_suite()
        worker.run_suite(evaluate_test=True)
    train, test = [item[0] for item in commands]
    assert train[:3] == [queue.sys.executable, '-m', queue.TRAINER]
    assert train[-1] == '--resume'
    assert test[-2:] == ['--resume', '--evaluate-test']
    assert all(command[command.index('--micro-batch-size') + 1] == '4'
               for command in (train, test))
    assert all(command[command.index('--batch-size') + 1] == '16'
               for command in (train, test))
    assert all(kwargs['pass_fds'] for _, kwargs in commands)
    assert all(kwargs['start_new_session'] is False for _, kwargs in commands)


class PollingProcess:
    def __init__(self, pid, code=None):
        self.pid, self.code = pid, code
        self.waits = []

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.code is None:
            raise queue.subprocess.TimeoutExpired('synthetic worker', timeout)
        return self.code


def test_second_worker_failure_is_seen_while_first_waits_for_gpu(monkeypatch):
    children = {0: PollingProcess(100), 1: PollingProcess(200, 7)}
    def unexpected_sleep(seconds):
        pytest.fail('A failed peer must be detected before sleeping')
    monkeypatch.setattr(queue.time, 'sleep', unexpected_sleep)
    with pytest.raises(RuntimeError, match=r'0: None, 1: 7'):
        queue.wait_workers(children, 45)
    assert children[0].waits == []


def test_worker_polling_finishes_only_when_both_workers_exit(monkeypatch):
    children = {0: PollingProcess(100, 0), 1: PollingProcess(200)}
    sleeps = []
    def finish_peer(seconds):
        sleeps.append(seconds)
        children[1].code = 0
    monkeypatch.setattr(queue.time, 'sleep', finish_peer)
    assert queue.wait_workers(children, 45) == {0: 0, 1: 0}
    assert sleeps == [original.POLL_SECONDS]


def test_cleanup_kills_owned_groups_even_if_worker_exited(monkeypatch):
    children = {0: PollingProcess(100), 1: PollingProcess(200, 7)}
    signals = []
    def killpg(pid, signum):
        signals.append((pid, signum))
        if pid == 100 and signum == signal.SIGKILL:
            children[0].code = -signal.SIGKILL
    monkeypatch.setattr(queue.os, 'killpg', killpg)
    assert queue.stop_workers(children, timeout=0) == {0: -signal.SIGKILL, 1: 7}
    assert signals == [(100, signal.SIGTERM), (200, signal.SIGTERM),
                       (100, signal.SIGKILL), (200, signal.SIGKILL)]


def test_worker_status_write_failure_terminates_started_trainer(plan, tmp_path, monkeypatch):
    prepared_protocol(plan, '0')
    child = PollingProcess(100)
    terminated = []
    def terminate():
        terminated.append(True)
        child.code = -signal.SIGTERM
    child.terminate = terminate
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *args, **kwargs: child)
    with (tmp_path / 'gpu.lock').open('a+') as lock:
        worker = queue.Worker80(plan, '0', lock)
        def fail_status(*args, **kwargs):
            raise OSError('Synthetic disk failure')
        monkeypatch.setattr(worker, 'update', fail_status)
        with pytest.raises(OSError, match='disk failure'):
            worker.run_suite()
    assert terminated == [True]
    assert child.waits == [10]


def test_controller_records_failure_and_stops_peer(plan, tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / 'fair80', fair_root=tmp_path / 'old',
                           research_status=tmp_path / 'research.json',
                           data_root=tmp_path / 'data', poll_seconds=45)
    monkeypatch.setattr(queue, 'fair_state', lambda *args: 'ready')
    monkeypatch.setattr(queue.parallel, 'prepare_plan', lambda *args: plan)
    monkeypatch.setattr(queue, 'prepare_shards', lambda *args: None)
    monkeypatch.setattr(queue, 'run_preflights', lambda *args: None)
    monkeypatch.setattr(queue, 'complete_root', lambda *args: False)
    children = [PollingProcess(100), PollingProcess(200, 7)]
    launches = []
    def start(command, **kwargs):
        launches.append(kwargs)
        return children[len(launches) - 1]
    monkeypatch.setattr(queue.subprocess, 'Popen', start)
    cleaned = []
    def cleanup(workers):
        status = json.loads((tmp_path / 'fair80-logs' / 'controller_status.json').read_text())
        assert status['state'] == 'failed'
        assert '1: 7' in status['error']
        cleaned.append(workers)
    monkeypatch.setattr(queue, 'stop_workers', cleanup)
    with pytest.raises(RuntimeError, match='1: 7'):
        queue.controller_main(args)
    assert cleaned == [{0: children[0], 1: children[1]}]
    assert all(launch['start_new_session'] for launch in launches)


def test_controller_term_enters_cleanup_path_and_restores_handler():
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError, match='received signal'):
        with queue.handle_termination():
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize('steps', [None, 0, 1])
def test_preflight_rejects_reports_without_two_completed_updates(plan, steps):
    protocols = {gpu: prepared_protocol(plan, gpu) for gpu in ('0', '1')}
    report = report_for(plan, 'task_id', protocols['0'])
    if steps is None:
        report.pop('optimizer_steps')
    else:
        report['optimizer_steps'] = steps
    with pytest.raises(ValueError, match='incomplete'):
        queue._validate_report(plan, 'task_id', report, protocols)


def test_preflight_rejects_impossible_worker_memory_admission(plan):
    protocols = {gpu: prepared_protocol(plan, gpu) for gpu in ('0', '1')}
    report = report_for(plan, 'task_id', protocols['0'])
    report['peak_reserved_mib'] = report['gpu_total_mib'] - 1000
    with pytest.raises(ValueError, match='exceeding the GPU capacity'):
        queue._validate_report(plan, 'task_id', report, protocols)
