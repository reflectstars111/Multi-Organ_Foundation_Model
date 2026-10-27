import json

from scripts.defer_abdomen_recovery import fair_state


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_queue_waits_for_all_test_runs_and_both_workers(tmp_path):
    fair = tmp_path / 'fair'
    status = tmp_path / 'research_status.json'
    assert fair_state(fair, status) == 'waiting'

    runs = [dict(kind=kind, seed=seed, test_evaluated=True)
            for kind in ('shared', 'task_id', 'region') for seed in (42, 43, 44)]
    summary = dict(test_evaluation_requested=True,
                   parallel=dict(training_barrier_verified=True), runs=runs)
    _write(fair / 'summary.json', summary)
    for gpu in (0, 1):
        _write(fair / f'worker_gpu{gpu}.json', dict(state='test_complete'))
        _write(fair / 'shards' / f'gpu{gpu}' / 'test_complete.json', {})
    assert fair_state(fair, status) == 'waiting'

    _write(fair / 'worker_gpu0.json', dict(state='complete'))
    _write(fair / 'worker_gpu1.json', dict(state='complete'))
    assert fair_state(fair, status) == 'ready'

    runs[0]['test_evaluated'] = False
    _write(fair / 'summary.json', summary)
    assert fair_state(fair, status) == 'waiting'


def test_queue_stops_on_failed_fair_worker_or_controller(tmp_path):
    fair = tmp_path / 'fair'
    status = tmp_path / 'research_status.json'
    _write(fair / 'worker_gpu0.json', dict(state='failed'))
    assert fair_state(fair, status) == 'fair GPU 0 worker failed'
    _write(fair / 'worker_gpu0.json', dict(state='training'))
    _write(status, dict(stage='failed:fair_comparison'))
    assert fair_state(fair, status) == 'fair controller failed:fair_comparison'
