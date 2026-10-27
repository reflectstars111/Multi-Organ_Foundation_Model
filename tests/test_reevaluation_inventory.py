import hashlib
import json
from pathlib import Path

from unet_moe.reevaluation_inventory import build_inventory, discover_jobs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_guarded(root, *, completed=(80, 40), tested=(False, False), guardrail=(None, None)):
    experiment = root / 'q01_organ_weighted_fixture'
    config = dict(models=['region'], arms=['image_mean'], seeds=[42, 43], epochs=80,
                  size=512, task_ids=[0, 1], base=4)
    write_json(experiment / 'protocol.json', dict(config=config, protocol_version='organ_weighted_v1'))
    write_json(experiment / 'records.json', [dict(split='train'), dict(split='val'), dict(split='test')])
    (experiment / 'source_snapshot').mkdir()
    for index, seed in enumerate(config['seeds']):
        folder = experiment / 'image_mean' / 'region' / f'seed_{seed}'
        write_json(folder / 'run_summary.json', dict(epochs_completed=completed[index],
                   best_epoch=23, guardrail_epoch=guardrail[index], test_evaluated=tested[index]))
        for name in ('best', 'last') + (('guardrail',) if guardrail[index] is not None else ()):
            (folder / f'{name}.pt').write_bytes(f'checkpoint-{seed}-{name}'.encode())
    return experiment


def test_guarded_test_waits_for_all_training_and_original_test_opening(tmp_path):
    experiment = make_guarded(tmp_path)
    jobs = discover_jobs(tmp_path)
    statuses = {(j['seed'], j['selection'], j['split']): j['status'] for j in jobs}
    assert statuses[42, 'best', 'val'] == 'ready'
    assert statuses[42, 'best', 'test'] == 'deferred'
    assert statuses[43, 'best', 'val'] == 'deferred'
    assert statuses[42, 'guardrail', 'val'] == 'not_selected'
    assert statuses[43, 'guardrail', 'val'] == 'deferred'
    path = experiment / 'image_mean/region/seed_43/run_summary.json'
    data = json.loads(path.read_text())
    data['epochs_completed'] = 80
    write_json(path, data)
    assert all(j['status'] == 'deferred' for j in discover_jobs(tmp_path)
               if j['split'] == 'test' and j['selection'] == 'best')
    data['test_evaluated'] = True
    write_json(path, data)
    assert all(j['status'] == 'ready' for j in discover_jobs(tmp_path)
               if j['split'] == 'test' and j['selection'] == 'best')


def test_guardrail_is_never_substituted_and_missing_selected_is_unavailable(tmp_path):
    experiment = make_guarded(tmp_path, completed=(80, 80), tested=(True, True), guardrail=(12, None))
    (experiment / 'image_mean/region/seed_42/guardrail.pt').unlink()
    jobs = discover_jobs(tmp_path)
    assert {j['status'] for j in jobs if j['seed'] == 42 and j['selection'] == 'guardrail'} == {'unavailable'}
    assert {j['status'] for j in jobs if j['seed'] == 43 and j['selection'] == 'guardrail'} == {'not_selected'}


def test_inventory_hashes_completed_checkpoints_once_and_keeps_frozen_sources(tmp_path):
    experiment = make_guarded(tmp_path, completed=(80, 80), tested=(True, True))
    jobs = discover_jobs(tmp_path, hash_checkpoints=True)
    for job in jobs:
        assert job['source_frozen'] is True
        assert job['source_root'] == str(experiment / 'source_snapshot')
        if job['status'] == 'ready':
            assert job['checkpoint_sha256'] == hashlib.sha256(Path(job['checkpoint']).read_bytes()).hexdigest()
    assert len({j['job_id'] for j in jobs}) == len(jobs)


def test_legacy_own_records_and_no_snapshot_not_misrepresented(tmp_path):
    experiment = tmp_path / 'region_top1_v1'
    write_json(experiment / 'config.json', dict(size=256, epochs=50, seed=42, base=16))
    write_json(experiment / 'records.json', dict(records=[dict(split='val'), dict(split='test')]))
    (experiment / 'metrics.jsonl').write_text(json.dumps(dict(epoch=50)) + '\n')
    (experiment / 'best.pt').write_bytes(b'best')
    (experiment / 'last.pt').write_bytes(b'last')
    jobs = discover_jobs(tmp_path, project_root=tmp_path / 'repo')
    assert len(jobs) == 4
    assert all(j['family'] == 'legacy_region' and j['status'] == 'ready' for j in jobs)
    assert all(j['size'] == 256 and j['source_frozen'] is False for j in jobs)


def test_unfinished_guarded_runs_exist_without_checkpoint_files(tmp_path):
    experiment = make_guarded(tmp_path)
    folder = experiment / 'image_mean/region/seed_43'
    for path in folder.iterdir():
        path.unlink()
    jobs = discover_jobs(tmp_path)
    assert len(jobs) == 12
    assert all(j['status'] == 'deferred' for j in jobs if j['seed'] == 43)


def test_smoke_and_operational_directories_remain_in_inventory(tmp_path):
    experiment = tmp_path / 'fair_smoke'
    write_json(experiment / 'protocol.json', dict(config=dict(models=['shared'], seeds=[42],
               epochs=1, size=32, protocol_version='fair_image_matched_v1')))
    write_json(experiment / 'records.json', [dict(split='val')])
    folder = experiment / 'shared/seed_42'
    folder.mkdir(parents=True)
    (folder / 'best.pt').write_bytes(b'best')
    (tmp_path / 'experiment-logs').mkdir()
    (tmp_path / 'preflight').mkdir()
    result = build_inventory(tmp_path)
    assert len(result['experiments']) == 3
    assert result['counts'] == {'excluded': 4}
    assert result['checkpoint_count'] == 1


def test_fair_records_digest_mismatch_blocks_evaluation(tmp_path):
    experiment = tmp_path / 'fair_full'
    write_json(experiment / 'protocol.json', dict(config=dict(models=['shared'], seeds=[42],
               epochs=1, size=512, protocol_version='fair_image_matched_v1'), record_sha256='bad'))
    write_json(experiment / 'records.json', [dict(split='val'), dict(split='test')])
    folder = experiment / 'shared/seed_42'
    write_json(folder / 'run_summary.json', dict(epochs_completed=1, best_epoch=1))
    for name in ('best', 'last'):
        (folder / f'{name}.pt').write_bytes(b'checkpoint')
    assert all(j['status'] == 'unavailable' for j in discover_jobs(tmp_path))


def test_malformed_experiment_reported_without_hiding_other_jobs(tmp_path):
    make_guarded(tmp_path)
    malformed = tmp_path / 'malformed'
    malformed.mkdir()
    (malformed / 'protocol.json').write_text('{broken')
    result = build_inventory(tmp_path)
    assert len(result['jobs']) == 12
    assert any(e['category'] == 'invalid_artifacts' for e in result['experiments'])


def test_recovery_validation_only_retains_original_reference_records(tmp_path):
    reference = tmp_path / 'fair_reference'
    write_json(reference / 'shards/gpu1/protocol.json', dict(config=dict(size=512, base=16, task_ids=[0])))
    write_json(reference / 'shards/gpu1/records.json', [dict(split='val'), dict(split='test')])
    experiment = tmp_path / 'abdomen_recovery'
    write_json(experiment / 'plan.json', dict(schema='abdomen_recovery_v1', fair_root=str(reference),
               seeds=[42], arms=[dict(name='control')], design=dict(size=512, epochs=8)))
    folder = experiment / 'control/seed_42'
    write_json(folder / 'result.json', dict(epochs_completed=8, selected_epoch=0))
    for name in ('best', 'last'):
        (folder / f'{name}.pt').write_bytes(b'checkpoint')
    jobs = [j for j in discover_jobs(tmp_path) if j['family'] == 'abdomen_recovery']
    assert len(jobs) == 4
    assert all(j['status'] == 'excluded' for j in jobs if j['split'] == 'test')
    assert all(j['status'] == 'ready' for j in jobs if j['split'] == 'val')
    assert next(j for j in jobs if j['selection'] == 'best')['expected_epoch'] == 0


def test_postprocess_valid_and_superseded_are_distinguished(tmp_path):
    reference = tmp_path / 'fair_reference/shards/gpu1'
    write_json(reference / 'protocol.json', dict(config=dict(size=512, task_ids=[0])))
    write_json(reference / 'records.json', [dict(split='val')])
    checkpoint = reference / 'region/seed_42/best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    for name in ('postprocess_v1', 'postprocess_v2'):
        experiment = tmp_path / name
        write_json(experiment / 'screen.json', dict(protocol=dict(checkpoint=str(checkpoint),
                   checkpoint_sha256=hashlib.sha256(b'checkpoint').hexdigest(), checkpoint_epoch=9,
                   input_size=512, candidate_grid=[dict(id='s000', threshold=.5, min_area=0, largest_only=False)])))
    write_json(tmp_path / 'postprocess_v1/superseded.json', dict(reason='bad rule'))
    jobs = [j for j in discover_jobs(tmp_path) if j['family'] == 'postprocess_screen']
    assert len(jobs) == 2
    assert {j['experiment']: j['status'] for j in jobs} == {'postprocess_v1': 'excluded', 'postprocess_v2': 'ready'}
    assert all(j['records_path'] == str(reference / 'records.json') for j in jobs)


def test_archived_partial_ablation_has_descriptive_val_but_never_test(tmp_path):
    experiment = tmp_path / 'empty_ablation_20260930'
    protocol_root = experiment / 'weight_0.1'
    write_json(protocol_root / 'protocol.json', dict(config=dict(models=['region'], seeds=[42],
               epochs=12, size=256, protocol_version='fair_image_matched_v1')))
    write_json(protocol_root / 'records.json', [dict(split='val'), dict(split='test')])
    folder = protocol_root / 'region/seed_42'
    folder.mkdir(parents=True)
    (folder / 'metrics.jsonl').write_text(json.dumps(dict(epoch=3)) + '\n')
    for name in ('best', 'last'):
        (folder / f'{name}.pt').write_bytes(b'checkpoint')
    jobs = discover_jobs(tmp_path)
    assert all(j['archived_partial'] for j in jobs)
    assert all(j['status'] == 'ready' for j in jobs if j['split'] == 'val')
    assert all(j['status'] == 'excluded' for j in jobs if j['split'] == 'test')
    assert next(j for j in jobs if j['selection'] == 'last')['expected_epoch'] == 3


def test_complete_summary_does_not_parse_large_or_partial_history(tmp_path):
    experiment = make_guarded(tmp_path, completed=(80, 80))
    (experiment / 'image_mean/region/seed_42/metrics.jsonl').write_text('not json\nnot json\n')
    assert len(discover_jobs(tmp_path)) == 12


def test_single_union_finished_seed_does_not_open_test_early(tmp_path):
    experiment = tmp_path / 'single_channel_union_fixture'
    write_json(experiment / 'protocol.json', dict(config=dict(seeds=[42, 43], epochs=80,
               size=512, protocol_version='single_channel_paper5_union_v1')))
    write_json(experiment / 'records.json', [dict(split='val'), dict(split='test')])
    folder = experiment / 'seed_42'
    write_json(folder / 'run_summary.json', dict(epochs_completed=80, best_epoch=11, test_evaluated=False))
    for name in ('best', 'last'):
        (folder / f'{name}.pt').write_bytes(b'checkpoint')
    jobs = discover_jobs(tmp_path)
    assert all(j['status'] == 'deferred' for j in jobs if j['split'] == 'test')
    assert all(j['status'] == 'ready' for j in jobs if j['split'] == 'val' and j['seed'] == 42)
