import copy
import hashlib
import json

import pytest

from scripts.summarize_empty_ablation import ABDOMEN_STRUCTURES, summarize


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture_run(root, weight, metrics):
    folder = root / f'weight_{weight:g}'
    (folder / 'region/seed_42').mkdir(parents=True)
    config = dict(empty_weight=weight, empty_fraction=.01, epochs=len(metrics),
                  models=['region'], seeds=[42], size=128, base=8, lr=.001,
                  checkpoint_selection='best_validation_macro_positive_dice', threshold=.5)
    source_files = {'fair_suite.py': 'a' * 64, 'empty_loss.py': 'b' * 64}
    protocol = dict(config=config, config_sha256=digest(config), record_sha256='c' * 64,
                    source_files=source_files, source_sha256=digest(source_files),
                    budget=dict(images_per_epoch=18, planned_epochs=len(metrics),
                                planned_images_per_run=18 * len(metrics)))
    write_json(folder / 'protocol.json', protocol)
    path = folder / 'region/seed_42/metrics.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in metrics))
    return folder


def metric(epoch, dice=.7, area=.04, fp=.2, miss=.1):
    structure = dict(samples=20, positive_samples=10, empty_samples=10, positive_dice=dice,
                     empty_false_positive_rate=fp, empty_mean_false_positive_area_fraction=area,
                     empty_mean_false_positive_pixels=area * 1024, positive_zero_overlap_rate=miss)
    per_structure = {name: dict(structure) for name in ['breast_lesion', *ABDOMEN_STRUCTURES]}
    return dict(epoch=epoch, kind='region', seed=42, threshold=.5,
                macro_positive_dice=dice, per_structure=per_structure, sample_count=18,
                sample_sha256=digest(['same', epoch]), annotated_channel_queries=36,
                model_input_images=18, parameter_count=1200, lr=.001)


def prepare(root, baseline=None, candidate=None):
    baseline = baseline or [metric(1, .7), metric(2, .72)]
    candidate = candidate or [metric(1, .721, area=.02), metric(2, .71, area=.01)]
    return (fixture_run(root, 0., baseline), fixture_run(root, .1, candidate))


def mutate_protocol(folder, mutate):
    path = folder / 'protocol.json'
    protocol = json.loads(path.read_text())
    mutate(protocol)
    protocol['config_sha256'] = digest(protocol['config'])
    write_json(path, protocol)


def test_selects_own_best_validation_epoch_and_never_reads_test(tmp_path):
    baseline, candidate = prepare(tmp_path)
    # Any attempt to parse the release test would fail.
    (candidate / 'region/seed_42/test_metrics.json').write_text('not JSON')
    result = summarize(tmp_path)
    assert result['baseline']['selected_epoch'] == 2
    assert result['candidate']['selected_epoch'] == 1
    assert result['decision']['status'] == 'pilot_promising'
    assert result['default_empty_weight'] == 0
    assert result['differences']['abdomen']['positive_dice'] == pytest.approx(.001)
    assert result['differences']['abdomen']['empty_mean_false_positive_area_fraction'] == pytest.approx(-.02)
    assert (tmp_path / 'comparison.json').is_file()
    with pytest.raises(FileExistsError):
        summarize(tmp_path)


@pytest.mark.parametrize('changed', ['record_sha256', 'source_sha256'])
def test_refuses_incompatible_record_or_source_hash(tmp_path, changed):
    _, candidate = prepare(tmp_path)
    mutate_protocol(candidate, lambda p: p.update({changed: 'd' * 64}))
    with pytest.raises(ValueError, match='[Rr]ecord|[Ss]ource'):
        summarize(tmp_path)
    assert not (tmp_path / 'comparison.json').exists()


def test_refuses_other_config_changes_even_with_updated_hash(tmp_path):
    _, candidate = prepare(tmp_path)
    mutate_protocol(candidate, lambda p: p['config'].update(size=256))
    with pytest.raises(ValueError, match='[Cc]onfig'):
        summarize(tmp_path)


@pytest.mark.parametrize('change', ['sample_sha256', 'sample_count', 'incomplete'])
def test_refuses_different_draws_or_incomplete_budget(tmp_path, change):
    baseline = [metric(1), metric(2)]
    candidate = copy.deepcopy(baseline)
    if change == 'sample_sha256':
        candidate[0][change] = 'e' * 64
    elif change == 'sample_count':
        candidate[0][change] = 17
    first, second = prepare(tmp_path, baseline, candidate)
    if change == 'incomplete':
        path = second / 'region/seed_42/metrics.jsonl'
        path.write_text(json.dumps(candidate[0]) + '\n')
    with pytest.raises(ValueError, match='draw|sample|budget|epoch'):
        summarize(tmp_path)


def test_rejects_reduced_false_positives_when_dice_suffers(tmp_path):
    prepare(tmp_path, [metric(1, .7)], [metric(1, .68, area=.001, fp=.01)])
    result = summarize(tmp_path)
    assert result['decision']['status'] == 'not_selected'
    assert result['decision']['checks']['abdomen_empty_area_reduced']['passed']
    assert not result['decision']['checks']['abdomen_positive_dice_preserved']['passed']
    assert not result['decision']['checks']['overall_positive_dice_preserved']['passed']
    assert result['default_empty_weight'] == 0


@pytest.mark.parametrize('field,value,rule', [
    ('empty_false_positive_rate', .22, 'abdomen_empty_fp_rate_preserved'),
    ('positive_zero_overlap_rate', .12, 'abdomen_zero_overlap_preserved'),
    ('empty_mean_false_positive_area_fraction', .04, 'abdomen_empty_area_reduced'),
])
def test_each_false_positive_and_miss_safeguard_is_required(tmp_path, field, value, rule):
    candidate = metric(1, area=.01)
    for row in candidate['per_structure'].values():
        row[field] = value
    prepare(tmp_path, [metric(1)], [candidate])
    result = summarize(tmp_path)
    assert result['decision']['status'] == 'not_selected'
    assert not result['decision']['checks'][rule]['passed']


def test_tied_best_macro_uses_first_epoch_as_training_does(tmp_path):
    prepare(tmp_path, [metric(1), metric(2)], [metric(1, area=.03), metric(2, area=.01)])
    result = summarize(tmp_path)
    assert result['candidate']['selected_epoch'] == 1
    assert result['candidate']['abdomen']['empty_mean_false_positive_area_fraction'] == pytest.approx(.03)


def test_missing_abdominal_denominator_cannot_satisfy_rules(tmp_path):
    baseline = metric(1)
    baseline['per_structure']['abdomen_liver']['empty_samples'] = 0
    baseline['per_structure']['abdomen_liver']['empty_false_positive_rate'] = None
    prepare(tmp_path, [baseline], [metric(1, area=.01)])
    with pytest.raises(ValueError, match='samples|denominator'):
        summarize(tmp_path)
