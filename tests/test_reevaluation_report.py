"""Scientific aggregation contracts, independent of GPU/model implementation."""
import copy
import math

import numpy as np
import pytest

from unet_moe.reevaluation_report import (
    COHORTS, csv_text, markdown_report, report_rows, seed_groups, summarize_rows,
    with_independent_ensembles,
)
from unet_moe.surface_metrics import binary_metrics


def row(structure='organ_a', domain='BUSI', dsc=100.0, hd95=0.0, assd=0.0,
        *, view='structure', variant=None, image_id='image', source=''):
    target = np.zeros((5, 5), dtype=np.int8)
    target[2, 2] = 1
    metrics = binary_metrics(target, target)
    metrics.update(dsc_percent=dsc, hd95_pixel=hd95, assd_pixel=assd)
    return dict(image_id=image_id, structure=structure, domain=domain, view=view,
                variant=variant, source=source, **metrics)


def result(rows, *, seed=42, split='test', size=512, training_status='complete',
           comparability_group='same_data_protocol', kind='region', arm='control',
           domain=None, selection='best', experiment='paired', family='organ_weighted'):
    return dict(checkpoint_sha256=f'sha-{seed}-{domain}',
                job=dict(job_id=f'{experiment}-{seed}-{domain}-{split}-{size}-{training_status}',
                         experiment=experiment, family=family, kind=kind, arm=arm,
                         seed=seed, selection=selection, split=split, size=size, domain=domain,
                         training_status=training_status, archived_partial=training_status == 'archived_partial',
                         comparability_group=comparability_group, training_epochs=80,
                         epochs_completed=7 if training_status == 'archived_partial' else 80),
                metrics=summarize_rows(rows))


def test_structure_and_union_have_separate_macro_denominators():
    views = summarize_rows([row(dsc=25), row(structure='organ_b', dsc=75),
                            row(structure=None, view='union', dsc=90)])
    by_view = {view['view']: view for view in views}
    assert set(by_view) == {'structure', 'union'}
    assert by_view['structure']['macro']['positive']['dsc_percent'] == 50
    assert by_view['structure']['macro']['positive']['n_groups'] == 2
    assert by_view['union']['macro']['positive']['dsc_percent'] == 90
    assert by_view['union']['macro']['positive']['n_groups'] == 1
    assert by_view['union']['group_by'] == 'domain'


def test_organ_macro_is_equal_group_weight_not_image_mean():
    rows = [row(structure='common', dsc=100, hd95=0, image_id=f'common{i}') for i in range(9)]
    rows.append(row(structure='rare', dsc=0, hd95=10, assd=8, image_id='rare'))
    summary = summarize_rows(rows)[0]
    assert summary['macro']['positive']['dsc_percent'] == 50
    assert summary['macro']['positive']['hd95_pixel'] == 5
    assert summary['macro']['positive']['assd_pixel'] == 4
    assert summary['image_mean']['positive']['dsc_percent'] == 90
    assert summary['image_mean']['positive']['hd95_pixel'] == 1
    assert summary['macro']['positive']['n_included'] == 10


def test_union_macro_is_equal_domain_weight():
    rows = [row(None, 'BUSI', dsc=100, view='union', image_id=str(i)) for i in range(3)]
    rows.append(row(None, 'DDTI', dsc=0, view='union', image_id='rare'))
    summary = summarize_rows(rows)[0]
    assert summary['macro']['positive']['dsc_percent'] == 50
    assert summary['image_mean']['positive']['dsc_percent'] == 75


def test_positive_cohort_excludes_empty_target_but_all_keeps_it():
    empty = row(structure='organ_a', image_id='empty')
    empty.update(binary_metrics(np.zeros((5, 5), np.int8), np.zeros((5, 5), np.int8)))
    summary = summarize_rows([row(dsc=20), empty])[0]
    assert summary['macro']['positive']['dsc_percent'] == 20
    assert summary['macro']['all']['dsc_percent'] == 60
    assert summary['macro']['positive']['n_included'] == 1
    assert summary['macro']['all']['n_included'] == 2


@pytest.mark.parametrize('failure', ['undefined', 'infinite'])
def test_nonfinite_surface_cases_propagate_without_dropping_dsc(failure):
    bad = row(structure='bad', dsc=50)
    if failure == 'undefined':
        bad.update(distance_status='undefined', hd95_pixel=None, assd_pixel=None)
    else:
        bad.update(distance_status='infinite', hd95_pixel=None, assd_pixel=None,
                   case_status='pred_empty', dsc_percent=0, empty_policy='strict')
    summary = summarize_rows([row(dsc=100, hd95=4, assd=2), bad])[0]
    macro = summary['macro']['positive']
    assert macro['distance_status'] == failure
    assert macro['hd95_pixel'] is None and macro['assd_pixel'] is None
    assert macro['dsc_percent'] == (75 if failure == 'undefined' else 50)
    assert macro['n_included'] == 2
    assert macro['n_' + failure] == 1
    assert summary['image_mean']['positive']['distance_status'] == failure


def test_all_ignore_group_excluded_from_macro_denominator():
    ignored = row(structure='ignored')
    ignored.update(binary_metrics(np.zeros((5, 5), np.int8), -np.ones((5, 5), np.int8)))
    summary = summarize_rows([row(dsc=75), ignored])[0]
    assert summary['macro']['positive']['n_groups'] == 1
    assert summary['macro']['positive']['dsc_percent'] == 75
    assert summary['groups']['ignored']['all']['n_excluded'] == 1


def test_abdomen_source_macro_retains_each_ct_and_each_structure():
    rows = [row('abdomen_liver', 'AbdomenUS', 100, source='ct3'),
            row('abdomen_spleen', 'AbdomenUS', 0, source='ct3'),
            row('abdomen_liver', 'AbdomenUS', 80, source='ct4'),
            row('other', 'BUSI', 100)]
    summary = summarize_rows(rows)[0]
    assert summary['abdomen_macro']['positive']['dsc_percent'] == 45
    assert summary['abdomen_by_source']['ct3']['positive']['dsc_percent'] == 50
    assert summary['abdomen_by_source']['ct4']['positive']['dsc_percent'] == 80


def test_postprocessing_variants_never_mix():
    rows = [row(dsc=0, variant='s000'), row(dsc=100, variant='s001')]
    summaries = summarize_rows(rows)
    assert {view['variant']: view['macro']['positive']['dsc_percent'] for view in summaries} == {'s000': 0, 's001': 100}
    groups = seed_groups([result(rows, seed=42), result(rows, seed=43)])
    positive = [group for group in groups if group['cohort'] == 'positive']
    assert len(positive) == 2
    assert {group['variant'] for group in positive} == {'s000', 's001'}
    assert all(group['n'] == 2 for group in positive)


def test_seed_statistics_use_sample_std_and_keep_split_size_selection_separate():
    results = [result([row(dsc=70)], seed=42), result([row(dsc=90)], seed=43),
               result([row(dsc=20)], seed=44, split='val'),
               result([row(dsc=10)], seed=44, size=256),
               result([row(dsc=5)], seed=44, selection='last')]
    groups = [group for group in seed_groups(results) if group['cohort'] == 'positive']
    assert len(groups) == 4
    repeated = next(group for group in groups if group['n'] == 2)
    assert repeated['seeds'] == [42, 43]
    assert repeated['dsc_percent']['mean'] == 80
    assert repeated['dsc_percent']['sample_std'] == pytest.approx(math.sqrt(200))
    assert all(group['dsc_percent']['sample_std'] is None for group in groups if group['n'] == 1)


def test_archived_partial_budget_never_pools_with_complete_seeds():
    results = [result([row(dsc=80)], seed=42),
               result([row(dsc=10)], seed=43, training_status='archived_partial')]
    groups = [group for group in seed_groups(results) if group['cohort'] == 'positive']
    assert len(groups) == 2
    assert all(group['n'] == 1 for group in groups)


def test_distinct_record_protocols_never_pool_as_seed_replicates():
    results = [result([row(dsc=80)], seed=42, comparability_group='split_A'),
               result([row(dsc=10)], seed=43, comparability_group='split_B')]
    groups = [group for group in seed_groups(results) if group['cohort'] == 'positive']
    assert len(groups) == 2
    assert all(group['n'] == 1 for group in groups)


def test_seed_undefined_distances_propagate_with_numeric_dsc():
    bad = row(dsc=60)
    bad.update(distance_status='undefined', hd95_pixel=None, assd_pixel=None)
    groups = [group for group in seed_groups([result([row(dsc=80)], seed=42),
                                             result([bad], seed=43)]) if group['cohort'] == 'positive']
    assert len(groups) == 1
    assert groups[0]['distance_status'] == 'undefined'
    assert groups[0]['hd95_pixel']['mean'] is None
    assert groups[0]['dsc_percent']['mean'] == 70


def test_duplicate_same_seed_evaluation_is_reported_not_silently_lost():
    one = result([row()], seed=42)
    with pytest.raises(ValueError, match='[Dd]uplicate|same seed|[Rr]epeat'):
        seed_groups([one, copy.deepcopy(one)])


def test_nine_independent_networks_have_one_overall_structure_weighted_result():
    domains = ('BUSI', 'MMOTU', 'DDTI', 'AUL', 'FALLMUD', 'Fetal_HC', 'CCA', 'CAMUS', 'AbdomenUS')
    results = []
    for domain in domains:
        # Abdomen contributes five structures, so averaging the nine individual
        # networks' own macro means would incorrectly give 800/9 instead of 800/13.
        rows = ([row('abdomen_' + str(i), domain, dsc=0) for i in range(5)] if domain == 'AbdomenUS'
                else [row(domain + '_organ', domain, dsc=100)])
        results.append(result(rows, kind='independent', arm=domain, domain=domain,
                              experiment='legacy9', family='legacy_baseline'))
    groups = [group for group in seed_groups(results) if group['cohort'] == 'positive'
              and group['kind'] == 'independent_ensemble']
    assert len(groups) == 1
    assert groups[0]['n'] == 1 and groups[0]['seeds'] == [42]
    assert groups[0]['dsc_percent']['mean'] == pytest.approx(800 / 13)


def test_csv_exposes_requested_units_and_distance_failure_counts():
    bad = row()
    bad.update(distance_status='undefined', hd95_pixel=None, assd_pixel=None)
    rows = report_rows([result([bad])])
    assert {'DSC (%)', 'HD95 (pixel)', 'ASSD (pixel)', 'undefined_count', 'penalized_count'} <= rows[0].keys()
    selected = next(item for item in rows if item['scope'] == 'macro' and item['cohort'] == 'positive')
    assert selected['DSC (%)'] == '100.0000'
    assert selected['HD95 (pixel)'] == selected['ASSD (pixel)'] == 'NA'
    assert selected['undefined_count'] == 1
    assert 'DSC (%)' in csv_text(rows).splitlines()[0]


def test_partial_budget_status_is_visible_in_csv_and_markdown():
    partial = result([row()], training_status='archived_partial')
    rows = report_rows([partial])
    assert all(item['training_status'] == 'archived_partial' for item in rows)
    assert all(item['epochs_completed'] == 7 and item['training_epochs'] == 80 for item in rows)
    text = markdown_report([partial], {}, {}, '2026-10-12')
    assert '7/80' in text and 'archived_partial' in text


def test_partial_independent_composition_is_explicit_and_keeps_original():
    one = result([row('breast', 'BUSI')], kind='independent', arm='BUSI', domain='BUSI')
    extended = with_independent_ensembles([one])
    assert len(extended) == 2 and extended[0] is one
    virtual = extended[1]
    assert virtual['job']['ensemble_status'] == 'partial_domains'
    assert virtual['job']['ensemble_complete'] is False
    assert len(virtual['job']['ensemble_missing_domains']) == 8
    assert virtual['checkpoint_sha256'] is None
    rows = report_rows([one])
    assert {item['kind'] for item in rows} == {'independent', 'independent_ensemble'}
    assert all(item['ensemble_status'] == 'partial_domains' for item in rows if item['kind'] == 'independent_ensemble')
    # Public reporters may receive the already-extended list without duplicating
    # virtual rows or pretending the synthetic total is another real run.
    assert report_rows(extended) == report_rows([one])


def test_independent_ensemble_keeps_union_and_structure_and_pooled_counts():
    first = result([row('breast', 'BUSI', dsc=100),
                    row(None, 'BUSI', dsc=100, view='union')],
                   kind='independent', arm='BUSI', domain='BUSI')
    second = result([row('thyroid', 'DDTI', dsc=0), row('thyroid', 'DDTI', dsc=0, image_id='b'),
                     row(None, 'DDTI', dsc=10, view='union')],
                    kind='independent', arm='DDTI', domain='DDTI')
    derived = [item for item in with_independent_ensembles([first, second]) if item.get('derived_from_independent')]
    assert len(derived) == 2
    structure = next(item['metrics'][0] for item in derived if item['metrics'][0]['view'] == 'structure')
    union = next(item['metrics'][0] for item in derived if item['metrics'][0]['view'] == 'union')
    assert structure['macro']['positive']['dsc_percent'] == 50
    assert structure['image_mean']['positive']['dsc_percent'] == pytest.approx(100 / 3)
    assert structure['image_mean']['positive']['n_included'] == 3
    assert union['macro']['positive']['dsc_percent'] == 55


def test_independent_ensemble_distance_undefined_propagates_and_finite_subset_stays_separate():
    bad = row('thyroid', 'DDTI', dsc=60)
    bad.update(distance_status='undefined', hd95_pixel=None, assd_pixel=None)
    first = result([row('breast', 'BUSI', dsc=80, hd95=4, assd=2)],
                   kind='independent', arm='BUSI', domain='BUSI')
    second = result([bad], kind='independent', arm='DDTI', domain='DDTI')
    derived = next(item for item in with_independent_ensembles([first, second]) if item.get('derived_from_independent'))
    image_mean = derived['metrics'][0]['image_mean']['positive']
    assert image_mean['distance_status'] == 'undefined'
    assert image_mean['hd95_pixel'] is None
    assert image_mean['finite_subset_mean']['hd95_pixel'] == 4
    assert image_mean['finite_subset_mean']['n'] == 1
    assert image_mean['n_included'] == 2


def partial_label_row(*, predict_positive, structure='liver_mass', domain='AUL', image_id='partial'):
    target = -np.ones((5, 5), dtype=np.int8)
    target[2, 2] = 1
    prediction = np.zeros((5, 5), dtype=np.int8)
    prediction[2, 2] = int(predict_positive)
    return dict(image_id=image_id, structure=structure, domain=domain, view='structure',
                variant=None, source='', **binary_metrics(prediction, target))


def test_surface_defined_cohort_selection_is_identical_for_empty_and_nonempty_predictions():
    model_a = [partial_label_row(predict_positive=False), row('liver_mass', 'AUL', image_id='complete')]
    model_b = [partial_label_row(predict_positive=True), row('liver_mass', 'AUL', image_id='complete')]
    assert model_a[0]['target_surface_pixels'] == model_b[0]['target_surface_pixels'] == 0
    assert model_a[0]['case_status'] == 'pred_empty'
    assert model_b[0]['distance_status'] == 'undefined'
    first, second = [summarize_rows(rows)[0] for rows in (model_a, model_b)]
    for summary in (first, second):
        subset = summary['macro']['positive_surface_defined']
        assert subset['n_coverage_total'] == 2
        assert subset['n_included'] == 1
        assert subset['n_target_boundary_undefined_excluded'] == 1
        assert subset['dsc_percent'] == 100
        assert subset['hd95_pixel'] == 0
        assert summary['macro']['positive']['n_included'] == 2
        assert summary['macro']['positive']['n_target_boundary_undefined_excluded'] == 0
    assert second['macro']['positive']['distance_status'] == 'undefined'


def test_surface_defined_cohort_keeps_prediction_failure_penalty():
    target = np.zeros((5, 5), dtype=np.int8)
    target[2, 2] = 1
    failed = row(image_id='missed')
    failed.update(binary_metrics(np.zeros_like(target), target))
    rows = [partial_label_row(predict_positive=True, structure='organ_a'), failed]
    summary = summarize_rows(rows)[0]
    subset = summary['macro']['positive_surface_defined']
    assert subset['n_included'] == 1
    assert subset['n_penalized'] == 1
    assert subset['dsc_percent'] == 0
    assert subset['hd95_pixel'] == subset['assd_pixel'] == math.hypot(4, 4)
    assert subset['n_target_boundary_undefined_excluded'] == 1


def test_surface_defined_cohort_retains_prediction_only_undefined_distance():
    # This GT has a retained surface. A prediction with no retained surface is
    # still an evaluation failure and cannot be filtered as if GT were missing.
    bad = row(dsc=60)
    bad.update(distance_status='undefined', hd95_pixel=None, assd_pixel=None, pred_surface_pixels=0)
    assert bad['target_surface_pixels'] > 0
    summary = summarize_rows([bad])[0]
    subset = summary['macro']['positive_surface_defined']
    assert subset['n_included'] == 1 and subset['n_undefined'] == 1
    assert subset['n_target_boundary_undefined_excluded'] == 0
    assert subset['distance_status'] == 'undefined' and subset['hd95_pixel'] is None


def test_surface_defined_all_cohort_keeps_empty_ground_truth():
    empty = row(image_id='empty')
    empty.update(binary_metrics(np.zeros((5, 5), np.int8), np.zeros((5, 5), np.int8)))
    summary = summarize_rows([partial_label_row(predict_positive=True, structure='organ_a'), empty])[0]
    subset = summary['macro']['all_surface_defined']
    assert subset['n_included'] == 1 and subset['dsc_percent'] == 100
    assert subset['n_coverage_total'] == 2 and subset['n_target_boundary_undefined_excluded'] == 1
    assert summary['macro']['positive_surface_defined']['n_included'] == 0
    assert summary['macro']['positive_surface_defined']['n_coverage_total'] == 1
    assert summary['macro']['positive_surface_defined']['n_target_boundary_undefined_excluded'] == 1


def test_surface_cohort_coverage_propagates_through_source_seed_and_independent_ensemble():
    bad = partial_label_row(predict_positive=True, structure='abdomen_liver', domain='AbdomenUS')
    bad['source'] = 'ct3'
    good = row('abdomen_liver', 'AbdomenUS', source='ct3', image_id='complete')
    first = result([bad, good], kind='independent', arm='AbdomenUS', domain='AbdomenUS')
    source = first['metrics'][0]['abdomen_by_source']['ct3']['positive_surface_defined']
    assert source['n_included'] == 1 and source['n_coverage_total'] == 2
    assert source['n_target_boundary_undefined_excluded'] == 1
    extended = with_independent_ensembles([first])
    ensemble = next(item for item in extended if item.get('derived_from_independent'))
    subset = ensemble['metrics'][0]['image_mean']['positive_surface_defined']
    assert subset['n_included'] == 1 and subset['n_coverage_total'] == 2
    assert subset['n_target_boundary_undefined_excluded'] == 1
    groups = [group for group in seed_groups([first]) if group['cohort'] == 'positive_surface_defined']
    assert all(group['coverage_by_seed'][0]['n_target_boundary_undefined_excluded'] == 1 for group in groups)


def test_report_exposes_all_four_cohorts_and_preserves_original_undefined():
    one = result([partial_label_row(predict_positive=True), row('liver_mass', 'AUL')])
    rows = report_rows([one])
    assert set(COHORTS) == {item['cohort'] for item in rows}
    original = next(item for item in rows if item['scope'] == 'macro' and item['cohort'] == 'positive')
    subset = next(item for item in rows if item['scope'] == 'macro' and item['cohort'] == 'positive_surface_defined')
    assert original['HD95 (pixel)'] == 'NA' and original['undefined_count'] == 1
    assert subset['HD95 (pixel)'] == '0.0000'
    assert subset['target_boundary_undefined_excluded'] == 1 and subset['coverage_denominator'] == 2
    text = markdown_report([one], {}, {}, '2026-10-12')
    assert '| positive |' in text and '| positive_surface_defined |' in text
    assert '1/2' in text


@pytest.mark.parametrize('kwargs', [dict(view='unknown'), dict(view='structure', structure=None)])
def test_invalid_view_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        summarize_rows([row(**kwargs)])
