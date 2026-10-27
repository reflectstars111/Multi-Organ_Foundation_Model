import copy
import json
import math
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from unet_moe import organ_diagnostics
from unet_moe import refinement_evaluation as evaluation
from unet_moe.region import PAPER_ALIGNED_TASK_IDS
from unet_moe.reevaluation_report import summarize_rows
from unet_moe.surface_metrics import binary_metrics


class FixedRegion(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.calls = 0
        self.marker = nn.Parameter(torch.tensor(1.))
        self.norm = nn.BatchNorm2d(1)

    def forward(self, image):
        self.calls += 1
        return {'logits': self.logits[image[:, 0, 0, 0].long()]}


def batches(target, batch_size=2, generator=None):
    images = torch.arange(float(len(target)))[:, None, None, None].expand(len(target), 1, *target.shape[-2:])
    return DataLoader(TensorDataset(images, target), batch_size=batch_size, generator=generator)


def group(domain, task, image='image.png', **kwargs):
    return [dict(dataset=domain, task=task, image=image, group=f'{domain}/case', **kwargs)]


def test_one_pass_matches_organ_metrics_and_keeps_union_unknown_background():
    target = torch.full((2, 16, 4, 4), -1.)
    target[0, 0] = 0
    target[0, 0, 1:3, 1:3] = 1
    mass = PAPER_ALIGNED_TASK_IDS.index(4)
    target[1, mass] = 0
    target[1, mass, 1:3, 1:3] = 1
    logits = torch.full_like(target, 10.)  # Unannotated outputs must not create union FP.
    logits[0, 0] = target[0, 0] * 20 - 10
    model = FixedRegion(logits)
    groups = [group('BUSI', 0), group('AUL', 4)]
    result = evaluation.evaluate(model, 'region', batches(target), 'cpu', groups)
    assert model.calls == 1
    reference = organ_diagnostics.evaluate(model, 'region', batches(target), 'cpu', groups)
    assert result['diagnostics'] == reference
    union = [row for row in result['rows'] if row['view'] == 'union']
    assert union[0]['dsc_percent'] == 100
    assert union[0]['pred_pixels'] == 4
    assert union[1]['dsc_percent'] == 100
    assert union[1]['valid_pixels'] == 4
    assert union[1]['distance_status'] == 'undefined'
    assert union[1]['target_surface_pixels'] == 0
    surface = next(view for view in result['surface'] if view['view'] == 'union')
    assert surface['macro']['all']['distance_status'] == 'undefined'
    assert surface['macro']['all_surface_defined']['n_groups'] == 1
    assert surface['macro']['all_surface_defined']['n_target_boundary_undefined_excluded'] == 1


def test_abdomen_union_ignores_excluded_classes_and_retains_empty_penalty(tmp_path):
    indexed = np.zeros((5, 5), dtype=np.uint8)
    indexed[1, 1] = 1
    indexed[3, 3] = 4
    indexed[0, 0] = 3
    indexed[0, 1] = 255
    mask_path = tmp_path / 'mask.png'
    Image.fromarray(indexed).save(mask_path)
    target = torch.full((1, 16, 5, 5), -1.)
    liver, vessels = [PAPER_ALIGNED_TASK_IDS.index(task) for task in (13, 16)]
    target[0, liver] = torch.from_numpy((indexed == 1).astype(np.float32))
    target[0, vessels] = torch.from_numpy((indexed == 4).astype(np.float32))
    logits = torch.full_like(target, -10.)
    logits[0, liver, 0, :2] = 10  # Predictions on excluded IDs never enter union.
    groups = [[dict(dataset='AbdomenUS', task=task, image='a.png', group='AbdomenUS/ct3',
                    mask=str(mask_path)) for task in (13, 16)]]
    result = evaluation.evaluate(FixedRegion(logits), 'region', batches(target), 'cpu', groups)
    union = next(row for row in result['rows'] if row['view'] == 'union')
    assert union['valid_pixels'] == 23
    assert union['pred_pixels'] == 0
    assert union['distance_penalized'] is True
    assert union['hd95_pixel'] == pytest.approx(math.hypot(4, 4))
    assert union['assd_pixel'] == union['hd95_pixel']
    assert result['diagnostics']['abdomen_by_source']['ct3']['per_structure']['abdomen_vessels']['positive_miss_rate'] == 1


def test_task_id_queries_include_annotated_empty_channels_and_scatter_correctly():
    target = torch.full((2, 16, 3, 3), -1.)
    for batch, task in [(0, 0), (1, 3), (1, 4)]:
        target[batch, PAPER_ALIGNED_TASK_IDS.index(task)] = 0
    target[0, 0, 1, 1] = 1
    target[1, PAPER_ALIGNED_TASK_IDS.index(3), 1, 1] = 1

    class TaskModel(nn.Module):
        def forward(self, image, tasks):
            self.queries = tasks.tolist()
            selected = torch.stack([target[int(i[0, 0, 0]), PAPER_ALIGNED_TASK_IDS.index(t)]
                                    for i, t in zip(image, tasks.tolist())])
            return {'logits': (selected[:, None] * 20 - 10)}

    model = TaskModel()
    groups = [group('BUSI', 0), group('AUL', 3) + group('AUL', 4)]
    report = evaluation.evaluate(model, 'task_id', batches(target), 'cpu', groups)
    assert model.queries == [0, 3, 4]
    assert len(report['rows']) == 5
    assert all(row['dsc_percent'] == 100 for row in report['rows'])
    assert report['diagnostics']['per_structure']['liver_mass']['empty_samples'] == 1


def test_evaluation_restores_rng_buffers_gradients_modes_and_loader_generator():
    target = torch.full((1, 16, 3, 3), -1.)
    target[:, 0] = 0

    class Mutating(FixedRegion):
        def forward(self, image):
            random.random()
            np.random.rand()
            torch.rand(1)
            self.norm.running_mean.add_(1)
            return super().forward(image)

    model = Mutating(torch.full_like(target, -10.))
    model.train()
    model.norm.eval()
    model.marker.grad = torch.tensor(5.)
    gradient = model.marker.grad
    generator = torch.Generator().manual_seed(7)
    loader = batches(target, generator=generator)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
              generator.get_state().clone(), copy.deepcopy(model.state_dict()))
    result = evaluation.evaluate(model, 'region', loader, 'cpu', [group('BUSI', 0)])
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert torch.equal(generator.get_state(), states[3])
    assert all(torch.equal(value, states[4][name]) for name, value in model.state_dict().items())
    assert model.training and not model.norm.training
    assert model.marker.grad is gradient
    assert model.marker.grad.item() == 5
    assert all(row['dsc_percent'] == 100 and row['hd95_pixel'] == 0 for row in result['rows'])


def test_evaluation_restores_state_even_if_metadata_fails():
    target = torch.zeros(1, 16, 2, 2)
    model = FixedRegion(torch.full_like(target, -10.))
    model.train()
    generator = torch.Generator().manual_seed(10)
    state = generator.get_state().clone()
    with pytest.raises(ValueError, match='CT source'):
        evaluation.evaluate(model, 'region', batches(target, generator=generator), 'cpu', [group('BUSI', 0)])
    assert model.training
    assert torch.equal(generator.get_state(), state)


def test_evaluation_rejects_random_or_missing_groups():
    target = torch.zeros(2, 16, 2, 2)
    model = FixedRegion(target)
    loader = batches(target)
    with pytest.raises(ValueError, match='metadata'):
        evaluation.evaluate(model, 'region', loader, 'cpu')
    with pytest.raises(ValueError, match='length'):
        evaluation.evaluate(model, 'region', loader, 'cpu', [group('BUSI', 0)])
    random_loader = DataLoader(loader.dataset, shuffle=True)
    with pytest.raises(ValueError, match='sequential'):
        evaluation.evaluate(model, 'region', random_loader, 'cpu', [group('BUSI', 0)] * 2)


def scalar(seed, arm, value, metric='dsc_percent'):
    return dict(kind='region', seed=seed, arm=arm, value=value, selection='guardrail', split='test',
                view='structure', scope='macro', group='all', cohort='positive', metric=metric)


def test_pairing_uses_same_seeds_and_propagates_nonfinite_distances():
    rows = [scalar(42, 'control', 60), scalar(43, 'control', 80), scalar(42, 'refined', 65),
            scalar(44, 'refined', 90), scalar(42, 'control', None, 'hd95_pixel'),
            scalar(42, 'refined', 2, 'hd95_pixel')]
    groups, paired = evaluation.aggregate_paired(rows, [42, 43, 44])
    dsc = next(row for row in paired if row['metric'] == 'dsc_percent')
    assert dsc['paired_seeds'] == [42]
    assert dsc['mean'] == 5
    assert dsc['n_paired'] == 1
    assert dsc['missing_pair_seeds'] == [43, 44]
    hd95 = next(row for row in paired if row['metric'] == 'hd95_pixel')
    assert hd95['mean'] is None and hd95['n_numeric'] == 0
    control = next(row for row in groups if row['metric'] == 'dsc_percent' and row['arm'] == 'control')
    assert control['mean'] == 70 and control['missing_seeds'] == [44]
    with pytest.raises(ValueError, match='Duplicate'):
        evaluation.aggregate_paired(rows + [rows[0]], [42, 43, 44])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed_matrix(tmp_path):
    protocol = dict(protocol_sha256='protocol', config=dict(models=['region', 'task_id'], seeds=[42, 43, 44], epochs=80))
    write(tmp_path / 'main_loss_selection.json', {'selection_sha256': 'main'})
    surface_row = binary_metrics(np.ones((2, 2)), np.ones((2, 2)))
    surface_row.update(view='structure', structure='breast_lesion', domain='BUSI')
    union_row = dict(surface_row, view='union')
    surface = summarize_rows([surface_row, union_row])
    for kind in protocol['config']['models']:
        for arm in ('control', 'refined'):
            for seed in protocol['config']['seeds']:
                folder = tmp_path / arm / kind / f'seed_{seed}'
                summary = dict(protocol_sha256='protocol', main_selection_sha256='main', kind=kind, arm=arm,
                               seed=seed, epochs_completed=80, test_evaluated=True, best_epoch=65, guardrail_epoch=None)
                write(folder / 'run_summary.json', summary)
                for selection, epoch in [('best', 65), ('last', 80)]:
                    for split in ('val', 'test'):
                        value = 0.50 if arm == 'control' else 0.55
                        result = dict(protocol_sha256='protocol', main_selection_sha256='main', checkpoint_epoch=epoch,
                            kind=kind, arm=arm, seed=seed, checkpoint_selection=selection, split=split, surface=surface,
                            diagnostics=dict(
                                per_structure={'abdomen_vessels': {'positive_dice': value, 'positive_miss_rate': 1-value}},
                                abdomen_by_source={'ct3': {'macro_positive_dice': value}}))
                        write(folder / 'evaluation' / f'{selection}_{split}.json', result)
    return tmp_path, protocol


def test_summary_writes_full_paired_reports_and_exposes_missing_guardrails(completed_matrix):
    root, protocol = completed_matrix
    report = evaluation.summarize(root, protocol)
    assert report['complete'] and report['main_selection_sha256'] == 'main'
    guardrail = [row for row in report['selection_coverage'] if row['selection'] == 'guardrail']
    assert len(guardrail) == 12 and all(row['selected'] is False for row in guardrail)
    vessel = next(row for row in report['paired'] if row['kind'] == 'region'
                  and row['selection'] == 'best' and row['split'] == 'test'
                  and row['metric'] == 'per_structure.abdomen_vessels.positive_dice_percent')
    assert vessel['n_paired'] == 3 and vessel['mean'] == pytest.approx(5)
    assert any(row['metric'] == 'abdomen_by_source.ct3.macro_positive_dice_percent' for row in report['paired'])
    for name in ('evaluation_summary.json', 'evaluation_metrics.csv', 'evaluation_paired.csv',
                 'evaluation_per_group.csv', 'EVALUATION_REPORT.md'):
        assert (root / name).is_file()
    markdown = (root / 'EVALUATION_REPORT.md').read_text()
    assert 'positive_surface_defined' in markdown
    assert '视觉上下文与任务调制' in markdown
    assert 'refined' in markdown
    assert not (root / 'summary.json').exists()  # The suite owns the authoritative summary.


def test_summary_refuses_partial_matrix_or_wrong_main_loss_hash(completed_matrix):
    root, protocol = completed_matrix
    path = root / 'control/region/seed_42/run_summary.json'
    record = json.loads(path.read_text())
    record['epochs_completed'] = 79
    write(path, record)
    with pytest.raises(ValueError, match='Complete verified'):
        evaluation.summarize(root, protocol)
    record['epochs_completed'] = 80
    write(path, record)
    path = root / 'control/region/seed_42/evaluation/best_test.json'
    result = json.loads(path.read_text())
    result['main_selection_sha256'] = 'wrong'
    write(path, result)
    with pytest.raises(ValueError, match='identity differs'):
        evaluation.summarize(root, protocol)


def test_summary_refuses_evaluation_for_unselected_guardrail(completed_matrix):
    root, protocol = completed_matrix
    write(root / 'control/region/seed_42/evaluation/guardrail_val.json', {})
    with pytest.raises(ValueError, match='unselected guardrail'):
        evaluation.summarize(root, protocol)
