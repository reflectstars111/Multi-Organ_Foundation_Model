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
from unet_moe import mechanism_evaluation as evaluation
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


def test_three_arm_pairing_single_seed_and_nonfinite_propagation():
    arms = ['experts0', 'experts1', 'experts9']
    rows = [scalar(42, arm, value) for arm, value in zip(arms, [50, 55, 53])]
    rows += [scalar(42, arm, value, 'hd95_pixel') for arm, value in zip(arms, [None, 3, 2])]
    groups, paired = evaluation.aggregate_paired(rows, [42], arms)
    assert len(paired) == 6
    assert {r['comparison'] for r in paired} == {
        'experts1_minus_experts0', 'experts9_minus_experts0', 'experts9_minus_experts1'}
    assert all(r['sample_std'] is None for r in groups + paired)
    result = next(r for r in paired if r['comparison'] == 'experts9_minus_experts1' and r['metric'] == 'dsc_percent')
    assert result['mean'] == -2
    assert all(r['mean'] is None for r in paired if r['metric'] == 'hd95_pixel' and r['left_arm'] == 'experts0')
    with pytest.raises(ValueError, match='Duplicate'):
        evaluation.aggregate_paired(rows + [rows[0]], [42], arms)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('stage,kind,arms', [('q06','region',['control','routing']),
    ('q07','task_id',['experts0','experts1','experts9'])])
def test_complete_stage_reports_dynamic_matrix_and_absent_guardrails(tmp_path, stage, kind, arms):
    protocol = dict(stage=stage, protocol_sha256='p', config=dict(models=[kind], arms=arms, seeds=[42], epochs=80))
    write(tmp_path / 'main_loss_selection.json', {'selection_sha256': 'main'})
    row = binary_metrics(np.ones((3,3)),np.ones((3,3)))
    row.update(view='structure', structure='breast_lesion', domain='BUSI')
    surface = summarize_rows([row, dict(row, view='union')])
    for arm in arms:
        folder=tmp_path / arm / kind / 'seed_42'
        write(folder / 'run_summary.json', dict(protocol_sha256='p', main_selection_sha256='main',
            epochs_completed=80, test_evaluated=True, best_epoch=12, guardrail_epoch=None))
        for selection,epoch in [('best',12),('last',80)]:
            for split in ['val','test']:
                write(folder / 'evaluation' / f'{selection}_{split}.json', dict(kind=kind, arm=arm,seed=42,
                    protocol_sha256='p',main_selection_sha256='main',checkpoint_epoch=epoch,
                    checkpoint_selection=selection,split=split,surface=surface,diagnostics={}))
    report=evaluation.summarize(tmp_path,protocol)
    assert report['complete']
    assert len([r for r in report['selection_coverage'] if r['selection']=='guardrail'])==len(arms)
    assert all(r['sample_std'] is None for r in report['paired'])
    assert {r['arm'] for r in report['seed_groups']} == set(arms)
    assert 'seed42' in (tmp_path / 'EVALUATION_REPORT.md').read_text()
    path=tmp_path / arms[-1] / kind / 'seed_42/run_summary.json'
    value=json.loads(path.read_text());value['epochs_completed']=79;write(path,value)
    with pytest.raises(ValueError,match='Complete verified'):
        evaluation.summarize(tmp_path,protocol)
