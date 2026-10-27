import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from unet_moe import abdomen_recovery as recovery
from unet_moe.region import DOMAINS, RegionDataset


def metric(abd=.5, overall=.7, fp_area=.04, fp_rate=.4, miss=.2, liver_fp=.6,
           kidney_fp_area=.03, kidney_dice=.5):
    per_structure = {recovery.TASKS[task]: dict(
        positive_dice=(kidney_dice if task == 14 else abd),
        empty_false_positive_rate=(liver_fp if task == 13 else fp_rate),
        empty_mean_false_positive_area_fraction=(kidney_fp_area if task == 14 else fp_area),
        positive_zero_overlap_rate=miss) for task in recovery.ABDOMEN_TASKS}
    return dict(abdomen_positive_dice=abd, overall_positive_dice=overall,
                abdomen_empty_fp_area=fp_area, abdomen_empty_fp_rate=fp_rate,
                abdomen_zero_overlap=miss, liver_empty_fp_rate=liver_fp,
                kidney_empty_fp_area=kidney_fp_area, kidney_positive_dice=kidney_dice,
                per_structure=per_structure)


def test_checkpoint_guards_reject_all_empty_prediction():
    initial = metric()
    all_empty = metric(abd=.01, overall=.5, fp_area=0., fp_rate=0., miss=1.)
    assert not recovery.eligible_checkpoint(all_empty, initial)
    assert recovery.eligible_checkpoint(metric(abd=.52, overall=.7, fp_area=.039,
                                               fp_rate=.39, miss=.19), initial)


def test_sampling_changes_only_abdomen_domain_and_is_reproducible():
    groups = []
    for domain in DOMAINS[:-1]:
        groups.extend([[dict(dataset=domain, image=f'{domain}/{n}')] for n in range(2)])
    groups.extend([[dict(dataset='AbdomenUS', image=f'AbdomenUS/{n}')] for n in range(4)])
    positive = {f'AbdomenUS/{n}': n < 2 for n in range(4)}
    control, weighted = recovery.ARMS[0], recovery.ARMS[4]
    changed = False
    for epoch in range(1, 9):
        baseline = recovery.sampled_indices(groups, positive, control, 42, epoch, quota=12)
        alternative = recovery.sampled_indices(groups, positive, weighted, 42, epoch, quota=12)
        assert baseline == recovery.sampled_indices(groups, positive, control, 42, epoch, quota=12)
        assert alternative == recovery.sampled_indices(groups, positive, weighted, 42, epoch, quota=12)
        assert [index for index in baseline if groups[index][0]['dataset'] != 'AbdomenUS'] == [
            index for index in alternative if groups[index][0]['dataset'] != 'AbdomenUS']
        assert sum(groups[index][0]['dataset'] == 'AbdomenUS' for index in alternative) == 12
        changed |= baseline != alternative
    assert changed


def test_objective_rare_positive_weight_and_small_empty_penalty():
    channel = len(recovery.PAPER_ALIGNED_TASK_IDS)
    logits = torch.zeros(1, channel, 4, 4, requires_grad=True)
    target = torch.full_like(logits, -1.)
    target[:, recovery.PAPER_ALIGNED_TASK_IDS.index(16)] = 0.
    target[:, recovery.PAPER_ALIGNED_TASK_IDS.index(16), 0, 0] = 1.
    target[:, recovery.PAPER_ALIGNED_TASK_IDS.index(13)] = 0.
    output = {'logits': logits}
    base = recovery.objective(output, target, recovery.ARMS[0])
    rare = recovery.objective(output, target, recovery.ARMS[3])
    empty = recovery.objective(output, target, recovery.ARMS[1])
    both = recovery.objective(output, target, recovery.ARMS[5])
    assert rare > base
    assert empty > base
    assert both > rare and both > empty
    both.backward()
    assert torch.isfinite(logits.grad).all()


def test_paired_three_seed_selection_and_miss_guard():
    rows = []
    for seed in recovery.SEEDS:
        for arm in recovery.ARMS:
            values = metric()
            if arm['name'] == 'rare_positive_x2':
                values = metric(abd=.525, fp_area=.03, fp_rate=.38, miss=.18)
            if arm['name'] == 'empty_0p01':
                values = metric(abd=.02, overall=.6, fp_area=0., fp_rate=0., miss=.95)
            rows.append(dict(seed=seed, arm=arm['name'], validation=values))
    result = recovery.compare_runs(rows)
    assert result['selected_candidate'] == 'rare_positive_x2'
    assert not result['candidate_comparisons']['empty_0p01']['selected']
    rows = [dict(row, validation=metric(abd=.525, kidney_fp_area=.1))
            if row['arm'] == 'rare_positive_x2' else row for row in rows]
    assert not recovery.compare_runs(rows)['candidate_comparisons']['rare_positive_x2']['selected']
    with pytest.raises(ValueError, match='six arms'):
        recovery.compare_runs(rows[:-1])


def test_reconcile_best_after_crash_between_last_and_best(tmp_path):
    arm = recovery.ARMS[0]
    plan = {'plan_sha256': 'plan', 'initial_checkpoints': {'42': {'sha256': 'initial'}}}
    old = dict(plan_sha256='plan', initial_sha256='initial', seed=42,
               arm=arm['name'], epoch=1, best_epoch=1, marker='old')
    latest = dict(old, epoch=2, best_epoch=2, marker='new')
    path = tmp_path / 'best.pt'
    torch.save(old, path)
    recovery.ensure_best_checkpoint(path, latest, plan, 42, arm)
    assert torch.load(path, weights_only=True)['marker'] == 'new'


def test_paired_history_gate_allows_only_declared_abdomen_sampling(tmp_path):
    plan = {'output': str(tmp_path), 'plan_sha256': 'plan', 'initial_checkpoints':
            {str(seed): {'sha256': f'init-{seed}'} for seed in recovery.SEEDS}}
    for seed in recovery.SEEDS:
        for arm in recovery.ARMS:
            folder = tmp_path / arm['name'] / f'seed_{seed}'
            folder.mkdir(parents=True)
            result = dict(seed=seed, arm=arm['name'], epochs_completed=recovery.EPOCHS,
                          test_evaluated=False, initial_sha256=f'init-{seed}',
                          plan_sha256='plan')
            (folder / 'result.json').write_text(json.dumps(result))
            rows = [dict(epoch=0, sample_count=0)]
            for epoch in range(1, recovery.EPOCHS + 1):
                sampling = arm['abdomen_sampling_weight'] > 1
                rows.append(dict(epoch=epoch, sample_count=len(DOMAINS) * recovery.QUOTA,
                                 annotated_channel_queries=2000,
                                 sample_sha256=('b' if sampling else 'a') * 64,
                                 non_abdomen_sample_sha256='c' * 64,
                                 abdomen_sample_sha256=('e' if sampling else 'd') * 64))
            (folder / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    recovery.validate_paired_histories(plan)
    path = tmp_path / 'rare_positive_x2' / 'seed_42' / 'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]['sample_sha256'] = 'f' * 64
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='Undeclared sampling difference'):
        recovery.validate_paired_histories(plan)


def test_validation_ct_diagnostics_reports_vessel_recall_and_kidney_fp(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'SIZE', 4)
    rows = []
    for ct, bright, vessel_pixel in [('ct3', 0, (0, 0)), ('ct4', 255, (1, 1))]:
        image = tmp_path / f'{ct}_image.png'
        mask = tmp_path / f'{ct}_mask.png'
        Image.fromarray(np.full((4, 4), bright, dtype=np.uint8)).save(image)
        indexed = np.zeros((4, 4), dtype=np.uint8)
        indexed[vessel_pixel] = 4
        Image.fromarray(indexed).save(mask)
        for task in (14, 16):
            rows.append(dict(split='val', dataset='AbdomenUS', task=task,
                             class_id=task - 12, image=str(image), mask=str(mask),
                             group=f'AbdomenUS/{ct}'))
    dataset = RegionDataset(rows, 4, recovery.PAPER_ALIGNED_TASK_IDS)
    class FakeModel(torch.nn.Module):
        def forward(self, images):
            logits = torch.full((len(images), len(recovery.PAPER_ALIGNED_TASK_IDS), 4, 4), -10.)
            for i, image in enumerate(images):
                if float(image.mean()) > .5:
                    logits[i, recovery.PAPER_ALIGNED_TASK_IDS.index(16), 1, 1] = 10.
                else:
                    logits[i, recovery.PAPER_ALIGNED_TASK_IDS.index(14), 0, 0] = 10.
            return {'logits': logits}
    result = recovery.validation_ct_diagnostics(FakeModel(), dataset, torch.device('cpu'))
    assert result['ct3']['vessel_zero_overlap_rate'] == 1
    assert result['ct3']['kidney_empty_mean_fp_area_fraction'] == pytest.approx(1 / 16)
    assert result['ct4']['vessel_mean_recall'] == 1
    assert result['ct4']['vessel_zero_overlap_rate'] == 0


def test_prepare_accepts_only_existing_controller_files(tmp_path, monkeypatch):
    mask_positive = tmp_path / 'positive.png'
    mask_empty = tmp_path / 'empty.png'
    positive = np.zeros((4, 4), dtype=np.uint8)
    positive[0, 0] = 4
    Image.fromarray(positive).save(mask_positive)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(mask_empty)
    records = []
    def image_path(domain, split, index):
        path = tmp_path / domain / split / f'{index}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'{domain}/{split}/{index}'.encode())
        return str(path)
    for split in ('train', 'val'):
        for domain in DOMAINS:
            if domain == 'AbdomenUS' and split == 'train':
                for index, path in enumerate((mask_positive, mask_empty)):
                    records.append(dict(split=split, dataset=domain, task=16, class_id=4,
                                        image=image_path(domain, split, index), mask=str(path)))
            else:
                task = 16 if domain == 'AbdomenUS' else 0
                records.append(dict(split=split, dataset=domain, task=task, class_id=4,
                                    image=image_path(domain, split, 0), mask=str(mask_empty)))
    records.append(dict(split='test', dataset='AbdomenUS', task=16, class_id=4,
                        image='AbdomenUS/test/0', mask=str(mask_empty)))
    source = Path(recovery.__file__).parent
    source_files = {name: recovery.file_sha256(source / name) for name in recovery.SOURCE_NAMES}
    fair_plan = dict(plan_sha256='fair-plan', source_files=source_files,
                     source_sha256='fair-source')
    counts = {split: dict((domain, 2 if split == 'train' and domain == 'AbdomenUS' else 1)
                          for domain in DOMAINS) for split in ('train', 'val')}
    fair_protocol = dict(record_sha256=recovery.digest(records), split_images=counts)
    checkpoints = {str(seed): dict(path=str(tmp_path / f'best_{seed}.pt'), sha256='hash')
                   for seed in recovery.SEEDS}
    monkeypatch.setattr(recovery, 'completed_fair', lambda fair_root: (fair_plan, {'complete': True}))
    monkeypatch.setattr(recovery, 'fair_records_and_checkpoints',
                        lambda plan: (records, fair_protocol, checkpoints))
    output = tmp_path / 'recovery'
    output.mkdir()
    (output / 'controller_status.json').write_text('{}')
    (output / 'recovery.log').write_text('')
    plan = recovery.prepare_plan(tmp_path / 'fair', output)
    assert recovery.prepare_plan(tmp_path / 'fair', output) == plan
    assert json.loads((output / 'plan.json').read_text())['design']['abdomen_positive_images'] == 1
    assert plan['design']['size'] == 512
    assert plan['train_val_content']['files'] == 21
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(mask_empty)
    with pytest.raises(ValueError, match='differs from this frozen plan'):
        recovery.prepare_plan(tmp_path / 'fair', output)
