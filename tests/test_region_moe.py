import torch
import numpy as np
import pytest
from PIL import Image
from unet_moe.data import choose_split
from unet_moe.region import (RegionUNet, masked_loss, region_loss, STRUCTURES,
                             TASK_IDS, PAPER_ALIGNED_TASK_IDS)
from unet_moe.region_train import abdomen_positive_weights, domain_sampling_weights


def test_grouped_regions_preserve_missing_and_ignore(tmp_path):
    from unet_moe.region import RegionDataset
    Image.new('L', (32, 32), 127).save(tmp_path/'image.png')
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[:8] = 255
    labels[8:16] = 8
    Image.fromarray(labels).save(tmp_path/'mask.png')
    record = dict(image=str(tmp_path/'image.png'), mask=str(tmp_path/'mask.png'),
                  dataset='AbdomenUS', class_id=8, task=20)
    ds = RegionDataset([record, dict(record, task=13, class_id=1), dict(record, task=7)], 32)
    assert len(ds) == 1
    image, target = ds[0]
    assert image.shape == (1, 32, 32)
    assert (target[-1] == -1).sum() == 256
    assert (target[-1] == 1).sum() == 256
    assert (target[:11] == -1).all()
    assert (target[11, 8:] == 0).all()


def test_image_only_region_routing_backward_reload():
    torch.manual_seed(2)
    model = RegionUNet(base=4)
    image = torch.randn(2, 1, 33, 35)
    output = model(image)
    assert output['logits'].shape == (2, 19, 33, 35)
    assert output['routing'][0].shape == (2, 19, 9)
    assert output['expert_load'].shape == (4, 9)
    target = torch.full_like(output['logits'], -1)
    target[:, 0] = 0; target[:, 0, :10, :10] = 1
    loss = region_loss(output, target); loss.backward()
    assert torch.isfinite(loss)
    assert model.moes[0].proposals.weight.grad.abs().sum() > 0
    assert model.moes[0].router[-1].weight.grad.abs().sum() > 0
    restored = RegionUNet(base=4); restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(image)['logits'], output['logits'])
    assert 'fascicle' not in STRUCTURES and 'ultrasound_outline' not in STRUCTURES


def test_missing_labels_and_all_ignore():
    x = torch.randn(1, 19, 32, 32, requires_grad=True)
    target = torch.full_like(x, -1)
    loss = masked_loss(x, target)
    assert loss.item() == 0
    loss.backward()
    assert x.grad.abs().sum() == 0
    x.grad = None
    target[:, 0] = 0
    masked_loss(x, target).backward()
    assert x.grad[:, 1:].abs().sum() == 0


def test_rare_positive_weight_increases_only_positive_image_gradient():
    target = torch.full((2, 19, 4, 4), -1.)
    target[:, 17] = 0
    target[0, 17, 0, 0] = 1
    weights = torch.ones(19)
    weights[17] = 6
    plain = torch.zeros_like(target, requires_grad=True)
    masked_loss(plain, target).backward()
    balanced = torch.zeros_like(target, requires_grad=True)
    masked_loss(balanced, target, weights).backward()
    torch.testing.assert_close(balanced.grad[0, 17], plain.grad[0, 17] * 6)
    torch.testing.assert_close(balanced.grad[1, 17], plain.grad[1, 17])
    assert balanced.grad[:, :17].abs().sum() == 0


def test_positive_weight_reaches_final_and_coarse_region_heads():
    target = torch.full((1, 19, 4, 4), -1.)
    target[:, 17] = 0
    target[:, 17, 0, 0] = 1
    weights = torch.ones(19)
    weights[17] = 6
    final = torch.zeros_like(target, requires_grad=True)
    coarse = torch.zeros_like(target, requires_grad=True)
    scores = torch.zeros((1, 19, 9), requires_grad=True)
    output = dict(logits=final, coarse=[coarse], routing=[scores])
    region_loss(output, target, weights).backward()
    assert final.grad[0, 17].abs().sum() > 0
    assert coarse.grad[0, 17].abs().sum() > 0
    assert torch.isfinite(scores.grad).all()


def test_abdomen_weights_use_training_presence_and_cap(tmp_path):
    groups = []
    for index in range(10):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[0, 0] = 1
        if index == 0:
            mask[0, 1:8] = np.arange(2, 9)
        path = tmp_path/f'{index}.png'
        Image.fromarray(mask).save(path)
        groups.append([dict(dataset='AbdomenUS', mask=str(path))])
    dataset = type('TrainingGroups', (), dict(groups=groups, size=16))()
    weights, stats = abdomen_positive_weights(dataset, max_weight=8.)
    assert weights[STRUCTURES.index('abdomen_liver')] == 1
    assert weights[STRUCTURES.index('abdomen_adrenal')] == 8
    assert weights[STRUCTURES.index('breast_lesion')] == 1
    assert stats['abdomen_adrenal']['positive_images'] == 1
    assert stats['abdomen_liver']['positive_images'] == 10


def test_paper_aligned_model_keeps_old_checkpoint_shape_available():
    assert len(PAPER_ALIGNED_TASK_IDS) == 16
    model = RegionUNet(base=4, top_k=1, task_ids=PAPER_ALIGNED_TASK_IDS)
    output = model(torch.randn(1, 1, 32, 32))
    assert output['logits'].shape == (1, 16, 32, 32)
    assert not {'abdomen_pancreas', 'abdomen_adrenal', 'abdomen_bone'} & set(model.structures)
    assert model.expert_for_region[-5:] == [8] * 5
    target = torch.full_like(output['logits'], -1)
    target[:, -1] = 0
    target[:, -1, 2:5, 2:5] = 1
    loss = region_loss(output, target, torch.ones(16), model.expert_for_region)
    loss.backward()
    assert torch.isfinite(loss)
    assert RegionUNet(base=4).head.out_channels == len(TASK_IDS)


def test_ct_balanced_sampler_prefers_training_foreground(tmp_path):
    task_ids = PAPER_ALIGNED_TASK_IDS
    rows = []
    for ct, is_positive in [('ct2', True), ('ct2', False), ('ct5', True), ('ct5', False)]:
        image = len(rows)
        mask = np.zeros((16, 16), dtype=np.uint8)
        if is_positive:
            mask[0, 0] = 4
        path = tmp_path/f'{image}.png'
        Image.fromarray(mask).save(path)
        rows.append([dict(dataset='AbdomenUS', group=f'AbdomenUS/{ct}', mask=str(path))])
    rows.append([dict(dataset='BUSI', group='BUSI/one')])
    dataset = type('TrainingGroups', (), dict(groups=rows, size=16, task_ids=task_ids))()
    positive_weights = torch.ones(len(task_ids))
    positive_weights[task_ids.index(16)] = 3
    weights = domain_sampling_weights(dataset, positive_weights, abdomen_strength=.5)
    assert weights[0] > weights[1] and weights[2] > weights[3]
    assert abs(sum(weights[:2]) - .5) < 1e-8
    assert abs(sum(weights[2:4]) - .5) < 1e-8
    assert weights[4] == 1


def test_paper_protocol_preserves_release_test_and_ct_groups():
    train = dict(dataset='AbdomenUS', split='train')
    test = dict(dataset='AbdomenUS', split='test')
    assert choose_split(train, 'AbdomenUS/ct3', 42, abdomen_split='ct7_2')[0] == 'val'
    assert choose_split(train, 'AbdomenUS/ct4', 42, abdomen_split='ct7_2')[0] == 'val'
    assert choose_split(train, 'AbdomenUS/ct5', 42, abdomen_split='ct7_2')[0] == 'train'
    assert choose_split(test, 'AbdomenUS/ct1', 42, abdomen_split='ct7_2')[0] == 'test'
    with pytest.raises(ValueError):
        choose_split(train, 'AbdomenUS/ct1', 42, abdomen_split='ct7_2')
