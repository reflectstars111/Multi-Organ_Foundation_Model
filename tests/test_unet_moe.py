import numpy as np
import torch
from PIL import Image

from unet_moe.model import UNetMoE, segmentation_loss
from unet_moe.data import SegmentationDataset


def test_fallmud_aponeurosis_only_and_nine_experts():
    from unet_moe.data import task_names
    assert task_names(dict(dataset='FALLMUD', label='aponeurosis')) == [(6, 0)]
    assert task_names(dict(dataset='FALLMUD', label='fascicle')) == []
    model = UNetMoE(tasks=21, base=4, experts=9, top_k=2)
    result = model(torch.randn(2, 1, 32, 32), torch.tensor([6, 20]))
    assert result['expert_load'].shape == (4, 9)
    segmentation_loss(result['logits'], torch.zeros_like(result['logits'])).backward()


def test_abdomen_exact_mapping_and_unknown():
    import pytest
    from unet_moe.abdomen import COLORS, rgb_to_ids
    rgb = np.array([COLORS + [(10, 10, 10)]], dtype=np.uint8)
    assert rgb_to_ids(rgb).tolist() == [list(range(9)) + [255]]
    with pytest.raises(ValueError, match='Unknown RGB'):
        rgb_to_ids(np.array([[[99, 0, 100]]], dtype=np.uint8))


def test_ignored_pixels_have_zero_gradient():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = torch.tensor([[[[-1., 1.], [0., -1.]]]])
    segmentation_loss(logits, target).backward()
    assert (logits.grad[target < 0] == 0).all()
    assert logits.grad[target >= 0].abs().sum() > 0


def test_abdomen_indexed_ignore_loading(tmp_path):
    Image.new('L', (32, 32), 127).save(tmp_path/'image.png')
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[:8] = 255
    labels[8:16] = 8
    Image.fromarray(labels).save(tmp_path/'mask.png')
    record = dict(image=str(tmp_path/'image.png'), mask=str(tmp_path/'mask.png'),
                  dataset='AbdomenUS', class_id=8, task=20)
    _, target, _ = SegmentationDataset([record], 32)[0]
    assert (target == -1).sum() == 256
    assert (target == 1).sum() == 256


def test_official_camus_lists_and_errors(tmp_path):
    import pytest
    from unet_moe.data import camus_splits, choose_split
    folder = tmp_path / 'data/CAMUS_public/CAMUS_public/database_split'
    folder.mkdir(parents=True)
    for name, patient in [('training', 'patient0001'), ('validation', 'patient0002'), ('testing', 'patient0003')]:
        (folder / f'subgroup_{name}.txt').write_text(patient)
    official = camus_splits(tmp_path)
    row = dict(dataset='CAMUS', image='images/patient0003/ED.png', split='')
    assert choose_split(row, 'CAMUS/patient0003', 42, official)[0] == 'test'
    assert choose_split(row, 'CAMUS/patient0003', 99, official)[0] == 'test'
    (folder / 'subgroup_training.txt').write_text('patient0003')
    with pytest.raises(ValueError):
        camus_splits(tmp_path)


def test_project_partitions_and_release_holdouts():
    from unet_moe.data import choose_split, validate_splits
    import pytest
    for dataset in ['BUSI', 'AUL', 'DDTI', 'CCA', 'FALLMUD']:
        row = dict(dataset=dataset, split='')
        assignments = [choose_split(row, f'{dataset}/source/{i}', 42)[0] for i in range(200)]
        assert set(assignments) == {'train', 'val', 'test'}
        assert assignments == [choose_split(row, f'{dataset}/source/{i}', 42)[0] for i in range(200)]
    assert choose_split(dict(dataset='MMOTU', split='test'), 'x', 42)[0] == 'test'
    assert {choose_split(dict(dataset='Fetal_HC', split='train'), str(i), 42)[0] for i in range(200)} == {'train', 'val', 'test'}
    with pytest.raises(ValueError, match='leakage'):
        validate_splits([dict(group='g', image='a', split='train'), dict(group='g', image='b', split='test')])


def test_shared_decoder_is_conditioned_and_reloadable():
    model = UNetMoE(tasks=13, base=4, decoder='shared')
    image, task = torch.randn(2, 1, 33, 35), torch.tensor([0, 3])
    result = model(image, task)
    assert result['logits'].shape == image.shape
    assert result['balance'].item() == result['router_z'].item() == 0
    assert not any('router' in name for name, _ in model.named_parameters())
    segmentation_loss(result['logits'], torch.zeros_like(image)).backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    restored = UNetMoE(tasks=13, base=4, decoder='shared')
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(image, task)['logits'], result['logits'])


def test_hc_exam_grouping(tmp_path):
    import csv
    from unet_moe.data import build_records
    folder = tmp_path / 'HC18' / 'processed_png'
    folder.mkdir(parents=True)
    with (folder / 'manifest.csv').open('w', newline='') as stream:
        fields = ['dataset', 'pairing_status', 'image', 'mask', 'split']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in ('010_HC', '010_2HC', '011_HC'):
            writer.writerow(dict(dataset='Fetal_HC', pairing_status='paired',
                                 image=f'images/train/{name}.png', mask=f'masks/train/{name}.png', split='train'))
    records, _ = build_records(tmp_path)
    assert records[0]['group'] == records[1]['group']
    assert records[0]['split'] == records[1]['split']
    assert records[0]['group'] != records[2]['group']


def test_sparse_decoder_backward_and_odd_size():
    torch.manual_seed(5)
    model = UNetMoE(tasks=13, base=4, experts=3, top_k=2)
    result = model(torch.randn(2, 1, 33, 35), torch.tensor([0, 10]))
    assert result['logits'].shape == (2, 1, 33, 35)
    loss = segmentation_loss(result['logits'], torch.zeros_like(result['logits']))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.decoders[0].router.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0
    torch.testing.assert_close(result['expert_load'].sum(-1), torch.ones(4))


def test_palette_mask_read_as_class_indices(tmp_path):
    Image.new('L', (32, 32), 127).save(tmp_path / 'image.png')
    indices = np.zeros((32, 32), dtype=np.uint8)
    indices[:16] = 2
    mask = Image.fromarray(indices).convert('P')
    mask.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * 756)
    mask.save(tmp_path / 'mask.png')
    data = SegmentationDataset([dict(image=str(tmp_path/'image.png'), mask=str(tmp_path/'mask.png'), task=11, class_id=2)], 32)
    _, target, _ = data[0]
    assert target.sum() == 512
