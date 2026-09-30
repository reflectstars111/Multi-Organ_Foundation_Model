import torch
from unet_moe.baseline import PlainUNet, image_balanced_loss, DOMAIN_CHANNELS
from unet_moe.baseline_suite import domain_draws


def test_shared_and_independent_heads_and_missing_labels():
    shared = PlainUNet(base=4, outputs=19)
    abdomen = PlainUNet(base=4, outputs=len(DOMAIN_CHANNELS['AbdomenUS']))
    x = torch.randn(2, 1, 33, 35)
    assert shared(x).shape == (2, 19, 33, 35)
    assert abdomen(x).shape == (2, 8, 33, 35)
    logits = shared(x)
    target = torch.full_like(logits, -1)
    target[0, 0, :8, :8] = 1
    target[0, 0, 8:, :] = 0
    loss = image_balanced_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert shared.head.weight.grad.abs().sum() > 0


def test_same_draws_for_shared_and_independent():
    groups = [[{'dataset': name}] for name in DOMAIN_CHANNELS]
    shared = domain_draws(groups, quota=5, seed=42, epoch=1)
    for name, index in zip(DOMAIN_CHANNELS, range(len(groups))):
        only = domain_draws([[{'dataset': name}]], quota=5, seed=42, epoch=1, domain=name)
        assert only == [0]*5
        assert shared.count(index) == 5


def test_weighted_draws_match_between_shared_moe_and_independent():
    names = list(DOMAIN_CHANNELS)
    groups = [[{'dataset': name, 'image': f'{name}_{i}'}]
              for name in names for i in range(3)]
    weights = [1., 2., 5.] * len(names)
    shared = domain_draws(groups, quota=17, seed=42, epoch=4, weights=weights)
    moe = domain_draws(groups, quota=17, seed=42, epoch=4, weights=weights)
    assert shared == moe
    for name in names:
        selected = [groups[i][0]['image'] for i in shared if groups[i][0]['dataset'] == name]
        local = [rows for rows in groups if rows[0]['dataset'] == name]
        independent = domain_draws(local, quota=17, seed=42, epoch=4,
                                   domain=name, weights=[1., 2., 5.])
        assert sorted(selected) == sorted(local[i][0]['image'] for i in independent)


def test_migration_compares_semantic_records_across_paths():
    from scripts.rebase_baseline_run import semantic_key
    row = dict(dataset='Fetal_HC', task=8, class_id=0, split='test', group='Fetal_HC/100',
               split_protocol='project_hc_labeled_70_15_15', label_version='hc18_filled_v1',
               image=r'F:\workspace\pointed_data\HC18\processed_png\images\train\100_HC.png',
               mask=r'F:\workspace\pointed_data\HC18\processed_png\masks_filled\train\100_HC.png')
    migrated = dict(row, image='/ssd1/data/project/pointed_data/HC18/processed_png/images/train/100_HC.png',
                    mask='/ssd1/data/project/pointed_data/HC18/processed_png/masks_filled/train/100_HC.png')
    assert semantic_key(row) == semantic_key(migrated)
