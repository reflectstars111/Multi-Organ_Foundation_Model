"""CPU checks for strict 80-epoch provenance and common-union scoring."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import defer_fair_80 as fair80
from scripts import fair_parallel as parallel
from unet_moe import union_evaluate_80 as evaluator


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _completed_root(tmp_path, monkeypatch):
    root = tmp_path / 'fair80'
    plan = fair80.build_plan(root, tmp_path / 'data')
    records = [{'split': 'test', 'image': '/synthetic/test.png'}]
    record_hash = parallel.digest(records)
    monkeypatch.setattr(evaluator, 'EXPECTED_RECORD_SHA256', record_hash)
    summary = dict(protocol_version='fair_image_matched_accum_v1',
                   record_sha256=record_hash, source_sha256=plan['source_sha256'],
                   runs=[dict(kind=kind, seed=seed, epochs_completed=80,
                              test_evaluated=True) for kind, seed in evaluator.RUNS])
    _json(root / 'parallel_plan.json', plan)
    _json(root / 'summary.json', summary)
    _json(root / 'shards/gpu0/records.json', records)
    monkeypatch.setattr(fair80, 'complete_root', lambda checked: checked == plan)
    return root, plan, summary, records


def test_completed_loader_verifies_new_plan_barrier_and_original_records(tmp_path, monkeypatch):
    root, plan, summary, records = _completed_root(tmp_path, monkeypatch)
    old_epochs = parallel.EPOCHS
    assert evaluator.load_completed_fair(root) == (plan, summary, records)
    assert parallel.EPOCHS == old_epochs
    monkeypatch.setattr(fair80, 'complete_root', lambda _: False)
    with pytest.raises(ValueError, match='test barrier'):
        evaluator.load_completed_fair(root)
    assert parallel.EPOCHS == old_epochs
    monkeypatch.setattr(fair80, 'complete_root', lambda _: True)
    monkeypatch.setattr(evaluator, 'EXPECTED_RECORD_SHA256', 'different-records')
    with pytest.raises(ValueError, match='records differ'):
        evaluator.load_completed_fair(root)


def test_completed_loader_rejects_changed_budget_and_incomplete_epochs(tmp_path, monkeypatch):
    root, plan, summary, _ = _completed_root(tmp_path, monkeypatch)
    summary['runs'][0]['epochs_completed'] = 79
    _json(root / 'summary.json', summary)
    with pytest.raises(ValueError, match='nine tested runs'):
        evaluator.load_completed_fair(root)
    plan['common_config']['batch_size'] = 4
    plan['plan_sha256'] = parallel.digest({key: value for key, value in plan.items()
                                          if key != 'plan_sha256'})
    _json(root / 'parallel_plan.json', plan)
    with pytest.raises(ValueError, match='fixed training configuration'):
        evaluator.load_completed_fair(root)


def test_incomplete_gate_never_constructs_test_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, 'UnionDataset',
                        lambda *_args, **_kwargs: pytest.fail('test data opened before gate'))
    args = evaluator.parse_args(['--fair-root', str(tmp_path / 'missing'),
                                 '--output', str(tmp_path / 'union'), '--split', 'test'])
    with pytest.raises(ValueError, match='summary is incomplete'):
        evaluator.run(args)
    assert not args.output.exists()


def test_nine_best_checkpoints_are_scored_on_available_union_channels(tmp_path, monkeypatch):
    root = tmp_path / 'fair80'
    output = tmp_path / 'union80'
    records = [{'split': 'test', 'image': '/synthetic/a.png'},
               {'split': 'test', 'image': '/synthetic/b.png'}]
    record_hash = parallel.digest(records)
    plan = dict(output=str(root), common_config=dict(size=512, base=24, top_k=1, experts=9),
                plan_sha256='plan80', shards={str(gpu): dict(relative_output=f'shards/gpu{gpu}')
                                             for gpu in (0, 1)})
    summary = dict(record_sha256=record_hash, source_sha256='source80', runs=[])
    for gpu in (0, 1):
        _json(root / f'shards/gpu{gpu}/protocol.json',
              dict(config_sha256=f'config{gpu}', split_images={'test': {'BUSI': 2}}))

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, supplied, size, return_multi_target):
            assert supplied == records and size == 512 and return_multi_target
            self.groups = [[row] for row in records]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            image = torch.full((1, 2, 2), float(index))
            target = torch.zeros(1, 2, 2)
            multi = torch.full((16, 2, 2), -1.)
            multi[0] = 0
            if index == 0:
                target[0, 0, 0] = 1
                multi[0, 0, 0] = 1
            return image, target, 'BUSI', 'case', multi

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(-10.))

    def forward(model, kind, image, target, task_ids):
        logits = image.new_full((len(image), 16, 2, 2), 10.)
        logits[:, 0] = model.bias
        logits[image[:, 0, 0, 0] == 0, 0, 0, 0] = 10.
        return {'logits': logits}

    for kind, seed in evaluator.RUNS:
        folder = root / kind / f'seed_{seed}'
        folder.mkdir(parents=True)
        summary['runs'].append(dict(kind=kind, seed=seed, best_epoch=17,
                                    checkpoint_directory=str(folder)))
        torch.save(dict(kind=kind, seed=seed, epoch=17,
                        protocol_version='fair_image_matched_accum_v1',
                        config_sha256='config1' if kind == 'region' else 'config0',
                        record_sha256=record_hash, source_sha256='source80',
                        task_ids=list(evaluator.PAPER_ALIGNED_TASK_IDS),
                        model=Model().state_dict()), folder / 'best.pt')
    monkeypatch.setattr(evaluator, 'load_completed_fair', lambda _: (plan, summary, records))
    monkeypatch.setattr(evaluator, 'UnionDataset', Dataset)
    monkeypatch.setattr(evaluator.fair_suite, 'create_model', lambda *_args, **_kwargs: Model())
    monkeypatch.setattr(evaluator.fair_suite, 'forward_model', forward)
    result = evaluator.run(SimpleNamespace(fair_root=root, output=output, split='test',
                                           device='cpu', batch_size=2, workers=0))
    assert result['protocol']['schema'] == evaluator.SCHEMA
    assert len(result['runs']) == 9
    assert all(row['checkpoint_epoch'] == 17 for row in result['runs'])
    assert all(row['metrics']['overall']['positive_dice'] == 1 for row in result['runs'])
    assert all(row['metrics']['overall']['empty_false_positive_rate'] == 0
               for row in result['runs'])
    assert all(result['across_seeds'][kind]['overall']['positive_dice']['seeds'] == 3
               for kind in ('shared', 'task_id', 'region'))
