"""Checks for the common binary scoring of completed fair models."""

import json

import pytest
import torch

from unet_moe import union_evaluate as ue


def test_union_prediction_ignores_unannotated_channels_at_half_threshold():
    target = torch.full((2, 3, 2, 2), -1.)
    target[0, 0] = 0
    target[1, 1] = 0
    logits = torch.zeros_like(target)  # Task-ID leaves unqueried outputs at logit 0.
    logits[0, 0] = -10
    logits[1, 1] = -10
    assert not ue.union_prediction(logits, target).any()
    logits[0, 0, 1, 1] = 10
    assert ue.union_prediction(logits, target)[0, 0].sum() == 1
    assert not ue.union_prediction(logits, target)[1].any()


def test_union_image_metrics_exclude_unknown_pixels_and_track_empty_fp():
    positive = ue.score_image(torch.tensor([[1, 0], [1, 1]]),
                              torch.tensor([[1., 0.], [-1., -1.]]))
    assert positive['dice'] == 1
    assert positive['false_positive_pixels'] == 0
    assert not positive['positive_miss']
    empty = ue.score_image(torch.tensor([[1, 0], [1, 1]]),
                           torch.tensor([[0., 0.], [-1., -1.]]))
    assert empty['empty_false_positive']
    assert empty['false_positive_pixels'] == 1
    assert empty['false_positive_area_fraction'] == .5
    with pytest.raises(ValueError, match='no valid pixels'):
        ue.score_image(torch.zeros(2, 2), torch.full((2, 2), -1.))


def test_domain_and_ct_summary_use_image_denominators():
    rows = [
        dict(ue.score_image(torch.tensor([[1, 0]]), torch.tensor([[1., 0.]])),
             domain='AbdomenUS', source='ct3'),
        dict(ue.score_image(torch.tensor([[1, 0]]), torch.tensor([[0., 0.]])),
             domain='AbdomenUS', source='ct4'),
        dict(ue.score_image(torch.tensor([[0, 0]]), torch.tensor([[1., 0.]])),
             domain='BUSI', source='patient1'),
    ]
    result = ue.summarize_run(rows)
    assert result['overall']['positive_samples'] == 2
    assert result['per_domain']['AbdomenUS']['empty_false_positive_rate'] == 1
    assert result['per_domain']['BUSI']['positive_miss_rate'] == 1
    assert result['abdomen_by_ct']['ct3']['positive_dice'] == 1
    assert result['abdomen_by_ct']['ct4']['empty_false_positive_rate'] == 1
    assert result['macro_domain_positive_dice'] == .5


def test_checkpoint_evaluation_scores_annotated_union_only():
    class Dataset(torch.utils.data.Dataset):
        groups = [[{'image': '/synthetic/a.png'}], [{'image': '/synthetic/b.png'}]]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            image = torch.zeros(1, 2, 2)
            image[0, 0, 0] = index
            target = torch.zeros(1, 2, 2)
            multi = torch.full((16, 2, 2), -1.)
            multi[0] = 0
            if index == 0:
                target[0, 0, 0] = 1
                multi[0, 0, 0] = 1
            return image, target, 'AbdomenUS', f'ct{index+3}', multi

    class Model(torch.nn.Module):
        def forward(self, image):
            logits = image.new_zeros((len(image), 16, 2, 2))
            logits[:, 0] = -10
            logits[image[:, 0, 0, 0] == 0, 0, 0, 0] = 10
            # This absent channel is strongly positive but must not contribute.
            logits[:, 15] = 10
            return logits

    result = ue.evaluate_checkpoint(Model(), 'shared', Dataset(), 'cpu', batch_size=2)
    assert result['metrics']['overall']['positive_dice'] == 1
    assert result['metrics']['overall']['empty_false_positive_rate'] == 0
    assert result['metrics']['abdomen_by_ct']['ct3']['positive_dice'] == 1
    assert result['metrics']['abdomen_by_ct']['ct4']['empty_false_positive_rate'] == 0


def test_fair_test_gate_rejects_incomplete_root(tmp_path):
    with pytest.raises(ValueError, match='completed nine-run summary'):
        ue.load_completed_fair(tmp_path)


@pytest.fixture
def completed_fair_root(tmp_path):
    from test_fair_parallel import materialize, metadata_for

    parallel = ue.fair_parallel
    root = tmp_path / 'completed'
    plan = parallel.build_plan(root, tmp_path / 'data')
    parallel.write_json(root / 'parallel_plan.json', plan)
    shards = {gpu: metadata_for(plan, gpu) for gpu in parallel.PARTITIONS}
    for gpu, metadata in shards.items():
        materialize(plan, gpu, metadata)
        parallel.write_json(root / f'worker_gpu{gpu}.json', {
            'state': 'complete', 'gpu_at_start': {'index': int(gpu), 'free_mib': 40000}})
    summary = parallel.aggregate_shards(plan, shards)
    # This is the extra metadata written by Worker.run after aggregation.
    summary['parallel']['gpu_observations'] = {
        gpu: {'index': int(gpu), 'free_mib': 40000} for gpu in parallel.PARTITIONS}
    parallel.write_json(root / 'summary.json', summary)
    return root


def test_completed_fair_accepts_worker_gpu_observations(completed_fair_root):
    _, summary, records = ue.load_completed_fair(completed_fair_root)
    assert len(summary['runs']) == 9
    assert summary['parallel']['gpu_observations']['0']['free_mib'] == 40000
    assert records == [{'record': 'synthetic'}]


@pytest.mark.parametrize('changed', ['metric', 'parallel_contract', 'gpu_observation'])
def test_completed_fair_still_rejects_modified_results(completed_fair_root, changed):
    path = completed_fair_root / 'summary.json'
    summary = json.loads(path.read_text())
    if changed == 'metric':
        summary['runs'][0]['test']['macro_positive_dice'] = .99
    elif changed == 'parallel_contract':
        summary['parallel']['timing_interpretation'] = 'changed'
    else:
        summary['parallel']['gpu_observations']['0']['free_mib'] = 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='differs from its verified shards'):
        ue.load_completed_fair(completed_fair_root)
