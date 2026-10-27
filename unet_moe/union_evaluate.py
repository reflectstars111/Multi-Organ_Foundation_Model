"""Score completed fair-suite checkpoints on a common binary union target.

This is a downstream, one-channel *evaluation* of models trained with the
original structure-wise objective. It does not turn those models into a
controlled one-channel training comparison.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader

from scripts import fair_parallel
from . import fair_suite
from . import single_channel_suite
from .region import PAPER_ALIGNED_TASK_IDS
from .single_channel_suite import UnionDataset


RUNS = [(kind, seed) for seed in fair_parallel.SEEDS for kind in fair_parallel.MODELS]
SCORE_FIELDS = ('mean_dice', 'positive_dice', 'empty_false_positive_rate',
                'empty_mean_false_positive_area_fraction', 'positive_miss_rate',
                'positive_zero_overlap_rate')


def union_prediction(logits, multi_target, threshold=.5):
    """OR only annotation-available channels; absent Task-ID logits are zero."""
    if logits.ndim != 4 or logits.shape != multi_target.shape:
        raise ValueError('Expected matching B,C,H,W model logits and targets')
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be between zero and one')
    active = multi_target.ge(0).any((2, 3), keepdim=True)
    if not active.any(1).all():
        raise ValueError('Every image must have at least one annotated channel')
    probability = logits.sigmoid().masked_fill(~active, -1.)
    return probability.amax(dim=1, keepdim=True) >= threshold


def score_image(prediction, target):
    """Score one image on valid union pixels, preserving empty-mask behavior."""
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError('Expected matching H,W prediction and target')
    valid = target >= 0
    valid_pixels = int(valid.sum())
    if not valid_pixels:
        raise ValueError('A union target has no valid pixels')
    truth = target > 0
    pred = prediction.bool() & valid
    truth_pixels = int(truth.sum())
    pred_pixels = int(pred.sum())
    intersection = int((pred & truth).sum())
    false_positive_pixels = int((pred & ~truth).sum())
    denominator = pred_pixels + truth_pixels
    return dict(positive=truth_pixels > 0,
                dice=2 * intersection / denominator if denominator else 1.,
                predicted_pixels=pred_pixels, truth_pixels=truth_pixels,
                valid_pixels=valid_pixels, intersection_pixels=intersection,
                false_positive_pixels=false_positive_pixels,
                false_positive_area_fraction=false_positive_pixels / valid_pixels,
                positive_miss=truth_pixels > 0 and pred_pixels == 0,
                positive_zero_overlap=truth_pixels > 0 and intersection == 0,
                empty_false_positive=truth_pixels == 0 and pred_pixels > 0)


def summarize_images(rows):
    if not rows:
        return None
    positive = [row for row in rows if row['positive']]
    empty = [row for row in rows if not row['positive']]
    return dict(samples=len(rows), positive_samples=len(positive), empty_samples=len(empty),
                mean_dice=statistics.mean(row['dice'] for row in rows),
                positive_dice=statistics.mean(row['dice'] for row in positive) if positive else None,
                empty_false_positive_rate=statistics.mean(row['empty_false_positive'] for row in empty) if empty else None,
                empty_mean_false_positive_pixels=statistics.mean(row['false_positive_pixels'] for row in empty) if empty else None,
                empty_mean_false_positive_area_fraction=statistics.mean(
                    row['false_positive_area_fraction'] for row in empty) if empty else None,
                positive_miss_rate=statistics.mean(row['positive_miss'] for row in positive) if positive else None,
                positive_zero_overlap_rate=statistics.mean(row['positive_zero_overlap'] for row in positive) if positive else None)


def summarize_run(rows):
    domains = defaultdict(list)
    abdomen_cts = defaultdict(list)
    for row in rows:
        domains[row['domain']].append(row)
        if row['domain'] == 'AbdomenUS':
            abdomen_cts[row['source']].append(row)
    per_domain = {domain: summarize_images(items) for domain, items in sorted(domains.items())}
    return dict(overall=summarize_images(rows),
                macro_domain_positive_dice=statistics.mean(
                    value['positive_dice'] for value in per_domain.values()
                    if value['positive_dice'] is not None),
                per_domain=per_domain,
                abdomen_by_ct={ct: summarize_images(items) for ct, items in sorted(abdomen_cts.items())})


def load_completed_fair(fair_root):
    """Require the complete nine-run training and test barrier before reading test."""
    fair_root = Path(fair_root).resolve()
    summary_path = fair_root / 'summary.json'
    if not summary_path.is_file():
        raise ValueError('Fair root has no completed nine-run summary; test evaluation is gated')
    plan_path = fair_root / 'parallel_plan.json'
    if not plan_path.is_file():
        raise ValueError('Expected a frozen parallel fair experiment')
    plan_on_disk = fair_parallel.read_json(plan_path)
    data_root = Path(plan_on_disk['common_config']['data_root'])
    plan = fair_parallel.load_plan(fair_root, data_root)
    summary = fair_parallel.read_json(summary_path)
    if (not summary.get('parallel', {}).get('training_barrier_verified')
            or summary['parallel'].get('plan_sha256') != plan['plan_sha256']
            or not summary.get('test_evaluation_requested')
            or len(summary.get('runs', [])) != len(RUNS)
            or {(run['kind'], run['seed']) for run in summary['runs']} != set(RUNS)
            or any(not run.get('test_evaluated') for run in summary['runs'])):
        raise ValueError('All nine fair runs and the test barrier must complete first')
    shards = {gpu: fair_parallel.load_shard(plan, gpu, require_test=True)
              for gpu in fair_parallel.PARTITIONS}
    expected = fair_parallel.aggregate_shards(plan, shards, require_test=True)
    if 'gpu_observations' in summary['parallel']:
        # Worker.run appends admission observations after aggregating results.
        # Reproduce that metadata, retaining exact checks of all result fields.
        expected['parallel']['gpu_observations'] = {
            gpu: fair_parallel.read_json(fair_root / f'worker_gpu{gpu}.json').get('gpu_at_start')
            for gpu in fair_parallel.PARTITIONS
            if (fair_root / f'worker_gpu{gpu}.json').is_file()}
    if expected != summary:
        raise ValueError('Fair root summary differs from its verified shards')
    records = fair_parallel.read_json(fair_parallel.shard_path(plan, '0') / 'records.json')
    return plan, summary, records


@torch.inference_mode()
def evaluate_checkpoint(model, kind, dataset, device, batch_size=2, workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    rows = []
    offset = 0
    for image, target, domain, source, multi_target in loader:
        image = image.to(device)
        multi_target = multi_target.to(device)
        output = fair_suite.forward_model(model, kind, image, multi_target,
                                          PAPER_ALIGNED_TASK_IDS)
        prediction = union_prediction(output['logits'], multi_target).cpu()
        for index in range(len(image)):
            row = score_image(prediction[index, 0], target[index, 0])
            row.update(domain=domain[index], source=source[index],
                       image=dataset.groups[offset + index][0]['image'])
            rows.append(row)
        offset += len(image)
    if offset != len(dataset):
        raise ValueError('Dataset length changed during evaluation')
    return dict(metrics=summarize_run(rows), per_image=rows)


def _seed_statistics(values):
    values = [value for value in values if value is not None]
    return dict(seeds=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def summarize_seeds(runs):
    result = {}
    for kind in fair_parallel.MODELS:
        matched = [run['metrics'] for run in runs if run['kind'] == kind]
        by_domain = {}
        for domain in sorted({domain for run in matched for domain in run['per_domain']}):
            by_domain[domain] = {field: _seed_statistics([
                run['per_domain'].get(domain, {}).get(field) for run in matched])
                for field in SCORE_FIELDS}
        by_abdomen_ct = {}
        for ct in sorted({ct for run in matched for ct in run['abdomen_by_ct']}):
            by_abdomen_ct[ct] = {field: _seed_statistics([
                run['abdomen_by_ct'].get(ct, {}).get(field) for run in matched])
                for field in SCORE_FIELDS}
        result[kind] = dict(overall={field: _seed_statistics([
            run['overall'][field] for run in matched]) for field in SCORE_FIELDS},
            macro_domain_positive_dice=_seed_statistics([
                run['macro_domain_positive_dice'] for run in matched]),
            per_domain=by_domain, abdomen_by_ct=by_abdomen_ct)
    return result


def run(args):
    plan, fair_summary, records = load_completed_fair(args.fair_root)
    if args.output.exists():
        raise FileExistsError(f'Output exists: {args.output}')
    size = plan['common_config']['size']
    if size != 512:
        raise ValueError('This comparison requires the frozen 512x512 protocol')
    records = [row for row in records if row['split'] == args.split]
    dataset = UnionDataset(records, size=size, return_multi_target=True)
    if not len(dataset):
        raise ValueError(f'No {args.split} images in the frozen records')
    original_protocol = fair_parallel.read_json(
        fair_parallel.shard_path(plan, '0') / 'protocol.json')
    expected_images = sum(original_protocol['split_images'][args.split].values())
    if len(dataset) != expected_images:
        raise ValueError('Union dataset changed the frozen fair image count')
    args.output.mkdir(parents=True)
    protocol = dict(schema='fair_models_union_evaluation_v1', fair_root=str(args.fair_root.resolve()),
                    fair_plan_sha256=plan['plan_sha256'], fair_record_sha256=fair_summary['record_sha256'],
                    fair_source_sha256=fair_summary['source_sha256'], split=args.split,
                    images=len(dataset), image_size=size, threshold=.5,
                    target='union of available paper5 annotations; abdomen IDs 3/5/7/255 ignored; AUL mass-only background unknown',
                    prediction='max sigmoid probability across annotated channels only',
                    checkpoint_selection='original best_validation_macro_positive_dice',
                    interpretation='downstream union comparison; models were trained on structure-wise targets',
                    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    target_source_sha256=hashlib.sha256(
                        Path(single_channel_suite.__file__).read_bytes()).hexdigest())
    fair_parallel.write_json(args.output / 'protocol.json', protocol)
    results = []
    runs = {(row['kind'], row['seed']): row for row in fair_summary['runs']}
    for kind, seed in RUNS:
        metadata = runs[kind, seed]
        checkpoint_path = Path(metadata['checkpoint_directory']) / 'best.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        if (checkpoint.get('kind') != kind or checkpoint.get('seed') != seed
                or checkpoint.get('epoch') != metadata['best_epoch']
                or checkpoint.get('record_sha256') != fair_summary['record_sha256']
                or checkpoint.get('source_sha256') != fair_summary['source_sha256']
                or checkpoint.get('task_ids') != list(PAPER_ALIGNED_TASK_IDS)):
            raise ValueError(f'Checkpoint provenance mismatch: {checkpoint_path}')
        model = fair_suite.create_model(kind, base=plan['common_config']['base'],
                                        top_k=plan['common_config']['top_k'],
                                        experts=plan['common_config']['experts'])
        model.load_state_dict(checkpoint['model'])
        model = model.to(args.device)
        evaluated = evaluate_checkpoint(model, kind, dataset, args.device,
                                        args.batch_size, args.workers)
        run_result = dict(kind=kind, seed=seed, checkpoint=str(checkpoint_path),
                          checkpoint_epoch=checkpoint['epoch'], **evaluated)
        fair_parallel.write_json(args.output / f'{kind}_seed_{seed}.json', run_result)
        results.append(dict(kind=kind, seed=seed, checkpoint=str(checkpoint_path),
                            checkpoint_epoch=checkpoint['epoch'], metrics=evaluated['metrics']))
        print(json.dumps(dict(kind=kind, seed=seed,
                              positive_union_dice=evaluated['metrics']['overall']['positive_dice'],
                              images=evaluated['metrics']['overall']['samples'])), flush=True)
        del model, checkpoint
    summary = dict(protocol=protocol, runs=results, across_seeds=summarize_seeds(results))
    fair_parallel.write_json(args.output / 'summary.json', summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error('Batch size must be positive and workers nonnegative')
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
