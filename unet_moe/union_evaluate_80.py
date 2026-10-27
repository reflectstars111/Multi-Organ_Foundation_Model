"""Score completed 80-epoch fair models on the common paper5 union target.

The models retain their 16-structure training objective. These union scores
are downstream metrics and cannot be ranked against structure-wise Dice.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts import defer_fair_80 as fair80
from scripts import fair_parallel as parallel
from . import fair_suite, fair_suite_accum, single_channel_suite
from . import union_evaluate as common
from .region import PAPER_ALIGNED_TASK_IDS
from .single_channel_suite import UnionDataset


SCHEMA = 'fair_models_union_evaluation_80_v1'
RUNS = common.RUNS
EXPECTED_RECORD_SHA256 = 'f621896f10234777d3067a019d96a7cb1dae3b6add305bca32815298744315e0'


def load_completed_fair(fair_root):
    """Verify the new plan, both test markers, and all nine completed runs."""
    fair_root = Path(fair_root).resolve()
    if not (fair_root / 'summary.json').is_file():
        raise ValueError('80-epoch fair test summary is incomplete')
    plan_path = fair_root / 'parallel_plan.json'
    if not plan_path.is_file():
        raise ValueError('80-epoch fair plan is missing')
    saved = parallel.read_json(plan_path)
    data_root = Path(saved['common_config']['data_root'])
    with fair80.configured_parallel():
        plan = parallel.load_plan(fair_root, data_root)
        if not fair80.complete_root(plan):
            raise ValueError('80-epoch fair test barrier is incomplete')
    summary = parallel.read_json(fair_root / 'summary.json')
    if (summary.get('protocol_version') != fair_suite_accum.PROTOCOL_VERSION
            or len(summary.get('runs', [])) != len(RUNS)
            or {(row['kind'], row['seed']) for row in summary['runs']} != set(RUNS)
            or any(row.get('epochs_completed') != 80 or row.get('test_evaluated') is not True
                   for row in summary['runs'])):
        raise ValueError('80-epoch fair summary does not contain nine tested runs')
    records = parallel.read_json(parallel.shard_path(plan, '0') / 'records.json')
    if (summary.get('record_sha256') != EXPECTED_RECORD_SHA256
            or parallel.digest(records) != summary['record_sha256']):
        raise ValueError('80-epoch fair records differ from the completed summary')
    return plan, summary, records


def run(args):
    plan, fair_summary, records = load_completed_fair(args.fair_root)
    if args.output.exists():
        raise FileExistsError(f'Output exists: {args.output}')
    size = plan['common_config']['size']
    if size != 512:
        raise ValueError('Union comparison requires the fixed 512x512 protocol')
    selected = [row for row in records if row['split'] == args.split]
    dataset = UnionDataset(selected, size=size, return_multi_target=True)
    if not len(dataset):
        raise ValueError(f'No {args.split} images in the frozen records')
    shard_protocol = parallel.read_json(parallel.shard_path(plan, '0') / 'protocol.json')
    expected_images = sum(shard_protocol['split_images'][args.split].values())
    if len(dataset) != expected_images:
        raise ValueError('Union dataset changed the frozen fair image count')
    args.output.mkdir(parents=True)
    protocol = dict(schema=SCHEMA, fair_root=str(args.fair_root.resolve()),
                    fair_plan_sha256=plan['plan_sha256'],
                    fair_record_sha256=fair_summary['record_sha256'],
                    fair_source_sha256=fair_summary['source_sha256'],
                    split=args.split, images=len(dataset), image_size=size, threshold=.5,
                    target='union of available paper5 annotations; abdomen IDs 3/5/7/255 ignored; AUL mass-only background unknown',
                    prediction='max sigmoid probability across annotated channels only',
                    checkpoint_selection='best_validation_macro_positive_dice',
                    interpretation='downstream union comparison; fair models trained on 16 structure-wise targets',
                    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    shared_scoring_sha256=hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest(),
                    target_source_sha256=hashlib.sha256(
                        Path(single_channel_suite.__file__).read_bytes()).hexdigest())
    parallel.write_json(args.output / 'protocol.json', protocol)
    runs = {(row['kind'], row['seed']): row for row in fair_summary['runs']}
    results = []
    for kind, seed in RUNS:
        metadata = runs[kind, seed]
        checkpoint_path = Path(metadata['checkpoint_directory']) / 'best.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        shard_gpu = '1' if kind == 'region' else '0'
        kind_protocol = parallel.read_json(parallel.shard_path(plan, shard_gpu) / 'protocol.json')
        if (checkpoint.get('kind') != kind or checkpoint.get('seed') != seed
                or checkpoint.get('protocol_version') != fair_suite_accum.PROTOCOL_VERSION
                or checkpoint.get('epoch') != metadata['best_epoch']
                or checkpoint.get('config_sha256') != kind_protocol['config_sha256']
                or checkpoint.get('record_sha256') != fair_summary['record_sha256']
                or checkpoint.get('source_sha256') != fair_summary['source_sha256']
                or checkpoint.get('task_ids') != list(PAPER_ALIGNED_TASK_IDS)):
            raise ValueError(f'Checkpoint provenance mismatch: {checkpoint_path}')
        model = fair_suite.create_model(kind, base=plan['common_config']['base'],
                                        top_k=plan['common_config']['top_k'],
                                        experts=plan['common_config']['experts'])
        model.load_state_dict(checkpoint['model'])
        model = model.to(args.device)
        evaluated = common.evaluate_checkpoint(model, kind, dataset, args.device,
                                                args.batch_size, args.workers)
        run_result = dict(kind=kind, seed=seed, checkpoint=str(checkpoint_path),
                          checkpoint_epoch=checkpoint['epoch'], **evaluated)
        parallel.write_json(args.output / f'{kind}_seed_{seed}.json', run_result)
        results.append(dict(kind=kind, seed=seed, checkpoint=str(checkpoint_path),
                            checkpoint_epoch=checkpoint['epoch'], metrics=evaluated['metrics']))
        print(json.dumps(dict(kind=kind, seed=seed,
                              positive_union_dice=evaluated['metrics']['overall']['positive_dice'],
                              images=evaluated['metrics']['overall']['samples'])), flush=True)
        del model, checkpoint
    summary = dict(protocol=protocol, runs=results, across_seeds=common.summarize_seeds(results))
    parallel.write_json(args.output / 'summary.json', summary)
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
