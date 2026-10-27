"""CPU-only, nine-image routing probe for frozen Q01 Region checkpoints.

This is a descriptive validation subset, not an accuracy evaluation or a
full-dataset estimate. It does not alter checkpoints, training, or the queue.
Run from the repository root with its medical environment:

    .conda/bin/python docs/experiments/moe_routing_probe_20261009.py
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path(
        '/ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arms', nargs='+', default=[
        'image_mean', 'organ_exposure', 'organ_exposure_fg_bg'], choices=[
        'image_mean', 'organ_exposure', 'organ_exposure_fg_bg'])
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or len(set(args.arms)) != len(args.arms):
        parser.error('Require positive threads and unique arms')
    root = args.run_root.resolve()
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    source_root = root / 'source_snapshot'
    source_checks = {}
    for name, expected in protocol['source_files'].items():
        actual = sha256(source_root / name)
        if actual != expected:
            raise ValueError(f'Frozen source changed: {name}')
        source_checks[name] = actual
    # Import the original frozen experiment source, not a later working copy.
    sys.path.insert(0, str(source_root))
    import torch
    from unet_moe.region import RegionDataset, RegionUNet

    torch.set_num_threads(args.threads)
    records_path = root / 'records.json'
    records = json.loads(records_path.read_text())
    config = protocol['config']
    if config['size'] != 512 or config['top_k'] != 1:
        raise ValueError('This documented probe requires frozen 512-square, top-1 Q01')
    data = RegionDataset([r for r in records if r['split'] == 'val'],
                         config['size'], config['task_ids'])
    first_per_domain = {}
    for index, group in enumerate(data.groups):
        first_per_domain.setdefault(group[0]['dataset'], index)
    if len(first_per_domain) != 9:
        raise ValueError('Expected exactly nine validation domains')
    samples = []
    for domain, index in first_per_domain.items():
        group = data.groups[index]
        samples.append(dict(domain=domain, dataset_index=index,
                            image=group[0]['image'], group=group[0]['group'],
                            annotated_task_ids=[r['task'] for r in group]))
    report = dict(
        schema='q01_region_routing_cpu_subset_v1',
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        device='cpu', threads=args.threads, torch_version=torch.__version__,
        script=str(Path(__file__).resolve()), script_sha256=sha256(__file__),
        run_root=str(root), source_snapshot=str(source_root),
        protocol=str(protocol_path), protocol_sha256=protocol['protocol_sha256'],
        protocol_file_sha256=sha256(protocol_path), records=str(records_path),
        records_file_sha256=sha256(records_path), verified_source_sha256=source_checks,
        selection_rule='First image of each domain in RegionDataset built from '
                       'frozen records.json filtered to split=val, preserving record order; '
                       'no random sampling, no test images',
        samples=samples, sample_count=len(samples), image_size=config['size'],
        base=config['base'], top_k=config['top_k'], task_ids=config['task_ids'],
        count_definition='One argmax of softmax(router scores) per predicted proposal '
                         'per image per decoder stage, including unannotated proposal slots. '
                         'Counts are not pixel counts, actual anatomical regions, or '
                         'the number of expert forward calls.',
        entropy_definition='Shannon entropy of all nine router probabilities; '
                           'natural logarithm, in nats; normalized entropy divides by ln(9)',
        limitations=[
            'Only one seed and the first validation image per domain; not a random '
            'or representative estimate of the complete validation distribution.',
            'All 16 latent proposal slots are counted, even when the corresponding '
            'output channel is unannotated; the frozen run has no direct proposal supervision.',
            'Concentration in a sampled layer does not prove collapse in all data or all layers.',
            'No Dice comparison, causal ablation, gradient measurement, or test evaluation '
            'is performed by this script.',
            'Softmax probability and a discrete expert assignment do not measure the '
            'magnitude or utility of that expert contribution versus the shared branch.',
        ], runs=[])
    model = RegionUNet(base=config['base'], top_k=config['top_k'],
                       task_ids=config['task_ids']).to('cpu').eval()
    feature_shapes = {}
    hooks = []
    for stage, module in enumerate(model.moes):
        def record_shape(_module, inputs, stage=stage):
            feature_shapes[stage] = list(inputs[0].shape)
        hooks.append(module.register_forward_pre_hook(record_shape))
    try:
        for arm in args.arms:
            checkpoint_path = root / arm / 'region' / f'seed_{args.seed}' / 'best.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if checkpoint['protocol_sha256'] != protocol['protocol_sha256']:
                raise ValueError('Checkpoint does not match frozen protocol')
            model.load_state_dict(checkpoint['model'])
            run = dict(arm=arm, seed=args.seed, checkpoint=str(checkpoint_path),
                       checkpoint_sha256=sha256(checkpoint_path),
                       checkpoint_epoch=checkpoint['epoch'],
                       initialization_sha256=checkpoint['initialization_sha256'], samples=[])
            del checkpoint
            with torch.inference_mode():
                for sample in samples:
                    image, target = data[sample['dataset_index']]
                    output = model(image.unsqueeze(0))
                    stages = []
                    for stage, (coarse, scores) in enumerate(zip(
                            output['coarse'], output['routing'])):
                        probabilities = scores[0].softmax(-1)
                        top_probability, selected = probabilities.max(-1)
                        entropy = -(probabilities * probabilities.clamp_min(
                            torch.finfo(probabilities.dtype).tiny).log()).sum(-1)
                        count = torch.bincount(selected, minlength=probabilities.shape[-1])
                        mask = coarse[0].sigmoid()
                        stages.append(dict(
                            decoder_stage_zero_based=stage,
                            feature_shape=feature_shapes[stage],
                            number_of_proposals=len(selected),
                            argmax_expert_per_proposal=selected.tolist(),
                            argmax_count_per_expert=count.tolist(),
                            experts_selected_count=int(count.gt(0).sum()),
                            softmax_probabilities_per_proposal=probabilities.tolist(),
                            top1_probability_per_proposal=top_probability.tolist(),
                            mean_top1_probability=float(top_probability.mean()),
                            entropy_nats_per_proposal=entropy.tolist(),
                            mean_entropy_nats=float(entropy.mean()),
                            mean_entropy_normalized=float(entropy.mean() /
                                torch.tensor(probabilities.shape[-1]).log()),
                            proposal_mask_mean=float(mask.mean()),
                            proposal_mask_mean_spatial_std=float(mask.flatten(1).std(-1).mean())))
                    run['samples'].append(dict(
                        **sample, positive_channels=target.gt(0).any((1, 2)).nonzero(
                            as_tuple=True)[0].tolist(), stages=stages))
                    del output, image, target
            aggregate = []
            for stage in range(len(model.moes)):
                rows = [sample['stages'][stage] for sample in run['samples']]
                counts = [sum(row['argmax_count_per_expert'][expert] for row in rows)
                          for expert in range(config['experts'])]
                total = sum(counts)
                aggregate.append(dict(
                    decoder_stage_zero_based=stage, total_proposal_assignments=total,
                    argmax_count_per_expert=counts,
                    argmax_fraction_per_expert=[count / total for count in counts],
                    experts_selected_count=sum(count > 0 for count in counts),
                    mean_top1_probability=sum(row['mean_top1_probability'] for row in rows) / len(rows),
                    mean_entropy_nats=sum(row['mean_entropy_nats'] for row in rows) / len(rows)))
            run['aggregate_by_stage'] = aggregate
            report['runs'].append(run)
            print(json.dumps(dict(arm=arm, checkpoint_epoch=run['checkpoint_epoch'],
                                  aggregate_by_stage=aggregate)), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
