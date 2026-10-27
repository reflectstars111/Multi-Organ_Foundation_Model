"""Exploratory abdomen threshold/component screen for frozen fair checkpoints.

Postprocessing operates on predictions alone. Ignore labels are applied only
when scoring a completed prediction, never when choosing its components.
The two validation CTs are used for tuning/checking, and the test is untouched.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np
import torch

from unet_moe.data import TASKS
from unet_moe.fair_suite import _digest, create_model, forward_model
from unet_moe.region import PAPER_ALIGNED_TASK_IDS, RegionDataset


ABDOMEN_TASKS = (13, 14, 16, 18, 20)
THRESHOLDS = (.05, .10, .15, .20, .30, .40, .50, .60, .70, .80)
MIN_AREAS = (0, 32, 128, 512)
CALIBRATION_CT = 'AbdomenUS/ct3'
CONFIRMATION_CT = 'AbdomenUS/ct4'
MIN_POSITIVE = 10
MIN_EMPTY = 10
VESSEL_MIN_POSITIVE_CONFIRMATION = 5


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def specifications():
    return [dict(id=f's{index:03d}', threshold=threshold, min_area=area,
                 largest_only=largest)
            for index, (threshold, area, largest) in enumerate(
                (t, a, l) for t in THRESHOLDS for a in MIN_AREAS for l in (False, True))]


def baseline_id(specs):
    return next(spec['id'] for spec in specs if spec['threshold'] == .5
                and spec['min_area'] == 0 and not spec['largest_only'])


def component_counts(probability, target, specs):
    """Apply rules to all predicted pixels, then score only valid target pixels."""
    probability, target = np.asarray(probability), np.asarray(target)
    if probability.shape != target.shape or probability.ndim != 2:
        raise ValueError('Probability and target must be equal 2D arrays')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid probability values')
    if not np.isin(target, (-1, 0, 1)).all():
        raise ValueError('Target must contain only -1, 0, and 1')
    valid = target >= 0
    truth = target > 0
    counts = {}
    for threshold in dict.fromkeys(spec['threshold'] for spec in specs):
        raw = np.ascontiguousarray((probability >= threshold).astype(np.uint8))
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        valid_pixels_by_label = np.bincount(labels[valid], minlength=component_count)
        true_pixels = np.bincount(labels[truth], minlength=component_count)
        foreground_labels = np.arange(1, component_count)
        for spec in (s for s in specs if s['threshold'] == threshold):
            selected = foreground_labels[areas[1:] >= spec['min_area']]
            if spec['largest_only'] and selected.size:
                selected = selected[np.argmax(areas[selected])][None]
            predicted = int(valid_pixels_by_label[selected].sum())
            tp = int(true_pixels[selected].sum())
            counts[spec['id']] = dict(valid_pixels=int(valid.sum()),
                                      truth_pixels=int(truth.sum()),
                                      predicted_pixels=predicted, tp=tp)
    return counts


def summarize(rows):
    rows = [r for r in rows if r['valid_pixels'] > 0]
    if not rows:
        raise ValueError('Cannot summarize zero rows')
    positives = [r for r in rows if r['truth_pixels'] > 0]
    empty = [r for r in rows if r['truth_pixels'] == 0]
    positive_dice = [2 * r['tp'] / (r['truth_pixels'] + r['predicted_pixels'])
                     for r in positives]
    return dict(images=len(rows), positive_images=len(positives), empty_images=len(empty),
                positive_dice=float(np.mean(positive_dice)) if positives else None,
                positive_miss_rate=float(np.mean([r['predicted_pixels'] == 0 for r in positives])) if positives else None,
                positive_zero_overlap_rate=float(np.mean([r['tp'] == 0 for r in positives])) if positives else None,
                empty_false_positive_rate=float(np.mean([r['predicted_pixels'] > 0 for r in empty])) if empty else None,
                empty_mean_false_positive_area_fraction=float(np.mean([
                    r['predicted_pixels'] / r['valid_pixels'] for r in empty])) if empty else None,
                empty_mean_false_positive_pixels=float(np.mean([r['predicted_pixels'] for r in empty])) if empty else None)


def eligible(candidate, baseline, mode, min_positive=MIN_POSITIVE):
    """Predeclared guardrails; evaluate exactly the same rules on CT3 and CT4."""
    for row in (candidate, baseline):
        if row['positive_images'] < min_positive or row['empty_images'] < MIN_EMPTY:
            return False
    positive = candidate['positive_dice']
    positive_base = baseline['positive_dice']
    miss = candidate['positive_miss_rate']
    miss_base = baseline['positive_miss_rate']
    zero = candidate['positive_zero_overlap_rate']
    zero_base = baseline['positive_zero_overlap_rate']
    fp = candidate['empty_false_positive_rate']
    fp_base = baseline['empty_false_positive_rate']
    area = candidate['empty_mean_false_positive_area_fraction']
    area_base = baseline['empty_mean_false_positive_area_fraction']
    eps = 1e-12
    if mode == 'precision':
        safe = (positive >= positive_base - .01 - eps and
                miss <= miss_base + .01 + eps and zero <= zero_base + .01 + eps)
        improved = (area <= area_base - max(.002, .10 * area_base) + eps or
                    fp <= fp_base - .05 + eps)
        return safe and improved
    if mode == 'recall':
        safe = (miss <= miss_base + .01 + eps and zero <= zero_base + .01 + eps and
                fp <= fp_base + .05 + eps and
                area <= area_base + max(.005, .10 * area_base) + eps)
        return safe and positive >= positive_base + .01 - eps
    raise ValueError(f'Unknown screen mode: {mode}')


def choose(calibration, baseline, mode, min_positive=MIN_POSITIVE):
    candidates = [(spec, metric) for spec, metric in calibration
                  if eligible(metric, baseline, mode, min_positive)]
    if not candidates:
        return None
    if mode == 'precision':
        return min(candidates, key=lambda item: (
            item[1]['empty_mean_false_positive_area_fraction'],
            item[1]['empty_false_positive_rate'], -item[1]['positive_dice'],
            abs(item[0]['threshold'] - .5), item[0]['min_area'], item[0]['largest_only']))[0]
    return min(candidates, key=lambda item: (
        -item[1]['positive_dice'], item[1]['positive_zero_overlap_rate'],
        item[1]['empty_mean_false_positive_area_fraction'],
        abs(item[0]['threshold'] - .5), item[0]['min_area'], item[0]['largest_only']))[0]


def analyze(rows, specs):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['ct'], row['structure'], row['spec'])].append(row)
    base_id = baseline_id(specs)
    by_structure = {}
    for name in (TASKS[task] for task in ABDOMEN_TASKS):
        vessel = name == TASKS[16]
        calibration_ct, confirmation_ct = ((CONFIRMATION_CT, CALIBRATION_CT) if vessel
                                           else (CALIBRATION_CT, CONFIRMATION_CT))
        min_positive_confirm = VESSEL_MIN_POSITIVE_CONFIRMATION if vessel else MIN_POSITIVE
        calibration = {spec['id']: summarize(grouped[(calibration_ct, name, spec['id'])]) for spec in specs}
        confirmation = {spec['id']: summarize(grouped[(confirmation_ct, name, spec['id'])]) for spec in specs}
        pooled = {spec['id']: summarize(grouped[(CALIBRATION_CT, name, spec['id'])] +
                                         grouped[(CONFIRMATION_CT, name, spec['id'])]) for spec in specs}
        base_cal, base_confirm = calibration[base_id], confirmation[base_id]
        sufficiently_observed = (base_cal['positive_images'] >= MIN_POSITIVE and
                                 base_confirm['positive_images'] >= min_positive_confirm and
                                 min(base_cal['empty_images'], base_confirm['empty_images']) >= MIN_EMPTY)
        selected = {}
        for mode in ('precision', 'recall'):
            spec = choose([(s, calibration[s['id']]) for s in specs if s['id'] != base_id],
                          base_cal, mode, MIN_POSITIVE)
            selected[mode] = (dict(status='inconclusive_low_counts' if not sufficiently_observed else 'none_selected')
                              if spec is None else dict(
                                  status='passed_limited_ct_check' if eligible(
                                      confirmation[spec['id']], base_confirm, mode, min_positive_confirm)
                                  else 'not_confirmed',
                                  rule=spec, calibration=calibration[spec['id']],
                                  confirmation=confirmation[spec['id']], pooled=pooled[spec['id']]))
        by_structure[name] = dict(calibration_ct=calibration_ct,
                                  confirmation_ct=confirmation_ct,
                                  confirmation_positive_minimum=min_positive_confirm,
                                  baseline=dict(rule=next(s for s in specs if s['id'] == base_id),
                                                calibration=base_cal, confirmation=base_confirm,
                                                pooled=pooled[base_id]),
                                  candidates=selected,
                                  calibration_grid=[dict(rule=spec, metrics=calibration[spec['id']]) for spec in specs])
    return by_structure


def verify_inputs(checkpoint, protocol_path, records_path):
    protocol = json.loads(protocol_path.read_text())
    records = json.loads(records_path.read_text())
    if _digest(records) != protocol['record_sha256']:
        raise ValueError('Record metadata differs from frozen fair protocol')
    from unet_moe import fair_suite
    for source, digest in protocol['source_files'].items():
        if sha256_file(Path(fair_suite.__file__).with_name(source)) != digest:
            raise ValueError(f'Frozen fair source differs: {source}')
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    last_path = checkpoint.with_name('last.pt')
    last = torch.load(last_path, map_location='cpu', weights_only=True)
    if checkpoint.name != 'best.pt' or cp['epoch'] != cp['best_epoch']:
        raise ValueError('Require frozen best.pt selected by validation Dice')
    if last['epoch'] != protocol['config']['epochs'] or last['best_epoch'] != cp['epoch']:
        raise ValueError('Training budget incomplete or best/last checkpoints disagree')
    for key in ('config_sha256', 'record_sha256', 'source_sha256', 'kind', 'seed'):
        if cp[key] != last[key] or (key.endswith('_sha256') and cp[key] != protocol[key]):
            raise ValueError(f'Checkpoint/protocol mismatch: {key}')
    if cp['task_ids'] != list(PAPER_ALIGNED_TASK_IDS) or protocol['config']['size'] != 512:
        raise ValueError('Require frozen paper5 512x512 protocol')
    if protocol['config']['abdomen_split'] != 'ct7_2' or protocol['config']['split_policy'] != 'trainmore_v6':
        raise ValueError('Unexpected abdominal split')
    if cp['kind'] not in ('shared', 'task_id', 'region'):
        raise ValueError('Unsupported fair model')
    val_rows = [r for r in records if r['split'] == 'val' and r['dataset'] == 'AbdomenUS'
                and r['task'] in ABDOMEN_TASKS]
    if set(r['group'] for r in val_rows) != {CALIBRATION_CT, CONFIRMATION_CT}:
        raise ValueError('Unexpected validation CT groups')
    dataset = RegionDataset(val_rows, size=512, task_ids=PAPER_ALIGNED_TASK_IDS)
    for group in dataset.groups:
        if {r['task'] for r in group} != set(ABDOMEN_TASKS) or len({r['group'] for r in group}) != 1:
            raise ValueError('Each abdominal image must have exactly five annotated channels from one CT')
    return protocol, records, cp, dataset


def run(args):
    args.checkpoint = args.checkpoint.resolve()
    args.protocol = args.protocol.resolve()
    args.records = args.records.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    if args.batch_size != 1:
        raise ValueError('This screen requires batch size 1 to keep per-image CT provenance')
    torch.set_num_threads(args.threads)
    protocol, records, cp, dataset = verify_inputs(args.checkpoint, args.protocol, args.records)
    config = protocol['config']
    model = create_model(cp['kind'], config['base'], config['top_k'], config['experts'])
    model.load_state_dict(cp['model'])
    model = model.to(args.device).eval()
    specs = specifications()
    task_channels = [(task, list(PAPER_ALIGNED_TASK_IDS).index(task)) for task in ABDOMEN_TASKS]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{args.output.name}.', dir=args.output.parent))
    started = time.perf_counter()
    try:
        with (temporary / 'per_image_counts.jsonl').open('w') as stream:
            with torch.inference_mode():
                for index, group in enumerate(dataset.groups):
                    image, target = dataset[index]
                    output = forward_model(model, cp['kind'], image[None].to(args.device),
                                           target[None].to(args.device))
                    probabilities = output['logits'].sigmoid()[0].cpu().numpy()
                    ct, path = group[0]['group'], group[0]['image']
                    for task, channel in task_channels:
                        counts = component_counts(probabilities[channel], target[channel].numpy(), specs)
                        for spec in specs:
                            row = dict(ct=ct, image=path, structure=TASKS[task], spec=spec['id'],
                                       **counts[spec['id']])
                            rows.append(row)
                            stream.write(json.dumps(row) + '\n')
                    if (index + 1) % 20 == 0 or index + 1 == len(dataset):
                        print(f'validation images {index + 1}/{len(dataset)}', flush=True)
        result = dict(protocol=dict(
            purpose='Validation-only exploratory postprocessing screen; no test images or masks opened',
            checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
            last_checkpoint_sha256=sha256_file(args.checkpoint.with_name('last.pt')),
            checkpoint_epoch=cp['epoch'], completed_training_epochs=config['epochs'],
            checkpoint_kind=cp['kind'], seed=cp['seed'], input_size=512,
            fair_config_sha256=protocol['config_sha256'], fair_record_sha256=protocol['record_sha256'],
            fair_source_sha256=protocol['source_sha256'], records_file_sha256=sha256_file(args.records),
            usual_calibration_ct=CALIBRATION_CT, usual_confirmation_ct=CONFIRMATION_CT,
            vessel_calibration_ct=CONFIRMATION_CT, vessel_confirmation_ct=CALIBRATION_CT,
            note='Both CT sources were inspected in superseded v1; this second screen is exploratory. The checkpoint was selected using validation data from both CTs. There is no independent validation source for model or postprocessing.',
            independent_units='2 CT source groups; image counts are descriptive, not independent CT replicates',
            candidate_grid=specs, minimum_positive_images_per_ct=MIN_POSITIVE,
            vessel_confirmation_minimum_positive_images=VESSEL_MIN_POSITIVE_CONFIRMATION,
            minimum_empty_images_per_ct=MIN_EMPTY,
            selection='Precision and recall rules selected under fixed safeguards on CT3 for four organs, CT4 for vessels; opposite CT is a limited check. Ignore labels are never used to form components.',
            confirmation_reporting='Only baseline and preselected candidate metrics appear in screen.json; per_image_counts.jsonl is an audit trace and must not be used to pick a replacement rule.',
            device=args.device, batch_size=1, torch_threads=args.threads,
            code_sha256=sha256_file(Path(__file__))),
            validation_images=len(dataset), elapsed_seconds=time.perf_counter() - started,
            per_structure=analyze(rows, specs))
        (temporary / 'screen.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        os.rename(temporary, args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(dict(output=str(args.output), validation_images=result['validation_images'],
                          elapsed_seconds=result['elapsed_seconds'],
                          selected={name: {mode: row['status'] for mode, row in item['candidates'].items()}
                                    for name, item in result['per_structure'].items()})), flush=True)


if __name__ == '__main__':
    main()
