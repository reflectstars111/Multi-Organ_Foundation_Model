"""Read-only abdomen audit of a frozen RegionUNet checkpoint, without calibration."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from unet_moe.diagnostics import AREA_EDGES, binary_diagnostics, distribution, summarize_binary
from unet_moe.region import RegionUNet


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit directory')
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    torch.set_num_threads(4)
    frozen = json.loads(args.records.read_text())
    records = frozen['records']
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if cp.get('architecture') != 'region_unet_v1':
        raise ValueError('Expected region_unet_v1 checkpoint')
    model = RegionUNet(**cp['config']).to(args.device)
    model.load_state_dict(cp['model']); model.eval()
    size = cp['size']
    class_channels = [(task - 12, i, model.structures[i]) for i, task in
                      enumerate(model.task_ids) if 13 <= task <= 20]
    groups = defaultdict(list)
    for row in records:
        if row['dataset'] == 'AbdomenUS':
            groups[row['image']].append(row)
    images = list(groups.values())
    protocol = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint),
                    checkpoint_epoch=cp['epoch'], checkpoint_selection='provided_frozen_checkpoint',
                    records=str(args.records), records_file_sha256=file_hash(args.records),
                    records_semantic_sha256=frozen.get('sha256'), size=size, threshold=.5,
                    area_bin_edges=list(AREA_EDGES), batch_size=args.batch_size,
                    purpose='Diagnosis only; no training, threshold search, area filtering, or calibration.',
                    analysis='Same fixed metrics on all available abdominal splits and CT groups.',
                    test_reuse_warning='Test findings are exploratory evidence; new changes require train/validation selection and fresh confirmatory evaluation.',
                    routing='Top-1 expert and mean probability of expert 8, by present/empty organ.',
                    missing_channels='Outputs without annotation are counted separately and are not false positives.',
                    split_ct_counts=dict(Counter(f"{r[0]['split']}:{r[0]['group']}" for r in images)))
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    measurements, mask_rows, missing_rows, routing = [], [], [], defaultdict(list)
    with (args.output/'image_metrics.jsonl').open('w') as detail:
        for start in range(0, len(images), args.batch_size):
            batch = images[start:start + args.batch_size]
            tensors, labels = [], []
            for rows in batch:
                row = rows[0]
                if len({r['mask'] for r in rows}) != 1:
                    raise ValueError('Abdomen image must have exactly one indexed mask')
                if set(r['task'] for r in rows) != {task for task in model.task_ids if 13 <= task <= 20}:
                    raise ValueError('Missing or extra abdominal supervision channel')
                with Image.open(row['image']) as im:
                    native_size = im.size
                    gray = np.array(im.convert('L').resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
                with Image.open(row['mask']) as im:
                    if im.size != native_size:
                        raise ValueError('Image/mask dimensions differ')
                    native = np.asarray(im)
                    if native.ndim != 2 or not set(np.unique(native)).issubset(set(range(9)) | {255}):
                        raise ValueError('Invalid indexed labels')
                    indices = np.asarray(im.resize((size, size), Image.Resampling.NEAREST))
                tensors.append(torch.from_numpy(gray[None])); labels.append(indices)
                for class_id, _, name in class_channels:
                    mask_rows.append(dict(image=row['image'], split=row['split'], group=row['group'], structure=name,
                                          native_width=native_size[0], native_height=native_size[1],
                                          native_pixels=int((native == class_id).sum()),
                                          resized_pixels=int((indices == class_id).sum()),
                                          native_valid_pixels=int((native != 255).sum()),
                                          resized_valid_pixels=int((indices != 255).sum()),
                                          native_ignore_fraction=float((native == 255).mean()),
                                          resized_ignore_fraction=float((indices == 255).mean())))
            with torch.inference_mode():
                output = model(torch.stack(tensors).to(args.device))
                probabilities = output['logits'].sigmoid().cpu().numpy()
                routes = [x.softmax(-1).cpu().numpy() for x in output['routing']]
            for b, rows in enumerate(batch):
                row, indices = rows[0], labels[b]
                for class_id, channel, name in class_channels:
                    target = (indices == class_id).astype(np.int8)
                    target[indices == 255] = -1
                    m = binary_diagnostics(probabilities[b, channel], target)
                    m.update(image=row['image'], group=row['group'], split=row['split'], structure=name)
                    pred = (probabilities[b, channel] >= .5) & (indices != 255)
                    m['prediction_by_label_pixels'] = {str(c): int((pred & (indices == c)).sum()) for c in range(9)}
                    measurements.append(m); detail.write(json.dumps(m) + '\n')
                    status = 'positive' if m['truth_pixels'] else 'empty'
                    for level, route in enumerate(routes):
                        routing[(row['split'], name, status, level)].append(
                            (int(route[b, channel].argmax()), float(route[b, channel, 8])))
                unannotated = [i for i, task in enumerate(model.task_ids) if task < 13]
                for channel in unannotated:
                    p = probabilities[b, channel]
                    missing_rows.append(dict(split=row['split'], structure=model.structures[channel],
                                             predicted_pixels=int((p >= .5).sum()), image=row['image']))
            print(f'audit images {min(start + args.batch_size, len(images))}/{len(images)}', flush=True)
    splits = sorted({r['split'] for r in measurements})
    summary = dict(protocol=protocol, per_split={}, mask_audit={}, routing={}, unannotated_outputs={})
    for split in splits:
        local = [r for r in measurements if r['split'] == split]
        names = [name for _, _, name in class_channels]
        summary['per_split'][split] = dict(
            per_structure={name: summarize_binary([r for r in local if r['structure'] == name]) for name in names},
            per_ct={ct: {name: summarize_binary([r for r in local if r['structure'] == name and r['group'] == ct])
                         for name in names} for ct in sorted({r['group'] for r in local})})
        mask_local = [r for r in mask_rows if r['split'] == split]
        summary['mask_audit'][split] = dict(
            image_dimensions=dict(Counter(f"{r['native_width']}x{r['native_height']}" for r in mask_local[::len(names)])),
            native_ignore_fraction=distribution([r['native_ignore_fraction'] for r in mask_local[::len(names)]]),
            resized_ignore_fraction=distribution([r['resized_ignore_fraction'] for r in mask_local[::len(names)]]),
            per_structure={name: dict(
                native_positive_images=sum(r['native_pixels'] > 0 for r in mask_local if r['structure'] == name),
                resized_positive_images=sum(r['resized_pixels'] > 0 for r in mask_local if r['structure'] == name),
                disappeared_positive_images=sum(r['native_pixels'] > 0 and r['resized_pixels'] == 0
                                                for r in mask_local if r['structure'] == name),
                positive_native_area_fraction=distribution([r['native_pixels'] / r['native_valid_pixels']
                    for r in mask_local if r['structure'] == name and r['native_pixels'] and r['native_valid_pixels']]))
                for name in names})
        summary['routing'][split] = {name: {status: {str(level): dict(
            samples=len(routing[(split, name, status, level)]),
            expert_histogram=dict(Counter(x[0] for x in routing[(split, name, status, level)])),
            abdomen_expert_probability=distribution([x[1] for x in routing[(split, name, status, level)]]))
            for level in range(len(model.moes))} for status in ('positive', 'empty')} for name in names}
        summary['unannotated_outputs'][split] = {name: dict(
            note='Unannotated on AbdomenUS; these predictions cannot be called false positives.',
            predicted_pixels=distribution([r['predicted_pixels'] for r in missing_rows
                                           if r['split'] == split and r['structure'] == name]))
            for name in sorted({r['structure'] for r in missing_rows})}
    (args.output/'mask_metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in mask_rows))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(output=str(args.output), images=len(images), threshold=.5, no_tuning=True)), flush=True)


if __name__ == '__main__':
    main()
