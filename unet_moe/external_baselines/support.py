"""Train-only observed-ROI support, with deterministic query identity exclusion."""
import hashlib
import json
import random
from collections import Counter

import numpy as np
import torch

from ..region import DOMAINS
from ..single_channel_suite import UnionDataset
from .valid_region import crop_resize, valid_roi


def subject_key(row):
    # Existing group metadata: CAMUS patient, CCA volume, HC subject, abdomen
    # CT source; in image-only datasets we cannot infer undocumented patients.
    return row['dataset'], row['group']


def _label_hash(target):
    return hashlib.sha256(target.detach().cpu().numpy().astype(np.int8).tobytes()).hexdigest()


def _summary(values):
    if not values:
        return None
    return dict(min=float(min(values)), mean=float(np.mean(values)), median=float(np.median(values)),
                p05=float(np.quantile(values, .05)), p25=float(np.quantile(values, .25)),
                p75=float(np.quantile(values, .75)), p95=float(np.quantile(values, .95)), max=float(max(values)))


def validate_query_support(records, pools, support_size):
    """Metadata-only validation for every unique train/val/test query image."""
    minima = {domain: {} for domain in pools}
    checked = Counter()
    seen = set()
    for row in records:
        key = row['dataset'], row['image']
        if key in seen:
            continue
        seen.add(key)
        eligible = [s for s in pools[row['dataset']]
                    if s['image'] != row['image'] and s['group'] != row['group']]
        if len(eligible) < support_size:
            raise ValueError(f'Insufficient train supports outside query group: '
                             f'{subject_key(row)}, {len(eligible)} < {support_size}')
        split = row['split']
        checked[split] += 1
        previous = minima[row['dataset']].get(split, len(eligible))
        minima[row['dataset']][split] = min(previous, len(eligible))
    return dict(unique_queries_by_split=dict(checked), minimum_eligible_images=minima,
                inspected_query_pixels=False)


def support_manifest(records, *, size=512, support_size=8, min_side=32,
                     minimum_foreground_fraction=0.25):
    if support_size < 1:
        raise ValueError('support_size must be positive')
    train = UnionDataset([row for row in records if row['split'] == 'train'], size)
    pools = {name: [] for name in DOMAINS}
    stats = {name: dict(train_images=0, full_image=0, cropped=0, excluded={}) for name in DOMAINS}
    exclusions = []
    for index, rows in enumerate(train.groups):
        _, target, domain, _ = train[index]
        stat = stats[domain]
        stat['train_images'] += 1
        full_fg = int(target.eq(1).sum())
        full_bg = int(target.eq(0).sum())
        roi = valid_roi(target[0], min_side=min_side,
                        minimum_foreground_fraction=minimum_foreground_fraction) if full_fg and full_bg else None
        if not full_fg:
            reason = 'no_known_foreground'
        elif not full_bg:
            reason = 'no_known_background'
        elif roi is None:
            reason = 'no_complete_mixed_roi_meeting_size_and_foreground'
        else:
            reason = None
        if reason:
            stat['excluded'][reason] = stat['excluded'].get(reason, 0) + 1
            exclusions.append(dict(index=index, dataset=domain, image=rows[0]['image'], reason=reason))
            continue
        top, left, bottom, right = roi
        selected = target[:, top:bottom, left:right]
        foreground = int(selected.eq(1).sum())
        area = (bottom - top) * (right - left)
        full = roi == (0, 0, size, size)
        stat['full_image' if full else 'cropped'] += 1
        pools[domain].append(dict(
            index=index, image=rows[0]['image'], group=rows[0]['group'], roi=list(roi),
            full_image=full, roi_area=area, retained_image_fraction=area / (size * size),
            source_known_pixels=int(target.ge(0).sum()), source_foreground_pixels=full_fg,
            roi_foreground_pixels=foreground, roi_background_pixels=area-foreground,
            retained_foreground_fraction=foreground / full_fg,
            roi_foreground_fraction=foreground / area, roi_target_sha256=_label_hash(selected),
        ))
    for domain, entries in pools.items():
        stats[domain]['support_images'] = len(entries)
        stats[domain]['support_groups'] = len({entry['group'] for entry in entries})
        for key in ['retained_image_fraction', 'retained_foreground_fraction', 'roi_foreground_fraction']:
            stats[domain][key] = _summary([entry[key] for entry in entries])
        stats[domain]['cropped_retained_foreground_fraction'] = _summary(
            [entry['retained_foreground_fraction'] for entry in entries if not entry['full_image']])
    verification = validate_query_support(records, pools, support_size)
    return dict(
        schema='q04_support_v2_observed_roi', size=size, support_size=support_size,
        minimum_roi_side=min_side, selection_seed=1729, pools=pools,
        minimum_retained_foreground_fraction=minimum_foreground_fraction,
        domain_statistics=stats, excluded_images=exclusions, query_verification=verification,
        roi_policy='largest-area fully observed rectangle containing foreground and background '
                   'and meeting minimum_retained_foreground_fraction and minimum_roi_side; '
                   'tie: more foreground, then smallest (top,left,bottom,right); complete images unchanged',
        support_label_policy='only known 0/1; positive-and-background supports; empty query images remain in training',
        resize='ROI on 512 union grid; paired bilinear image / nearest label to 512; query is never cropped',
        identity='same domain; exclude exact query image and every image with the query group; sample images without replacement',
        limitation='Unknown patient IDs in image-only datasets cannot be excluded. Supports may share a group with '
                   'each other but never with query. ROI supports alter field of view and may retain only part of union foreground.',
        training_selection='stateless seed + epoch + draw_position + query identity',
        evaluation_selection='fixed query identity; same supports across seeds/checkpoints',
    )


class SupportBank:
    def __init__(self, records, manifest, size=512):
        self.dataset = UnionDataset([r for r in records if r['split'] == 'train'], size)
        self.manifest = manifest
        self.size = size
        self.entries = {}
        if manifest.get('schema') != 'q04_support_v2_observed_roi' or manifest.get('size') != size:
            raise ValueError('Expected a matching frozen observed-ROI support manifest')
        if not 0 <= manifest.get('minimum_retained_foreground_fraction', -1) <= 1:
            raise ValueError('Support manifest must freeze its minimum retained foreground fraction')
        for domain, entries in manifest['pools'].items():
            for entry in entries:
                index = entry['index']
                if index in self.entries or index < 0 or index >= len(self.dataset):
                    raise ValueError('Duplicate or invalid support index')
                row = self.dataset.groups[index][0]
                if row['dataset'] != domain or any(row[k] != entry[k] for k in ('image', 'group')):
                    raise ValueError('Support manifest no longer matches frozen train records')
                self.entries[index] = entry
        validate_query_support(records, manifest['pools'], manifest['support_size'])

    def indices(self, query, *, training_seed=None, epoch=0, draw_position=0):
        eligible = [s for s in self.manifest['pools'][query['dataset']]
                    if s['image'] != query['image'] and s['group'] != query['group']]
        if len(eligible) < self.manifest['support_size']:
            raise ValueError('Insufficient supports outside query group; replacement is forbidden')
        identity = [self.manifest['selection_seed'], query['dataset'], query['image']]
        if training_seed is not None:
            identity += [training_seed, epoch, draw_position]
        digest = hashlib.sha256(json.dumps(identity).encode()).digest()
        rng = random.Random(int.from_bytes(digest, 'big'))
        return [entry['index'] for entry in rng.sample(eligible, self.manifest['support_size'])]

    def item(self, index):
        entry = self.entries[index]
        image, target, _, _ = self.dataset[index]
        top, left, bottom, right = entry['roi']
        if _label_hash(target[:, top:bottom, left:right]) != entry['roi_target_sha256']:
            raise ValueError('Frozen support ROI labels changed')
        if int(target.eq(1).sum()) != entry['source_foreground_pixels']:
            raise ValueError('Frozen support foreground outside the ROI changed')
        if (entry['retained_foreground_fraction'] < self.manifest['minimum_retained_foreground_fraction']
                or min(bottom-top, right-left) < self.manifest['minimum_roi_side']):
            raise ValueError('Frozen support ROI violates its declared size/foreground constraints')
        image, target = crop_resize(image, target, entry['roi'], size=self.size)
        if not target.eq(0).any() or not target.eq(1).any():
            raise ValueError('Frozen support no longer contains both foreground/background after resizing')
        return image, target

    def batch(self, queries, *, device, training_seed=None, epoch=0, draw_positions=None):
        images, labels, chosen = [], [], []
        positions = list(range(len(queries))) if draw_positions is None else draw_positions
        for query, position in zip(queries, positions, strict=True):
            indices = self.indices(query, training_seed=training_seed, epoch=epoch, draw_position=position)
            items = [self.item(index) for index in indices]
            images.append(torch.stack([x[0] for x in items]))
            labels.append(torch.stack([x[1] for x in items]))
            chosen.append(indices)
        return torch.stack(images).to(device), torch.stack(labels).to(device), chosen
