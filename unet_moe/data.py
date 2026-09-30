import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


TASKS = ['breast_lesion', 'ovarian_lesion', 'thyroid_nodule', 'liver', 'liver_mass',
         'ultrasound_outline', 'aponeurosis', 'fascicle', 'head_contour', 'carotid',
         'lv_cavity', 'myocardium', 'left_atrium', 'abdomen_liver', 'abdomen_kidney',
         'abdomen_pancreas', 'abdomen_vessels', 'abdomen_adrenal', 'abdomen_gallbladder',
         'abdomen_bone', 'abdomen_spleen']

SPLIT_VERSION = 'source_aware_v4_no_fascicle'
ABDOMEN_7_2_SPLIT_VERSION = 'source_aware_v5_abdomen_ct7_2_release_test'
ABDOMEN_RELEASE_TRAIN_CTS = frozenset({'ct2', 'ct3', 'ct4', 'ct5', 'ct7', 'ct8', 'ct9', 'ct14', 'ct15'})
ABDOMEN_VALIDATION_CTS = frozenset({'ct3', 'ct4'})
# Five source-level validation folds. Every original training CT is validated
# at least once; ct15 is repeated to keep each fold at two CT sources.
ABDOMEN_CV5_VALIDATION_CTS = (
    frozenset({'ct2', 'ct15'}),
    frozenset({'ct3', 'ct7'}),
    frozenset({'ct4', 'ct14'}),
    frozenset({'ct5', 'ct8'}),
    frozenset({'ct9', 'ct15'}),
)
TRAINMORE_SPLIT_VERSION = 'source_aware_v6_frozen_test_more_train'


def camus_splits(dataset_root):
    folder = dataset_root / 'data/CAMUS_public/CAMUS_public/database_split'
    result = {}
    for filename, split in [('subgroup_training.txt', 'train'),
                            ('subgroup_validation.txt', 'val'), ('subgroup_testing.txt', 'test')]:
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f'Official CAMUS split required: {path}')
        for patient in path.read_text(encoding='utf-8-sig').split():
            if patient in result:
                raise ValueError(f'Duplicate CAMUS patient in split lists: {patient}')
            result[patient] = split
    return result


def choose_split(row, group, seed, official=None, abdomen_split='legacy',
                 abdomen_fold=None, split_policy='legacy'):
    value = int(hashlib.sha256(f'{seed}/{group}'.encode()).hexdigest()[:8], 16) / 2**32
    declared = row.get('split', '').lower()
    if row['dataset'] == 'CAMUS':
        patient = Path(row['image']).parent.name
        if patient not in official:
            raise ValueError(f'CAMUS patient absent from official lists: {patient}')
        return official[patient], 'official_camus_patient_lists'
    if row['dataset'] == 'AbdomenUS' and abdomen_split in ('ct7_2', 'ct_cv5', 'ct_final'):
        if declared == 'test':
            return 'test', 'release_manifest_partition'
        ct = group.split('/')[-1]
        if declared != 'train' or ct not in ABDOMEN_RELEASE_TRAIN_CTS:
            raise ValueError(f'Unexpected AbdomenUS release training CT: {ct} ({declared})')
        if abdomen_split == 'ct_cv5':
            validation_cts = ABDOMEN_CV5_VALIDATION_CTS[abdomen_fold]
            return ('val' if ct in validation_cts else 'train'), f'project_abdomen_ct_cv5_fold{abdomen_fold}'
        if abdomen_split == 'ct_final':
            return 'train', 'project_abdomen_all9_final_fit'
        return ('val' if ct in ABDOMEN_VALIDATION_CTS else 'train'), 'project_abdomen_ct7_2_release_test'
    if declared in ('test', 'val', 'validation'):
        return ('val' if declared == 'validation' else declared), 'release_manifest_partition'
    if row['dataset'] == 'Fetal_HC':
        val_end = .25 if split_policy == 'trainmore_v6' else .30
        protocol = 'project_hc_labeled_70_15_15' if split_policy == 'legacy' else 'project_hc_labeled_75_10_15'
        return ('test' if value < .15 else 'val' if value < val_end else 'train'), protocol
    if declared == 'train':
        val_fraction = .1 if split_policy == 'trainmore_v6' else .2
        return ('val' if value < val_fraction else 'train'), f'release_train_project_validation_{int(val_fraction*100)}pct'
    if row['dataset'] == 'FALLMUD':
        # Source name is part of group. Expected fractions, not exact author sample lists.
        val_end = .25 if split_policy == 'trainmore_v6' else .32
        protocol = 'project_fallmud_68_17_15' if split_policy == 'legacy' else 'project_fallmud_75_10_15'
        return ('test' if value < .15 else 'val' if value < val_end else 'train'), protocol
    val_end = .25 if split_policy == 'trainmore_v6' else .30
    protocol = 'project_70_15_15' if split_policy == 'legacy' else 'project_75_10_15'
    return ('test' if value < .15 else 'val' if value < val_end else 'train'), protocol


def validate_splits(records):
    assigned = {}
    for record in records:
        for key in [('group', record['group']), ('image', record['image'])]:
            previous = assigned.setdefault(key, record['split'])
            if previous != record['split']:
                raise ValueError(f'Split leakage: {key} occurs in {previous} and {record["split"]}')


def task_names(row):
    dataset = row['dataset']
    if dataset == 'CAMUS':
        return [(10, 1), (11, 2), (12, 3)]
    if dataset == 'AbdomenUS':
        return [(12 + index, index) for index in range(1, 9)]
    mapping = {'BUSI': 0, 'MMOTU': 1, 'DDTI': 2, 'Fetal_HC': 8, 'CCA': 9}
    if dataset == 'AUL':
        return [({'liver': 3, 'mass': 4, 'outline': 5}[row['label']], 0)]
    if dataset == 'FALLMUD':
        return [(6, 0)] if row['label'] == 'aponeurosis' else []
    return [(mapping[dataset], 0)] if dataset in mapping else []


def build_records(root, seed=42, abdomen_split='legacy', abdomen_fold=None,
                  split_policy='legacy'):
    if abdomen_split not in ('legacy', 'ct7_2', 'ct_cv5', 'ct_final'):
        raise ValueError(f'Unsupported AbdomenUS split: {abdomen_split}')
    if split_policy not in ('legacy', 'trainmore_v6'):
        raise ValueError(f'Unsupported split policy: {split_policy}')
    if abdomen_split == 'ct_cv5':
        if abdomen_fold is None or not 0 <= abdomen_fold < len(ABDOMEN_CV5_VALIDATION_CTS):
            raise ValueError('ct_cv5 requires --abdomen-fold in 0..4')
    elif abdomen_fold is not None:
        raise ValueError('abdomen_fold is only valid with ct_cv5')
    split_version = (TRAINMORE_SPLIT_VERSION if split_policy == 'trainmore_v6' else
                     SPLIT_VERSION if abdomen_split == 'legacy' else ABDOMEN_7_2_SPLIT_VERSION)
    if abdomen_split in ('ct_cv5', 'ct_final'):
        split_version += f'_{abdomen_split}' + (f'_fold{abdomen_fold}' if abdomen_fold is not None else '')
    records, skipped = [], {}
    for manifest in sorted(Path(root).glob('*/processed_png/manifest.csv')):
        filled_manifest = manifest.with_name('manifest_filled.csv')
        if 'HC18' in manifest.parent.parent.name and filled_manifest.exists():
            manifest = filled_manifest
        if 'AbdomenUS' in manifest.parent.parent.name:
            manifest = manifest.with_name('manifest_indexed.csv')
            if not manifest.exists():
                raise FileNotFoundError('Run scripts/convert_abdomen_masks.py before training')
        official = None
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['dataset'] == 'FALLMUD' and row['label'] == 'fascicle':
                    skipped['FALLMUD:disabled_fascicle'] = skipped.get('FALLMUD:disabled_fascicle', 0) + 1
                    continue
                if row['pairing_status'] != 'paired' or not task_names(row):
                    reason = row['dataset'] + ':' + ('unsupported_labels' if row['dataset'] == 'AbdomenUS' else row['pairing_status'])
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                # Avoid duplicate ED/ES samples and temporally correlated dense sequence labels.
                if row['dataset'] == 'CAMUS' and 'half_sequence' in row['image']:
                    skipped['CAMUS:sequence_frames'] = skipped.get('CAMUS:sequence_frames', 0) + 1
                    continue
                if row['dataset'] == 'CAMUS' and official is None:
                    official = camus_splits(manifest.parent.parent)
                group = row['dataset'] + '/' + row['image']
                if row['dataset'] == 'CAMUS':
                    group = 'CAMUS/' + Path(row['image']).parent.name
                elif row['dataset'] == 'CCA':
                    group = 'CCA/' + Path(row['image']).stem.split('_slice_')[0]
                elif row['dataset'] == 'Fetal_HC':
                    group = 'Fetal_HC/' + Path(row['image']).stem.split('_')[0]
                elif row['dataset'] == 'AbdomenUS':
                    group = 'AbdomenUS/' + row['sample_id'].split('-')[0]
                split, protocol = choose_split(row, group, seed, official, abdomen_split,
                                               abdomen_fold, split_policy)
                for task, class_id in task_names(row):
                    records.append(dict(image=str(manifest.parent / row['image']),
                                        mask=str(manifest.parent / row['mask']), task=task,
                                        class_id=class_id, split=split, group=group,
                                        dataset=row['dataset'], split_protocol=protocol,
                                        label_version=('hc18_filled_v1' if row['dataset'] == 'Fetal_HC' and manifest.name == 'manifest_filled.csv' else 'original'),
                                        split_version=split_version))
    if not records:
        raise ValueError(f'No supported paired samples found under {root}')
    validate_splits(records)
    return records, skipped


class SegmentationDataset(Dataset):
    def __init__(self, records, size=256):
        self.records, self.size = records, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record['image']) as image:
            original_size = image.size
            x = np.array(image.convert('L').resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        with Image.open(record['mask']) as mask:
            if mask.size != original_size:
                raise ValueError(f"Image/mask size mismatch: {record['mask']}")
            # Indexed CAMUS PNG must be read before any grayscale conversion.
            if record['class_id']:
                indices = np.array(mask)
                abdomen = record.get('dataset') == 'AbdomenUS'
                allowed = set(range(9)) | {255} if abdomen else {0, 1, 2, 3}
                if indices.ndim != 2 or not set(np.unique(indices)).issubset(allowed):
                    raise ValueError('Unexpected class indices')
                y = (indices == record['class_id']).astype(np.uint8) * 255
                if abdomen:
                    y[indices == 255] = 128
            else:
                y = (np.array(mask.convert('L')) >= 128).astype(np.uint8) * 255
            y = np.array(Image.fromarray(y).resize((self.size, self.size), Image.Resampling.NEAREST), dtype=np.float32) / 255
            if record.get('dataset') == 'AbdomenUS':
                y[(y > 0) & (y < 1)] = -1
        return torch.from_numpy(x[None]), torch.from_numpy(y[None]), torch.tensor(record['task'])
