"""Read-only organ diagnostics for the Q01 paired loss experiment.

The absolute guardrails are a frozen experimental selection rule, not clinical
acceptability thresholds. The test split has already been used in historical
analyses and must not be described as an unseen external validation cohort.
"""
from contextlib import contextmanager
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from .data import TASKS
from .fair_suite import forward_model
from .region import PAPER_ALIGNED_TASK_IDS


ABDOMEN_TASK_IDS = (13, 14, 16, 18, 20)
ABDOMEN_STRUCTURES = tuple(TASKS[task] for task in ABDOMEN_TASK_IDS)
GUARDRAIL_CONFIG = {
    'minimum_positive_dice': .10,
    'minimum_vessels_positive_dice': .20,
    'minimum_abdomen_macro_positive_dice': .40,
    'maximum_positive_zero_overlap_rate': .50,
    'maximum_empty_false_positive_area_fraction': .02,
    'require_all_five_organs': True,
}


@contextmanager
def _read_only_model(model):
    """Restore RNG, buffers and individual mode flags, including on failure.

    autograd.grad (rather than backward) preserves pre-existing parameter .grad
    objects. CUDA state is read only if CUDA was already initialized; a CPU
    diagnostic must not initialize or reserve memory on an unrelated GPU.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        model.eval()
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                if not torch.equal(buffer, saved):
                    buffer.copy_(saved)
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _empty_totals():
    return dict(samples=0, positive_samples=0, empty_samples=0,
                dice=0., positive_dice=0., empty_fp=0, positive_miss=0,
                positive_zero_overlap=0, fp_pixels=0., fp_fraction=0.,
                empty_fp_pixels=0., empty_fp_fraction=0.,
                positive_pred_pixels=0., positive_pred_fraction=0.)


def _accumulate(row, values):
    dice, pred_pixels, truth_pixels, intersection, fp_pixels, valid_pixels = values
    row['samples'] += 1
    row['dice'] += dice
    row['fp_pixels'] += fp_pixels
    row['fp_fraction'] += fp_pixels / valid_pixels
    if truth_pixels:
        row['positive_samples'] += 1
        row['positive_dice'] += dice
        row['positive_miss'] += int(pred_pixels == 0)
        row['positive_zero_overlap'] += int(intersection == 0)
        row['positive_pred_pixels'] += pred_pixels
        row['positive_pred_fraction'] += pred_pixels / valid_pixels
    else:
        row['empty_samples'] += 1
        row['empty_fp'] += int(pred_pixels > 0)
        row['empty_fp_pixels'] += fp_pixels
        row['empty_fp_fraction'] += fp_pixels / valid_pixels


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _finalize(totals):
    per_structure = {}
    for name, row in totals.items():
        samples, positive, empty = row['samples'], row['positive_samples'], row['empty_samples']
        per_structure[name] = dict(
            samples=samples, positive_samples=positive, empty_samples=empty,
            dice=row['dice'] / samples if samples else None,
            positive_dice=row['positive_dice'] / positive if positive else None,
            empty_false_positive_rate=row['empty_fp'] / empty if empty else None,
            positive_miss_rate=row['positive_miss'] / positive if positive else None,
            positive_zero_overlap_rate=row['positive_zero_overlap'] / positive if positive else None,
            empty_mean_false_positive_pixels=row['empty_fp_pixels'] / empty if empty else None,
            empty_mean_false_positive_area_fraction=row['empty_fp_fraction'] / empty if empty else None,
            mean_false_positive_pixels=row['fp_pixels'] / samples if samples else None,
            mean_false_positive_area_fraction=row['fp_fraction'] / samples if samples else None,
            positive_mean_predicted_pixels=row['positive_pred_pixels'] / positive if positive else None,
            positive_mean_predicted_area_fraction=row['positive_pred_fraction'] / positive if positive else None)
    result = {'per_structure': per_structure, 'threshold': .5}
    for key, field in [
        ('macro_positive_dice', 'positive_dice'), ('macro_all_dice', 'dice'),
        ('macro_empty_false_positive_rate', 'empty_false_positive_rate'),
        ('macro_positive_miss_rate', 'positive_miss_rate'),
        ('macro_positive_zero_overlap_rate', 'positive_zero_overlap_rate'),
        ('macro_empty_false_positive_pixels', 'empty_mean_false_positive_pixels'),
        ('macro_empty_false_positive_area_fraction', 'empty_mean_false_positive_area_fraction'),
        ('macro_false_positive_pixels', 'mean_false_positive_pixels'),
        ('macro_false_positive_area_fraction', 'mean_false_positive_area_fraction'),
    ]:
        result[key] = _mean(row[field] for row in per_structure.values())
    result['abdomen_macro_positive_dice'] = _mean(
        per_structure[name]['positive_dice'] for name in ABDOMEN_STRUCTURES if name in per_structure)
    return result


def _source(group):
    """RegionDataset.groups contains records for one image, from one source."""
    if not group:
        raise ValueError('Empty groups metadata')
    records = [group] if isinstance(group, dict) else group
    abdominal = [row for row in records if row.get('dataset') == 'AbdomenUS']
    if not abdominal:
        return None
    sources = {str(row.get('group', '')).rsplit('/', 1)[-1] for row in abdominal}
    if len(sources) != 1 or '' in sources or len(abdominal) != len(records):
        raise ValueError('groups metadata must identify exactly one abdomen source per image')
    return next(iter(sources))


@torch.no_grad()
def evaluate(model, kind, loader, device, groups=None, task_ids=None):
    """Compute fair metrics and abdomen source strata in a single forward pass.

    groups must follow the sequential, complete loader order. A randomized
    DataLoader is rejected rather than silently attributing predictions to the
    wrong CT. Missing positive/empty denominators are represented by None.
    """
    task_ids = list(PAPER_ALIGNED_TASK_IDS if task_ids is None else task_ids)
    if groups is not None and isinstance(loader, DataLoader):
        if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
            raise ValueError('groups require a sequential complete DataLoader')
        if len(groups) != len(loader.dataset):
            raise ValueError('groups length must match loader dataset')
    totals = {TASKS[task]: _empty_totals() for task in task_ids}
    source_totals = {}
    offset = 0
    with _read_only_model(model):
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            if target.shape[1] != len(task_ids):
                raise ValueError('Target channels do not match task_ids')
            valid, truth = target.ge(0), target.gt(0)
            logits = forward_model(model, kind, image, target, task_ids)['logits']
            pred = (logits.sigmoid() >= .5) & valid
            pred_pixels, truth_pixels = pred.sum((2, 3)), truth.sum((2, 3))
            intersection = (pred & truth).sum((2, 3))
            denominator = pred_pixels + truth_pixels
            dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.)
            values = torch.stack([dice, pred_pixels, truth_pixels, intersection,
                                  (pred & ~truth).sum((2, 3)), valid.sum((2, 3))], -1).cpu().tolist()
            sources = [None] * len(image)
            if groups is not None:
                if offset + len(image) > len(groups):
                    raise ValueError('groups shorter than loader image count')
                sources = [_source(groups[offset + index]) for index in range(len(image))]
                for source in sources:
                    if source is not None:
                        source_totals.setdefault(source, {name: _empty_totals() for name in ABDOMEN_STRUCTURES})
            for batch, channel in valid.any((2, 3)).cpu().nonzero().tolist():
                name = TASKS[task_ids[channel]]
                _accumulate(totals[name], values[batch][channel])
                if groups is not None and name in ABDOMEN_STRUCTURES and sources[batch] is None:
                    raise ValueError('groups must identify the CT source of every abdomen target')
                if name in ABDOMEN_STRUCTURES and sources[batch] is not None:
                    _accumulate(source_totals[sources[batch]][name], values[batch][channel])
            offset += len(image)
    if groups is not None and offset != len(groups):
        raise ValueError('groups longer than loader image count')
    result = _finalize(totals)
    result['abdomen_by_source'] = {source: _finalize(rows) for source, rows in sorted(source_totals.items())}
    return result


def guardrail_passes(metrics, config=None):
    """Apply the same absolute validation guardrails to A, B and C.

    Every abdomen organ must have applicable positive and empty-label metrics.
    Missing values, NaN and infinity fail instead of being silently skipped.
    """
    limits = dict(GUARDRAIL_CONFIG)
    if config is not None:
        unknown = set(config) - set(limits)
        if unknown:
            raise ValueError(f'Unknown guardrail keys: {sorted(unknown)}')
        limits.update(config)
    if limits['require_all_five_organs'] is not True:
        raise ValueError('Q01 guardrails require all five abdomen organs')

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    if not all(finite(value) and 0 <= value <= 1 for key, value in limits.items()
               if key != 'require_all_five_organs'):
        raise ValueError('Guardrail thresholds must be finite fractions in [0, 1]')
    macro = metrics.get('abdomen_macro_positive_dice')
    if not finite(macro) or macro < limits['minimum_abdomen_macro_positive_dice']:
        return False
    for name in ABDOMEN_STRUCTURES:
        row = metrics.get('per_structure', {}).get(name, {})
        dice = row.get('positive_dice')
        zero_overlap = row.get('positive_zero_overlap_rate')
        empty_area = row.get('empty_mean_false_positive_area_fraction')
        if not all(finite(value) for value in (dice, zero_overlap, empty_area)):
            return False
        minimum = limits['minimum_vessels_positive_dice'] if name == 'abdomen_vessels' else limits['minimum_positive_dice']
        if (dice < minimum or zero_overlap > limits['maximum_positive_zero_overlap_rate']
                or empty_area > limits['maximum_empty_false_positive_area_fraction']):
            return False
    return True


def _gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared = [gradient.detach().double().square().sum() for gradient in gradients if gradient is not None]
    return math.sqrt(float(torch.stack(squared).sum())) if squared else 0.


def diagnose_gradients(model, kind, dataset, indices, device, arm, weights, micro_batch_size=1):
    """Measure fixed training probes without consuming training RNG or gradients.

    Each channel contribution is I_ic * l_ic / K_i / N, with N the fixed number
    of probe images, and the weighted analogue. Gradients are measured against
    all trainable shared encoder parameters. We report the mean of the L2 norms
    of the individual image contributions, NOT the norm of their vector sum.
    For predictable memory, forwards always use one image at a time, even when
    the caller's permitted micro_batch_size is larger than one.
    """
    from .organ_weighting import loss_components

    indices = list(indices)
    if not indices or any(not isinstance(index, int) or not 0 <= index < len(dataset) for index in indices):
        raise ValueError('indices must be a nonempty list of valid dataset indices')
    if micro_batch_size < 1:
        raise ValueError('micro_batch_size must be positive')
    if not hasattr(model, 'encoders'):
        raise ValueError('Q01 diagnostics require model.encoders')
    parameters = [parameter for parameter in model.encoders.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError('No trainable shared encoder parameters')
    task_ids = list(getattr(dataset, 'task_ids', PAPER_ALIGNED_TASK_IDS))
    records = {TASKS[task]: [] for task in task_ids}
    n = len(indices)
    with _read_only_model(model), torch.enable_grad():
        for index in indices:
            image, target = dataset[index]
            image, target = image.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            output = forward_model(model, kind, image, target, task_ids)
            logits = output['logits']
            components = loss_components(logits, target, arm, weights)
            active = components['active']
            k = active.sum(1).clamp_min(1)[0]
            valid, truth = target.ge(0), target.gt(0)
            pred = (logits.detach().sigmoid() >= .5) & valid
            unweighted = weighted = None
            for channel in active[0].nonzero().flatten().tolist():
                unweighted = components['per_channel_unweighted'][0, channel] / k / n
                weighted = components['per_channel_weighted'][0, channel] / k / n
                valid_pixels = int(valid[0, channel].sum())
                pred_pixels = int(pred[0, channel].sum())
                records[TASKS[task_ids[channel]]].append({
                    'index': index,
                    'unweighted_channel_loss': float(components['per_channel_unweighted'][0, channel].detach()),
                    'weighted_channel_loss': float(components['per_channel_weighted'][0, channel].detach()),
                    'unweighted_loss_contribution': float(unweighted.detach()),
                    'weighted_loss_contribution': float(weighted.detach()),
                    'unweighted_encoder_gradient_norm': _gradient_norm(unweighted, parameters),
                    'weighted_encoder_gradient_norm': _gradient_norm(weighted, parameters),
                    'positive': bool(truth[0, channel].any()),
                    'predicted_pixels': pred_pixels,
                    'predicted_area_fraction': pred_pixels / valid_pixels,
                    'truth_area_fraction': int(truth[0, channel].sum()) / valid_pixels,
                    'active_channels': int(k),
                })
            # Release every reference to this graph before the next forward,
            # including the final channel's scalar autograd nodes.
            del output, components, logits, unweighted, weighted
    per_structure = {}
    for name, rows in records.items():
        positives = [row for row in rows if row['positive']]
        empties = [row for row in rows if not row['positive']]
        summary = {'samples': len(rows), 'positive_samples': len(positives), 'empty_samples': len(empties)}
        for field in ('unweighted_channel_loss', 'weighted_channel_loss', 'unweighted_loss_contribution',
                      'weighted_loss_contribution', 'unweighted_encoder_gradient_norm', 'weighted_encoder_gradient_norm'):
            summary[field + '_mean'] = _mean(row[field] for row in rows)
            summary[field + '_sum'] = sum(row[field] for row in rows)
        for prefix, selected in [('positive', positives), ('empty', empties)]:
            for field in ('predicted_pixels', 'predicted_area_fraction', 'truth_area_fraction'):
                summary[f'{prefix}_mean_{field}'] = _mean(row[field] for row in selected)
        summary['images'] = rows
        per_structure[name] = summary
    return {
        'arm': arm, 'probe_indices': indices, 'probe_images': n,
        'threshold': .5, 'effective_micro_batch_size': 1,
        'encoder_parameters': sum(parameter.numel() for parameter in parameters),
        'loss_normalization': 'I_ic * channel_loss / K_i / N_probe_images',
        'gradient_aggregation': ('Mean/sum of single-image, per-structure encoder gradient L2 norms, '
                                 'including 1/K_i and 1/N_probe_images; not the norm of a joint gradient.'),
        'per_structure': per_structure,
    }
