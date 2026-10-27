"""Read-only Q05 routing, proposal, adapter/expert-gradient and residual diagnostics."""
from contextlib import contextmanager
import math

import torch
from torch.utils.data import DataLoader, SequentialSampler

from . import organ_diagnostics as organs
from .data import TASKS
from .fair_suite import forward_model
from .region import PAPER_ALIGNED_TASK_IDS
from .routing_evaluation import _preserve_loader_generators
from .routing_objective import compute_objective


def _layers(model, kind):
    if kind == 'region':
        return model.moes
    if kind == 'task_id':
        return model.decoders
    raise ValueError('Q05 routing diagnostics require a MoE')


@contextmanager
def zero_expert_residual(model, kind):
    """Remove only expert outputs during inference, leaving the shared path intact."""
    handles = []
    try:
        for layer in _layers(model, kind):
            for expert in layer.experts:
                handles.append(expert.register_forward_hook(
                    lambda module, inputs, output: torch.zeros_like(output)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _router_scores(model, kind):
    scores, handles = {}, []
    try:
        for index, layer in enumerate(_layers(model, kind)):
            def capture(module, inputs, output, index=index):
                if index in scores:
                    raise RuntimeError('Router called twice before diagnostic consumption')
                scores[index] = output.detach().float().cpu()
            handles.append(layer.router.register_forward_hook(capture))
        yield scores
    finally:
        for handle in handles:
            handle.remove()


def _new_route(experts):
    return dict(queries=0, selected_counts=[0] * experts, probability_sum=[0.] * experts,
                entropy_sum=0., top1_probability_sum=0.)


def _add_route(row, probabilities, top_k):
    if probabilities.numel() == 0:
        return
    indices = probabilities.topk(top_k, -1).indices
    counts = torch.bincount(indices.flatten(), minlength=probabilities.shape[-1]).tolist()
    row['queries'] += len(probabilities)
    row['selected_counts'] = [a + b for a, b in zip(row['selected_counts'], counts)]
    row['probability_sum'] = [a + b for a, b in zip(row['probability_sum'], probabilities.sum(0).tolist())]
    row['entropy_sum'] += float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
    row['top1_probability_sum'] += float(probabilities.max(-1).values.sum())


def _finish_route(row):
    n = row['queries']
    return {**row, 'mean_softmax_entropy_nats': row['entropy_sum'] / n if n else None,
            'mean_top1_probability': row['top1_probability_sum'] / n if n else None,
            'mean_probability': [v / n for v in row['probability_sum']] if n else None,
            'experts_selected': sum(v > 0 for v in row['selected_counts'])}


def _add_predictions(totals, source_totals, logits, target, task_ids, sources):
    valid, truth = target.ge(0), target.gt(0)
    pred = logits.ge(0) & valid
    pn, tn = pred.sum((2, 3)), truth.sum((2, 3))
    overlap = (pred & truth).sum((2, 3))
    denom = pn + tn
    dice = torch.where(denom > 0, 2 * overlap / denom.clamp_min(1), 1.)
    values = torch.stack([dice, pn, tn, overlap, (pred & ~truth).sum((2, 3)),
                          valid.sum((2, 3))], -1).cpu().tolist()
    for b, c in valid.any((2, 3)).cpu().nonzero().tolist():
        name = TASKS[task_ids[c]]
        organs._accumulate(totals[name], values[b][c])
        if sources[b] is not None and name in organs.ABDOMEN_STRUCTURES:
            st = source_totals.setdefault(sources[b], {n: organs._empty_totals()
                                                       for n in organs.ABDOMEN_STRUCTURES})
            organs._accumulate(st[name], values[b][c])


def evaluate_routing(model, kind, loader, device, groups=None, task_ids=None):
    """Preserve the loader's independent RNG across both validation passes."""
    with _preserve_loader_generators(loader):
        return _evaluate_routing(model, kind, loader, device, groups, task_ids)


@torch.no_grad()
def _evaluate_routing(model, kind, loader, device, groups=None, task_ids=None):
    """Whole validation: all proposals vs annotated tasks, including positive/empty.

    Region global/domain counts include all 16 proposals; organ tables include
    only annotated channels. Task-ID counts only actual annotated queries.
    The second full pass zeros expert residuals, with identical batches/masks.
    """
    task_ids = list(PAPER_ALIGNED_TASK_IDS if task_ids is None else task_ids)
    if isinstance(loader, DataLoader) and (not isinstance(loader.sampler, SequentialSampler)
                                           or loader.drop_last):
        raise ValueError('Routing evaluation requires a complete sequential loader')
    if groups is None:
        groups = getattr(getattr(loader, 'dataset', None), 'groups', None)
    if groups is None:
        raise ValueError('Routing evaluation requires image/domain metadata')
    if isinstance(loader, DataLoader) and len(groups) != len(loader.dataset):
        raise ValueError('Routing metadata length differs from dataset')
    if any(row.get('split') != 'val' for group in groups for row in group):
        raise ValueError('Residual ablation is restricted to validation data')
    layers = _layers(model, kind)
    reports = [dict(all=_new_route(len(layer.experts)), by_domain={}, by_structure={})
               for layer in layers]
    proposals = [{TASKS[t]: organs._empty_totals() for t in task_ids} for _ in layers]
    totals = {TASKS[t]: organs._empty_totals() for t in task_ids}
    source_totals, offset = {}, 0
    with organs._read_only_model(model), _router_scores(model, kind) as captured:
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            metadata = groups[offset:offset + len(image)]
            if len(metadata) != len(image):
                raise ValueError('Insufficient image metadata')
            domains = [rows[0]['dataset'] for rows in metadata]
            sources = [organs._source(rows) for rows in metadata]
            active = target.ge(0).any((2, 3)).cpu()
            positive = target.gt(0).any((2, 3)).cpu()
            batch, channel = active.nonzero(as_tuple=True)
            output = forward_model(model, kind, image, target, task_ids)
            _add_predictions(totals, source_totals, output['logits'], target, task_ids, sources)
            if len(captured) != len(layers):
                raise ValueError('Missing router output')
            for li, layer in enumerate(layers):
                p = captured.pop(li).softmax(-1)
                r, e = reports[li], len(layer.experts)
                expected = (len(image), len(task_ids), e) if kind == 'region' else (len(batch), e)
                if tuple(p.shape) != expected:
                    raise ValueError('Router dimensions differ from annotated query order')
                _add_route(r['all'], p.reshape(-1, e), layer.top_k)
                for domain in sorted(set(domains)):
                    mask = torch.tensor([d == domain for d in domains])
                    selected = p[mask].reshape(-1, e) if kind == 'region' else p[mask[batch]]
                    _add_route(r['by_domain'].setdefault(domain, _new_route(e)), selected, layer.top_k)
                annotated = p[batch, channel] if kind == 'region' else p
                for c, task in enumerate(task_ids):
                    rows = r['by_structure'].setdefault(TASKS[task], {
                        cohort: _new_route(e) for cohort in ('all', 'positive', 'empty')})
                    for cohort in rows:
                        mask = channel == c
                        if cohort != 'all':
                            pos = positive[batch, channel]
                            mask = mask & (pos if cohort == 'positive' else ~pos)
                        _add_route(rows[cohort], annotated[mask], layer.top_k)
                if kind == 'region':
                    _add_predictions(proposals[li], {}, output['coarse'][li], target, task_ids,
                                     [None] * len(image))
            offset += len(image)
        if offset != len(groups):
            raise ValueError('Validation pass did not consume all metadata')
    normal = organs._finalize(totals)
    normal['abdomen_by_source'] = {s: organs._finalize(t) for s, t in sorted(source_totals.items())}
    with zero_expert_residual(model, kind):
        zero = organs.evaluate(model, kind, loader, device, groups=groups, task_ids=task_ids)
    for li, report in enumerate(reports):
        report['all'] = _finish_route(report['all'])
        report['by_domain'] = {d: _finish_route(v) for d, v in report['by_domain'].items()}
        report['by_structure'] = {s: {c: _finish_route(v) for c, v in rows.items()}
                                  for s, rows in report['by_structure'].items()}
        report['proposal_metrics'] = organs._finalize(proposals[li]) if kind == 'region' else None
    delta = {key: normal[key] - zero[key] if normal[key] is not None and zero[key] is not None else None
             for key in ('macro_positive_dice', 'abdomen_macro_positive_dice')}
    return dict(split='val', images=offset, layer_order='coarsest to finest decoder',
                region_global_scope='all proposals including unannotated; organ tables annotated only',
                entropy_unit='nats', threshold=.5, layers=reports, normal=normal,
                zero_expert_residual=zero, normal_minus_zero=delta)


def diagnose(model, kind, dataset, indices, device, main_loss_arm, weights, arm='control'):
    """Train-only, fixed single-image gradient probes; no .grad/RNG mutation.

    Norms are of each image's total objective, averaged over all probe images;
    they are not the norm of the accumulated effective-batch gradient.
    """
    indices = list(indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Gradient probe requires unique nonempty image indices')
    if any(r.get('split') != 'train' for i in indices for r in dataset.groups[i]):
        raise ValueError('Expert gradients require training-only probes')
    if arm not in ('control', 'refined'):
        raise ValueError('Unknown Q05 architecture arm')
    named = []
    for li, layer in enumerate(_layers(model, kind)):
        named.extend((f'layer_{li}/expert_{ei}', expert) for ei, expert in enumerate(layer.experts))
        named.extend([(f'layer_{li}/shared', layer.shared), (f'layer_{li}/router', layer.router)])
        for name in ('contrast_projection', 'film'):
            if hasattr(layer, name):
                named.append((f'layer_{li}/{name}', getattr(layer, name)))
    parameters, spans = [], []
    for name, module in named:
        start = len(parameters)
        parameters.extend(p for p in module.parameters() if p.requires_grad)
        spans.append((name, start, len(parameters)))
    totals = {name: dict(norm_sum=0., nonzero_images=0, connected_images=0) for name, _, _ in spans}
    components = dict(main=0., coarse=0., balance=0., total=0.)
    with organs._read_only_model(model), torch.enable_grad():
        for index in indices:
            image, target = dataset[index]
            loss, terms = compute_objective(model, kind, image[None].to(device), target[None].to(device),
                                            'auxiliary', main_loss_arm, weights)
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            for name, start, end in spans:
                gs = [g for g in gradients[start:end] if g is not None]
                norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs))
                if not math.isfinite(norm):
                    raise FloatingPointError('Nonfinite expert gradient probe')
                totals[name]['norm_sum'] += norm
                totals[name]['nonzero_images'] += int(norm > 0)
                totals[name]['connected_images'] += int(bool(gs))
            for key in components:
                components[key] += terms[key]
            del loss, gradients
    n = len(indices)
    return dict(split='train', arm=arm, objective_arm='q02_auxiliary', main_loss_arm=main_loss_arm, probe_indices=indices,
                probe_images=n, physical_micro_batch_size=1,
                gradient_definition='mean per-image objective gradient L2 norm, not norm of a batch gradient',
                mean_objective={k: v / n for k, v in components.items()},
                modules={name: {**v, 'mean_gradient_norm': v['norm_sum'] / n} for name, v in totals.items()})
