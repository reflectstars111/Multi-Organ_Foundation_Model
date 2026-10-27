"""Q05 evaluation and paired reports; checkpoint selection lives in the suite.

Structures and foreground unions have different macro denominators.  This
module deliberately preserves both, the full all/positive cohorts and the
ground-truth-only surface-defined cohorts used by the common re-evaluation.
"""
from collections import defaultdict
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader, SequentialSampler

from .data import TASKS
from .fair_suite import forward_model
from .organ_diagnostics import (
    ABDOMEN_STRUCTURES, _accumulate, _empty_totals, _finalize,
    _read_only_model, _source,
)
from .reevaluation_inference import union_for_rows
from .reevaluation_report import COHORTS, FIELDS, summarize_rows
from .region import PAPER_ALIGNED_TASK_IDS
from .surface_metrics import binary_metrics


@contextmanager
def _preserve_loader_generators(loader):
    """A DataLoader iterator advances its private generator even without shuffle."""
    generators = {}
    for owner in (loader, getattr(loader, 'sampler', None),
                  getattr(getattr(loader, 'batch_sampler', None), 'sampler', None)):
        generator = getattr(owner, 'generator', None)
        if isinstance(generator, torch.Generator):
            generators.setdefault(id(generator), (generator, generator.get_state()))
    try:
        yield
    finally:
        for generator, state in generators.values():
            generator.set_state(state)


def _metadata(groups, index):
    records = groups[index]
    records = [records] if isinstance(records, dict) else records
    if not records or len({(row['dataset'], row['image']) for row in records}) != 1:
        raise ValueError('Each metadata group must identify exactly one image and domain')
    return records, records[0]


@torch.no_grad()
def evaluate(model, kind, loader, device, groups=None, task_ids=None):
    """Return diagnostics, per-image metric rows and common surface aggregates.

    Exactly one model forward is made per loader batch.  ``groups`` is required
    (or inferred from ``loader.dataset.groups``) because abdominal excluded
    classes and AUL mass-only unknown background cannot be recovered safely
    from the binary channel targets alone.  All model buffers, mode flags,
    global RNG and private loader generators are restored on exit.
    """
    task_ids = list(PAPER_ALIGNED_TASK_IDS if task_ids is None else task_ids)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError('task_ids must be unique')
    if groups is None:
        groups = getattr(getattr(loader, 'dataset', None), 'groups', None)
    if groups is None:
        raise ValueError('groups metadata is required for the annotated foreground union')
    if isinstance(loader, DataLoader):
        if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
            raise ValueError('groups require a sequential complete DataLoader')
        if len(groups) != len(loader.dataset):
            raise ValueError('groups length must match loader dataset')
    totals = {TASKS[task]: _empty_totals() for task in task_ids}
    source_totals = {}
    rows = []
    offset = 0
    with _read_only_model(model), _preserve_loader_generators(loader):
        for image, target in loader:
            if offset + len(image) > len(groups):
                raise ValueError('groups shorter than loader image count')
            image, target = image.to(device), target.to(device)
            if target.ndim != 4 or target.shape[1] != len(task_ids):
                raise ValueError('Target channels do not match task_ids')
            logits = forward_model(model, kind, image, target, task_ids)['logits']
            if logits.shape != target.shape:
                raise ValueError('Prediction shape differs from target shape')
            valid, truth = target.ge(0), target.gt(0)
            active = valid.any((2, 3))
            if not active.any(1).all():
                raise ValueError('An image has no active annotated channel')
            prediction = logits.sigmoid().ge(.5)
            pred = prediction & valid
            pred_pixels, truth_pixels = pred.sum((2, 3)), truth.sum((2, 3))
            intersection = (pred & truth).sum((2, 3))
            denominator = pred_pixels + truth_pixels
            dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.)
            values = torch.stack([dice, pred_pixels, truth_pixels, intersection,
                                  (pred & ~truth).sum((2, 3)), valid.sum((2, 3))], -1).cpu().tolist()
            predictions, targets, active_cpu = prediction.cpu(), target.cpu(), active.cpu()
            for batch in range(len(image)):
                records, meta = _metadata(groups, offset + batch)
                source = _source(records)
                if source is not None:
                    source_totals.setdefault(source, {name: _empty_totals() for name in ABDOMEN_STRUCTURES})
                metadata = dict(domain=meta['dataset'], source=source or '', image=meta['image'],
                                group=meta.get('group', ''), sample_index=offset + batch, variant='default')
                for channel in active_cpu[batch].nonzero().flatten().tolist():
                    task = task_ids[channel]
                    name = TASKS[task]
                    _accumulate(totals[name], values[batch][channel])
                    if name in ABDOMEN_STRUCTURES:
                        if source is None:
                            raise ValueError('groups must identify the CT source of every abdomen target')
                        _accumulate(source_totals[source][name], values[batch][channel])
                    row = binary_metrics(predictions[batch, channel].numpy(),
                                         targets[batch, channel].numpy(), empty_policy='diagonal')
                    row.update(metadata, view='structure', structure=name, task_id=task)
                    rows.append(row)
                union = union_for_rows(targets[batch], records, task_ids)
                union_prediction = predictions[batch, active_cpu[batch]].any(0)
                row = binary_metrics(union_prediction.numpy(), union.numpy(), empty_policy='diagonal')
                row.update(metadata, view='union')
                rows.append(row)
            offset += len(image)
    if offset != len(groups):
        raise ValueError('groups longer than loader image count')
    diagnostics = _finalize(totals)
    diagnostics['abdomen_by_source'] = {
        source: _finalize(values) for source, values in sorted(source_totals.items())}
    return dict(diagnostics=diagnostics, rows=rows, surface=summarize_rows(rows))


def _read(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def _diagnostic_scalars(value, prefix=''):
    """Report Dice/rates as percentages; preserve pixel counts as pixels."""
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            yield from _diagnostic_scalars(item, path)
        elif ('dice' in key or key.endswith('_rate') or key.endswith('_fraction')):
            yield path + '_percent', None if item is None else item * 100
        elif key.endswith('_pixels'):
            yield path, item


def _report_rows(result):
    identity = {key: result[key] for key in ('kind', 'arm', 'seed', 'selection', 'split')}
    for view in result['surface']:
        scopes = [('macro', {'all': view['macro']}),
                  ('abdomen_macro', {'abdomen': view['abdomen_macro']}),
                  (view['group_by'], view['groups']),
                  ('abdomen_source', view['abdomen_by_source'])]
        for scope, groups in scopes:
            for name, cohorts in groups.items():
                for cohort in COHORTS:
                    metrics = cohorts[cohort]
                    yield dict(identity, view=view['view'], scope=scope, group=name, cohort=cohort,
                               **{key: metrics.get(key) for key in FIELDS},
                               **{key: metrics.get(key) for key in (
                                   'distance_status', 'n_included', 'n_penalized', 'n_undefined',
                                   'n_coverage_total', 'n_target_boundary_undefined_excluded')})


def _scalar_rows(result):
    for row in _report_rows(result):
        common = {key: row[key] for key in (
            'kind', 'arm', 'seed', 'selection', 'split', 'view', 'scope', 'group', 'cohort')}
        for metric in FIELDS:
            yield dict(common, metric=metric, value=row[metric])
    for metric, value in _diagnostic_scalars(result.get('diagnostics', result)):
        # With suite's flattened format, ignore the nested surface summaries.
        if metric.startswith('surface.'):
            continue
        yield dict(kind=result['kind'], arm=result['arm'], seed=result['seed'],
                   selection=result['selection'], split=result['split'], view='structure',
                   scope='diagnostics', group='all', cohort='all', metric=metric, value=value)


_GROUP_KEYS = ('kind', 'selection', 'split', 'view', 'scope', 'group', 'cohort', 'metric')


def _moments(values):
    finite = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    complete = bool(values) and len(finite) == len(values)
    return dict(n_numeric=len(finite), mean=statistics.mean(finite) if complete else None,
                sample_std=statistics.stdev(finite) if complete and len(finite) > 1 else None)


def aggregate_paired(scalar_rows, seeds):
    """Refined minus control with paired seeds; never substitute guardrails."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in scalar_rows:
        identity = tuple(row[key] for key in _GROUP_KEYS)
        if row['seed'] in grouped[identity][row['arm']]:
            raise ValueError('Duplicate evaluation scalar for one seed and arm')
        grouped[identity][row['arm']][row['seed']] = row['value']
    seed_groups, paired = [], []
    for identity, arms in sorted(grouped.items()):
        common = dict(zip(_GROUP_KEYS, identity))
        for arm in ('control', 'refined'):
            values = arms[arm]
            seed_groups.append(dict(common, arm=arm, expected_seeds=list(seeds),
                selected_seeds=sorted(values), n_selected=len(values),
                missing_seeds=sorted(set(seeds) - values.keys()),
                values=[dict(seed=seed, value=value) for seed, value in sorted(values.items())],
                **_moments(list(values.values()))))
        control, refined = arms['control'], arms['refined']
        matched = sorted(control.keys() & refined.keys())
        pairs = [dict(seed=seed, control=control[seed], refined=refined[seed],
                      delta=(refined[seed] - control[seed]
                             if control[seed] is not None and refined[seed] is not None else None))
                 for seed in matched]
        paired.append(dict(common, comparison='refined_minus_control', expected_seeds=list(seeds),
                           paired_seeds=matched, n_paired=len(matched),
                           missing_pair_seeds=sorted(set(seeds) - set(matched)), pairs=pairs,
                           **_moments([row['delta'] for row in pairs])))
    return seed_groups, paired


def summarize(root, protocol):
    """Write additive reports; the suite owns and validates its summary.json.

    Call only after the suite has verified every completed checkpoint/test.
    This function additionally refuses incomplete matrices and missing selected
    evaluation files, so direct use cannot silently produce a partial report.
    """
    root = Path(root)
    config = protocol['config']
    seeds = list(config['seeds'])
    main_selection = _read(root / 'main_loss_selection.json')['selection_sha256']
    results, selection_coverage = [], []
    summaries = []
    for kind in config['models']:
        for arm in ('control', 'refined'):
            for seed in seeds:
                folder = root / arm / kind / f'seed_{seed}'
                summary = _read(folder / 'run_summary.json')
                if (summary.get('epochs_completed') != config['epochs']
                        or summary.get('test_evaluated') is not True
                        or summary.get('protocol_sha256') != protocol['protocol_sha256']
                        or summary.get('main_selection_sha256') != main_selection):
                    raise ValueError('Complete verified training/test matrix is required for Q05 reporting')
                summaries.append((folder, kind, arm, seed, summary))
    for folder, kind, arm, seed, summary in summaries:
        for selection in ('best', 'last', 'guardrail'):
            epoch = config['epochs'] if selection == 'last' else summary.get(f'{selection}_epoch')
            selection_coverage.append(dict(kind=kind, arm=arm, seed=seed,
                                           selection=selection, selected=epoch is not None, epoch=epoch))
            for split in ('val', 'test'):
                path = folder / 'evaluation' / f'{selection}_{split}.json'
                if epoch is None:
                    if selection != 'guardrail' or path.exists():
                        raise ValueError('Missing selection or unexpected unselected guardrail metrics')
                    continue
                result = _read(path)
                expected = dict(protocol_sha256=protocol['protocol_sha256'], checkpoint_epoch=epoch,
                                main_selection_sha256=main_selection)
                if any(result.get(key) != value for key, value in expected.items()):
                    raise ValueError(f'Evaluation identity differs from selected checkpoint: {path}')
                for key, value in dict(kind=kind, arm=arm, seed=seed).items():
                    if key in result and result[key] != value:
                        raise ValueError(f'Evaluation run identity mismatch: {path}')
                if (result.get('split', split) != split or result.get('selection', selection) != selection
                        or result.get('checkpoint_selection', selection) != selection):
                    raise ValueError(f'Evaluation selection/split mismatch: {path}')
                if 'surface' not in result:
                    raise ValueError(f'Missing common surface evaluation: {path}')
                results.append(dict(result, kind=kind, arm=arm, seed=seed, selection=selection, split=split))
    scalars = [row for result in results for row in _scalar_rows(result)]
    seed_groups, paired = aggregate_paired(scalars, seeds)
    report = dict(complete=True, protocol_sha256=protocol['protocol_sha256'],
                  main_selection_sha256=main_selection,
                  selection_coverage=selection_coverage, seed_groups=seed_groups, paired=paired,
                  metric_units=dict(dsc_percent='percent', hd95_pixel='pixel', assd_pixel='pixel'),
                  interpretation=('Equal structure and equal union-domain macros are separate. '
                      'Paired deltas are refined minus control on the same seed. '
                      'Missing guardrail selections are absent candidates, never substituted. '
                      'Nonfinite distances propagate as null; surface-defined cohorts depend only on GT. '
                      'This test partition has been inspected historically.'))
    _write_json(root / 'evaluation_summary.json', report)
    _write_csv(root / 'evaluation_per_group.csv', [row for result in results for row in _report_rows(result)])
    _write_csv(root / 'evaluation_metrics.csv', seed_groups,
               omit={'values', 'expected_seeds', 'selected_seeds', 'missing_seeds'})
    _write_csv(root / 'evaluation_paired.csv', paired,
               omit={'pairs', 'expected_seeds', 'paired_seeds', 'missing_pair_seeds'})
    _write_markdown(root, protocol, report)
    return report


def _write_csv(path, rows, omit=()):
    keys = list(dict.fromkeys(key for row in rows for key in row if key not in omit))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: 'NA' if row.get(key) is None else row.get(key) for key in keys})


def _render(row):
    if row['mean'] is None:
        return 'NA'
    suffix = f" ± {row['sample_std']:.2f}" if row['sample_std'] is not None else ''
    return f"{row['mean']:.2f}{suffix}"


def _write_markdown(root, protocol, report):
    rows = report['seed_groups']
    lines = ['# Q05 视觉上下文与任务调制：配对评价', '',
        f"冻结协议：`{protocol['protocol_sha256']}`。三种子均值 ± 样本标准差。", '',
        '两臂均固定 Region proposal 监督和 Task-ID 弱负载均衡；refined 组仅增加纯视觉前景/背景对比上下文或 Task-FiLM。公共初始化、抽样和主损失保持配对。', '',
        '结构表按16个结构等权平均；并集表先对每数据域图像平均，再对9个域等权平均。二者不是同一个任务或同一个分母。', '',
        'DSC (%) 越大越好，HD95 (pixel)、ASSD (pixel) 越小越好。单侧空掩膜使用像素中心对角线 hypot(511,511) ≈ 722.66 pixel；双方为空为100/0/0。', '',
        '`positive` 只按真值是否有前景划分。`*_surface_defined` 仅排除真值边界完全不可观测的样本，保留预测为空或失效的惩罚。完整队列遇不可定义距离时报告NA。', '',
        '`best` 为验证集结构阳性DSC最佳点，`last` 为第80轮，`guardrail` 为同时满足器官保护条件的最佳点。无合格点时不拿其他检查点替代，配对表明确报告缺失种子。', '',
        '| 模型 | 组别 | 选点 | 视图 | 集合 | 入选种子 | DSC (%) | HD95 (pixel) | ASSD (pixel) |',
        '|---|---|---|---|---|---:|---:|---:|---:|']
    for kind in protocol['config']['models']:
        for arm in ('control', 'refined'):
            for selection in ('best', 'last', 'guardrail'):
                for view in ('structure', 'union'):
                    for cohort in ('positive_surface_defined', 'all_surface_defined'):
                        found = {r['metric']: r for r in rows if r['kind'] == kind and r['arm'] == arm
                            and r['selection'] == selection and r['split'] == 'test' and r['view'] == view
                            and r['scope'] == 'macro' and r['cohort'] == cohort}
                        if not found:
                            continue
                        values = [kind, arm, selection, view, cohort, str(found['dsc_percent']['n_selected'])]
                        values += [_render(found[field]) for field in FIELDS]
                        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '配对差值统一为refined 组减 control 组，见 [evaluation_paired.csv](evaluation_paired.csv)。'
        '腹部血管、阳性漏检、空标签误检、CT来源及缺失选点在 '
        '[evaluation_summary.json](evaluation_summary.json) 中分别保留；逐结构/域/来源的三指标及覆盖数量见 '
        '[evaluation_per_group.csv](evaluation_per_group.csv)。测试划分已在历史研究中查看，不应称为未见外部验证。', '']
    (Path(root) / 'EVALUATION_REPORT.md').write_text('\n'.join(lines))
