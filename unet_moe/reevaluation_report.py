"""Reports for additive DSC/HD95/ASSD evaluation; no checkpoint selection."""
from collections import defaultdict
from collections import Counter
import csv
import hashlib
import io
import json
import statistics

from .surface_metrics import aggregate_rows


FIELDS = ('dsc_percent', 'hd95_pixel', 'assd_pixel')
COHORTS = ('all', 'positive', 'all_surface_defined', 'positive_surface_defined')
COVERAGE_FIELDS = ('n_coverage_total', 'n_target_boundary_undefined_excluded', 'n_target_boundary_undefined_total')
HEADERS = {'dsc_percent': 'DSC (%)', 'hd95_pixel': 'HD95 (pixel)', 'assd_pixel': 'ASSD (pixel)'}
INDEPENDENT_DOMAINS = ('BUSI', 'MMOTU', 'DDTI', 'AUL', 'FALLMUD', 'Fetal_HC', 'CCA', 'CAMUS', 'AbdomenUS')
SEED_IDENTITY_FIELDS = ('experiment', 'family', 'kind', 'arm', 'selection', 'split', 'size',
                        'training_status', 'training_epochs', 'epochs_completed', 'comparability_group',
                        'ensemble_status')


def macro_groups(groups):
    """Equal group weighting, preserving nonfinite statuses and denominators."""
    groups = list(groups)
    included = [x for x in groups if x['n_included']]
    statuses = {x['distance_status'] for x in included}
    status = ('infinite' if 'infinite' in statuses else 'undefined' if 'undefined' in statuses
              else 'finite' if included else 'excluded')
    result = dict(aggregation='equal_group_macro_mean', n_groups=len(included),
                  n_included=sum(x['n_included'] for x in included), distance_status=status,
                  n_penalized=sum(x['n_penalized'] for x in included),
                  n_undefined=sum(x['n_undefined'] for x in included),
                  n_infinite=sum(x['n_infinite'] for x in included))
    # Coverage denominators include groups with no eligible surface-labelled
    # samples; omitting those groups here would hide a completely excluded organ.
    result.update({field: sum(x.get(field, 0) for x in groups) for field in COVERAGE_FIELDS})
    for field in FIELDS:
        result[field] = (statistics.mean(x[field] for x in included)
                         if included and (field == 'dsc_percent' or status == 'finite') else None)
    return result


def aggregate_cohort(rows, cohort):
    """Apply a fixed ground-truth boundary cohort, never a prediction filter.

    A nonempty GT needs at least one retained (nonignored) surface pixel. An
    empty GT remains eligible: predicting foreground on it is still penalized.
    In particular, empty/failed predictions on eligible positive GT stay in the
    cohort, and prediction-only surface failures still propagate as undefined.
    """
    if cohort not in COHORTS:
        raise ValueError(f'Unknown evaluation cohort: {cohort}')
    rows = list(rows)
    positive = cohort.startswith('positive')
    base = aggregate_rows(rows, positive_only=positive)
    relevant = [row for row in rows if row['case_status'] != 'excluded'
                and (not positive or row['is_positive'])]
    undefined = [row for row in relevant if row['is_positive'] and row['target_surface_pixels'] == 0]
    restricted = cohort.endswith('_surface_defined')
    if restricted:
        eligible = [row for row in rows if row['is_positive'] is not True or row['target_surface_pixels'] > 0]
        result = aggregate_rows(eligible, positive_only=positive)
    else:
        result = base
    result.update(subset=cohort, n_coverage_total=base['n_included'],
                  n_target_boundary_undefined_total=len(undefined),
                  n_target_boundary_undefined_excluded=len(undefined) if restricted else 0)
    return result


def summarize_rows(rows):
    """Structures and union domains never share a macro denominator."""
    separated = defaultdict(list)
    for row in rows:
        if row['view'] not in ('structure', 'union'):
            raise ValueError('Unknown prediction view')
        if row['view'] == 'structure' and not row.get('structure'):
            raise ValueError('Structural evaluation needs a named structure')
        separated[(row['view'], row.get('variant') or 'default')].append(row)
    result = []
    for (view, variant), items in sorted(separated.items()):
        group_field = 'structure' if view == 'structure' else 'domain'
        grouped, sources = defaultdict(list), defaultdict(list)
        for row in items:
            grouped[row[group_field]].append(row)
            if row.get('source') and row['domain'] == 'AbdomenUS':
                sources[(row['source'], row[group_field])].append(row)
        group_metrics = {
            name: {cohort: aggregate_cohort(values, cohort) for cohort in COHORTS}
            for name, values in sorted(grouped.items())}
        macros = {cohort: macro_groups([v[cohort] for v in group_metrics.values()])
                  for cohort in COHORTS}
        abdomen = [name for name in group_metrics if name.startswith('abdomen_') or name == 'AbdomenUS']
        result.append(dict(view=view, variant=variant, group_by=group_field,
                           groups=group_metrics, macro=macros,
                           image_mean={cohort: aggregate_cohort(items, cohort) for cohort in COHORTS},
                           abdomen_macro={cohort: macro_groups([group_metrics[n][cohort] for n in abdomen])
                                          for cohort in COHORTS},
                           abdomen_by_source={source: {
                               cohort: macro_groups([aggregate_cohort(values, cohort)
                                                      for (src, _), values in sources.items() if src == source])
                               for cohort in COHORTS}
                               for source in sorted({src for src, _ in sources})}))
    return result


def display(value, status='finite'):
    if status == 'infinite':
        return 'Inf'
    return 'NA' if value is None else f'{value:.4f}'


def _pooled_image_aggregates(groups):
    """Recover image-weighted totals from disjoint domains' saved aggregates."""
    groups = list(groups)
    included = [item for item in groups if item['n_included']]
    status_set = {item['distance_status'] for item in included}
    status = ('infinite' if 'infinite' in status_set else 'undefined' if 'undefined' in status_set
              else 'finite' if included else 'excluded')
    count_names = ('n_rows', 'n_excluded', 'n_positive', 'n_negative', 'n_filtered_negative',
                   'n_included', 'n_finite', 'n_infinite', 'n_undefined', 'n_penalized')
    pooled = {name: sum(item[name] for item in groups) for name in count_names}
    pooled.update({name: sum(item.get(name, 0) for item in groups) for name in COVERAGE_FIELDS})
    pooled.update(aggregation='image_macro_mean', subset=groups[0]['subset'] if groups else 'all',
                  distance_status=status,
                  case_counts=dict(sum((Counter(item['case_counts']) for item in groups), Counter())),
                  empty_policies=sorted({policy for item in groups for policy in item['empty_policies']}))
    for field in FIELDS:
        pooled[field] = (sum(item[field] * item['n_included'] for item in included) / pooled['n_included']
                         if included and (field == 'dsc_percent' or status == 'finite') else None)
    finite = [item['finite_subset_mean'] for item in groups if item['finite_subset_mean']['n']]
    n_finite = sum(item['n'] for item in finite)
    pooled['finite_subset_mean'] = dict(n=n_finite, **{
        field: sum(item[field] * item['n'] for item in finite) / n_finite if n_finite else None
        for field in ('hd95_pixel', 'assd_pixel')})
    return pooled


def with_independent_ensembles(results):
    """Keep each domain model and add a clearly labelled nine-model total.

    This is a composition of independently trained domain networks, not another
    training run or another seed. Structure means are merged *before* taking
    the macro, because each domain can contain a different number of structures.
    Partial compositions explicitly list missing domains and never share a
    comparability group with the complete nine-domain composition.
    """
    originals = [result for result in results if not result.get('derived_from_independent')]
    composed = defaultdict(dict)
    identity_names = ('experiment', 'family', 'seed', 'selection', 'split', 'size',
                      'training_status', 'training_epochs', 'epochs_completed', 'comparability_group')
    for result in originals:
        job = result['job']
        if job.get('kind') != 'independent':
            continue
        domain = job.get('domain') or job.get('arm')
        if domain not in INDEPENDENT_DOMAINS:
            raise ValueError(f'Independent network has no recognized domain: {domain}')
        identity = tuple(job.get(name) for name in identity_names)
        for view in result['metrics']:
            key = (*identity, view['view'], view['variant'])
            if domain in composed[key]:
                raise ValueError(f'Duplicate independent network for the same seed/domain: {domain}')
            composed[key][domain] = (result, view)
    output = list(originals)
    for key, members in sorted(composed.items(), key=lambda item: str(item[0])):
        domains = sorted(members)
        missing = sorted(set(INDEPENDENT_DOMAINS) - set(domains))
        complete = not missing
        views = [view for _, view in members.values()]
        all_groups, abdomen_sources = {}, {}
        for view in views:
            for name, value in view['groups'].items():
                if name in all_groups:
                    raise ValueError(f'Duplicate structure/domain in independent composition: {name}')
                all_groups[name] = value
            for source, value in view['abdomen_by_source'].items():
                if source in abdomen_sources:
                    raise ValueError(f'Duplicate abdomen source in independent composition: {source}')
                abdomen_sources[source] = value
        abdomen = [name for name in all_groups if name.startswith('abdomen_') or name == 'AbdomenUS']
        metrics = dict(view=key[-2], variant=key[-1], group_by=views[0]['group_by'],
                       groups=dict(sorted(all_groups.items())),
                       macro={cohort: macro_groups([value[cohort] for value in all_groups.values()])
                              for cohort in COHORTS},
                       image_mean={cohort: _pooled_image_aggregates([view['image_mean'][cohort] for view in views])
                                   for cohort in COHORTS},
                       abdomen_macro={cohort: macro_groups([all_groups[name][cohort] for name in abdomen])
                                      for cohort in COHORTS},
                       abdomen_by_source=abdomen_sources)
        job = dict(zip(identity_names, key[:-2]))
        checkpoint_set = {domain: member[0]['checkpoint_sha256'] for domain, member in members.items()}
        member_jobs = sorted(member[0]['job']['job_id'] for member in members.values())
        job.update(kind='independent_ensemble', arm='all_domains', domain=None,
                   job_id=hashlib.sha256(json.dumps([key, member_jobs], sort_keys=True).encode()).hexdigest(),
                   ensemble_status='complete_9_domains' if complete else 'partial_domains',
                   ensemble_complete=complete, ensemble_domains=domains, ensemble_missing_domains=missing,
                   ensemble_expected_domains=list(INDEPENDENT_DOMAINS), member_job_ids=member_jobs,
                   comparability_group=str(job.get('comparability_group')) + ':independent-domains:' + ','.join(domains))
        output.append(dict(job=job, checkpoint_sha256=None, checkpoint_set=checkpoint_set,
                           derived_from_independent=True, metrics=[metrics]))
    return output


def report_rows(results):
    rows = []
    for result in with_independent_ensembles(results):
        job = result['job']
        base = {k: job.get(k) for k in ('job_id', 'experiment', 'family', 'kind', 'arm', 'seed',
                                       'selection', 'split', 'size', 'training_status', 'training_epochs',
                                       'epochs_completed', 'comparability_group', 'ensemble_status')}
        base['ensemble_complete'] = job.get('ensemble_complete')
        base['ensemble_domains'] = ';'.join(job.get('ensemble_domains', []))
        base['ensemble_missing_domains'] = ';'.join(job.get('ensemble_missing_domains', []))
        base['checkpoint_sha256'] = result['checkpoint_sha256']
        for view in result['metrics']:
            scopes = [('macro', 'all_groups', view['macro']),
                      ('image_mean', 'all_images', view['image_mean']),
                      ('abdomen_macro', 'abdomen', view['abdomen_macro'])]
            scopes += [('group', name, metrics) for name, metrics in view['groups'].items()]
            scopes += [('abdomen_source', name, metrics) for name, metrics in view['abdomen_by_source'].items()]
            for scope, name, cohorts in scopes:
                for cohort, values in cohorts.items():
                    rows.append({**base, 'view': view['view'], 'variant': view['variant'],
                                 'scope': scope, 'group': name, 'cohort': cohort,
                                 'samples': values['n_included'],
                                 'coverage_denominator': values['n_coverage_total'],
                                 'target_boundary_undefined_excluded': values['n_target_boundary_undefined_excluded'],
                                 'target_boundary_undefined_total': values['n_target_boundary_undefined_total'],
                                 **{HEADERS[field]: display(values[field], values['distance_status']
                                                           if field != 'dsc_percent' else 'finite')
                                    for field in FIELDS},
                                 'distance_status': values['distance_status'],
                                 'penalized_count': values['n_penalized'],
                                 'undefined_count': values['n_undefined']})
    return rows


def csv_text(rows):
    if not rows:
        return ''
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def seed_groups(results):
    """Group only the same experiment/split/selection/resolution/view/variant."""
    grouped = defaultdict(list)
    for result in with_independent_ensembles(results):
        job = result['job']
        for view in result['metrics']:
            for cohort, metric in view['macro'].items():
                key = tuple(job.get(k) for k in SEED_IDENTITY_FIELDS)
                grouped[(*key, view['view'], view['variant'], cohort)].append((job.get('seed'), metric))
    output = []
    names = (*SEED_IDENTITY_FIELDS, 'view', 'variant', 'cohort')
    for key, values in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        seeds = [seed for seed, _ in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'Duplicate result for the same seed and scientific protocol: {key}')
        statuses = {metric['distance_status'] for _, metric in values}
        status = ('infinite' if 'infinite' in statuses else 'undefined' if 'undefined' in statuses
                  else 'excluded' if 'excluded' in statuses else 'finite')
        row = dict(zip(names, key), seeds=seeds, n=len(seeds), distance_status=status)
        row['coverage_by_seed'] = [dict(seed=seed, n_included=metric['n_included'],
                                        **{field: metric.get(field, 0) for field in COVERAGE_FIELDS})
                                   for seed, metric in values]
        for field in FIELDS:
            numbers = [metric[field] for _, metric in values]
            usable = bool(numbers) and all(x is not None for x in numbers)
            row[field] = dict(mean=statistics.mean(numbers) if usable else None,
                              sample_std=statistics.stdev(numbers) if usable and len(numbers) > 1 else None)
        output.append(row)
    return output


def markdown_report(results, inventory, failed, updated_at):
    rows = [r for r in report_rows(results) if r['scope'] == 'macro'
            and r['cohort'] in ('positive', 'positive_surface_defined')]
    lines = ['# 医学实验统一补评：DSC / HD95 / ASSD', '', f'更新时间：{updated_at}。', '',
             'DSC以百分比表示，HD95与ASSD使用评价网格的像素单位；越高／越低／越低越好。',
             '单边空掩膜的距离采用图像对角线有限惩罚，并记录惩罚样本数。双方空时DSC=100、距离=0。',
             '下面主表并列阳性图像全集与GT边界可定义子集：先结构内（并集则域内）平均，再等权宏平均。全样本及逐器官结果见metrics.csv。',
             'positive_surface_defined只依据GT筛选：阳性GT至少有一个未受忽略区域影响的边界像素；不会删除空预测或漏检。原positive保留全部阳性图及NA。',
             '验证与测试、检查点选择、历史分辨率分别列出；已有检查点选择规则保持原样，不按新指标重新选优。', '',
             f'已完成补评任务：{len(results)}；失败：{len(failed)}。完整待评／延期／排除原因见inventory.json。', '',
             '| 实验 | 模型/组 | seed | 选择 | split | 尺寸 | 已训/计划轮次 | 状态 | 目标/规则 | cohort | DSC (%) | HD95 (pixel) | ASSD (pixel) | 惩罚数/样本数 | GT边界排除/原分母 |',
             '|---|---|---:|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['kind']}/{row['arm'] or '-'} | {row['seed']} | "
                     f"{row['selection']} | {row['split']} | {row['size']} | "
                     f"{row['epochs_completed'] if row['epochs_completed'] is not None else '-'}/"
                     f"{row['training_epochs'] if row['training_epochs'] is not None else '-'} | "
                     f"{row['ensemble_status'] or row['training_status'] or '-'} | {row['view']}/{row['variant']} | {row['cohort']} | "
                     f"{row['DSC (%)']} | {row['HD95 (pixel)']} | {row['ASSD (pixel)']} | "
                     f"{row['penalized_count']}/{row['samples']} | "
                     f"{row['target_boundary_undefined_excluded']}/{row['coverage_denominator']} |")
    lines += ['', '跨种子均值与样本标准差见summary.json的seed_groups（n同时列出，不能把未完成种子平均当完整结论）。',
              'independent_ensemble是独立网络按域组合后的整体评价，保留各域原始行；完整整体要求9域齐全，partial_domains明确表示尚缺域。它不计为额外训练或额外种子。',
              'archived_partial是历史中断实验，仅报告实际训练轮次对应的描述性验证结果，不与完整预算混排统计。',
              '距离因忽略区域而无法定义时记NA，并保留undefined_count；不会静默删除该样本来改善主指标。', '']
    return '\n'.join(lines)
