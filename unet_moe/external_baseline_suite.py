"""Frozen 512-pixel Q04 training, resumable optimization, and held-out evaluation."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .external_baselines import METHODS, build_model
from .external_baselines.protocol import (assert_checkpoint, assert_matrix_complete, file_sha,
    read_json, run_folder, verify_protocol, write_json)
from .external_baselines.support import SupportBank
from .fair_suite import (_digest, _restore_rng, _rng_state, _save_checkpoint, _seed,
                         _synchronize, _write_history, common_loss)
from .region import DOMAINS
from .sampling import domain_draws
from .single_channel_suite import UnionDataset
from .surface_metrics import binary_metrics
from .reevaluation_report import summarize_rows, FIELDS, COHORTS


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def make_model(root, protocol, method, seed, *, allow_untrained_prior=False):
    prior = Path(root) / 'mofo_prior' / f'seed_{seed}' / 'last.pt'
    records_sha = file_sha(Path(root) / 'records.json')
    support_sha = file_sha(Path(root) / 'support_manifest.json')
    if method == 'mofo':
        checkpoint = torch.load(prior, map_location='cpu', weights_only=True)
        prior_protocol = checkpoint.get('protocol', {})
        expected = dict(epochs=protocol['config']['mofo_prior']['epochs'], seed=seed,
                        records_sha256=records_sha, support_manifest_sha256=support_sha,
                        size=protocol['config']['size'])
        if (checkpoint.get('complete') is not True or checkpoint.get('epoch') != expected['epochs']
                or any(prior_protocol.get(key) != value for key, value in expected.items())):
            raise ValueError('MOFO primary training requires the complete frozen 80-epoch, same-seed train-only prior')
        del checkpoint
    return build_model(method, assets_dir=protocol['config']['assets_dir'],
                       prior_checkpoint=prior if method == 'mofo' else None,
                       expected_seed=seed,
                       records_sha256=records_sha, support_manifest_sha256=support_sha)


def objective(model, method, output, target, domains):
    loss = common_loss(output['logits'], target)
    if method == 'smart':
        aux = output['auxiliary_logits']
        if len(aux) != 1:
            raise ValueError('SMART needs its original second-scale segmentation head')
        loss = loss + .4 * common_loss(aux[0], target)
    if method == 'mofo':
        loss = loss + model.auxiliary_loss(output, target, domains)
    return loss


def batch_forward(model, method, dataset, indices, device, support_bank=None,
                  *, seed=None, epoch=0, draw_positions=None):
    items = [dataset[index] for index in indices]
    images = torch.stack([item[0] for item in items]).to(device)
    target = torch.stack([item[1] for item in items]).to(device)
    domains = torch.tensor([DOMAINS.index(item[2]) for item in items], device=device)
    if not bool(target.ge(0).any((1, 2, 3)).all()):
        raise ValueError('Query contains no valid union supervision')
    kwargs = {}
    supports = []
    if method == 'universeg':
        queries = [dataset.groups[index][0] for index in indices]
        si, sl, supports = support_bank.batch(queries, device=device, training_seed=seed,
                                              epoch=epoch, draw_positions=draw_positions)
        kwargs = dict(support_images=si, support_labels=sl)
    output = model(images, domains, **kwargs)
    if output['logits'].shape != target.shape or not torch.isfinite(output['logits']).all():
        raise ValueError('Invalid/nonfinite segmentation logits')
    return output, target, domains, supports


def optimization_step(model, method, optimizer, dataset, indices, device,
                      micro_batch_size, support_bank=None, *, seed, epoch, offset=0):
    """Accumulate sample-weighted microbatches, including the final short batch.

    Model BatchNorm observes the declared physical batch, not the effective
    query batch. This is recorded per method and not claimed to be equivalent.
    """
    optimizer.zero_grad(set_to_none=True)
    total, support_count = 0., 0
    support_hash = hashlib.sha256()
    for start in range(0, len(indices), micro_batch_size):
        chunk = indices[start:start + micro_batch_size]
        output, target, domains, supports = batch_forward(
            model, method, dataset, chunk, device, support_bank, seed=seed, epoch=epoch,
            draw_positions=list(range(offset + start, offset + start + len(chunk))))
        loss = objective(model, method, output, target, domains)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite Q04 loss')
        (loss * (len(chunk) / len(indices))).backward()
        total += float(loss.detach()) * len(chunk) / len(indices)
        support_count += sum(map(len, supports))
        support_hash.update(json.dumps(supports).encode())
        del output, target, domains, loss
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return total, support_count, support_hash.hexdigest(), float(norm)


@torch.no_grad()
def validate(model, method, dataset, device, batch_size, support_bank=None, *, surfaces=False):
    model.eval()
    totals = {domain: [] for domain in DOMAINS}
    rows = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        output, target, _, _ = batch_forward(model, method, dataset, indices, device, support_bank)
        prediction = output['logits'].sigmoid().ge(.5).cpu().numpy()[:, 0]
        truth = target.cpu().numpy()[:, 0]
        for n, index in enumerate(indices):
            meta = dataset.groups[index][0]
            valid = truth[n] >= 0
            p, y = prediction[n] & valid, truth[n] > 0
            denominator = int(p.sum()) + int(y.sum())
            dsc = 200 * int((p & y).sum()) / denominator if denominator else 100.
            totals[meta['dataset']].append((dsc, bool(y.any())))
            if surfaces:
                row = binary_metrics(prediction[n], truth[n], empty_policy='diagonal')
                row.update(view='union', variant='default', domain=meta['dataset'],
                           source=meta['group'].split('/')[-1] if meta['dataset'] == 'AbdomenUS' else '',
                           image=meta['image'], group=meta['group'])
                rows.append(row)
    per_domain = {}
    for name, samples in totals.items():
        positives = [score for score, positive in samples if positive]
        if not positives:
            raise ValueError(f'No positive validation samples in domain {name}')
        per_domain[name] = dict(n=len(samples), n_positive=len(positives),
            positive_dsc_percent=statistics.mean(positives),
            all_dsc_percent=statistics.mean(x[0] for x in samples))
    metrics = dict(per_domain=per_domain,
        macro_positive_dsc_percent=statistics.mean(x['positive_dsc_percent'] for x in per_domain.values()))
    if surfaces:
        metrics['surface_metrics'] = summarize_rows(rows)
    return metrics, rows


def _datasets(records, size):
    return {split: UnionDataset([r for r in records if r['split'] == split], size)
            for split in ('train', 'val')}


def _bank(root, records, method, size):
    return SupportBank(records, read_json(Path(root) / 'support_manifest.json'), size) if method == 'universeg' else None


def train(root, protocol, records, method, seed, device, stop_after_epoch=None):
    config = protocol['config']
    report = read_json(Path(root) / 'preflight' / f'{method}.json')
    if (report.get('passed') is not True or report.get('protocol_sha256') != protocol['protocol_sha256']
            or report.get('method') != method or report.get('size') != config['size']
            or report.get('physical_batch') != config['micro_batch_sizes'][method]
            or report.get('effective_batch') != config['batch_size']
            or report.get('real_optimizer_updates') != 2
            or report.get('strict_model_optimizer_rng_reload') is not True
            or report.get('identical_reload_prediction') is not True):
        raise ValueError('Training requires successful GPU preflight of this frozen protocol')
    folder = run_folder(root, method, seed)
    folder.mkdir(parents=True, exist_ok=True)
    _seed(seed)
    model = make_model(root, protocol, method, seed).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    datasets = _datasets(records, config['size'])
    bank = _bank(root, records, method, config['size'])
    history, best_score, best_epoch, elapsed = [], -float('inf'), None, 0.
    if (folder / 'last.pt').exists():
        cp = torch.load(folder / 'last.pt', map_location=device, weights_only=True)
        assert_checkpoint(cp, protocol, method, seed)
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        history = cp['history']
        if [r['epoch'] for r in history] != list(range(1, cp['epoch'] + 1)):
            raise ValueError('Nonsequential checkpoint history')
        best_score, best_epoch, elapsed = cp['best_score'], cp['best_epoch'], cp['elapsed_seconds']
        if best_epoch == cp['epoch']:
            _save_checkpoint(folder / 'best.pt', cp)
        if not (folder / 'best.pt').exists():
            raise ValueError('Selected best checkpoint is missing')
        _restore_rng(cp['rng'])
        del cp
        _write_history(folder / 'metrics.jsonl', history)
        # last.pt is atomically published before history/summary files. A crash
        # in that small interval at epoch80 must be recoverable even though no
        # further training iteration will run.
        previous_summary = read_json(folder / 'run_summary.json') if (folder / 'run_summary.json').exists() else {}
        write_json(folder / 'run_summary.json', dict(method=method, seed=seed,
            protocol_sha256=protocol['protocol_sha256'], epochs_completed=len(history),
            best_epoch=best_epoch, best_validation_dsc_percent=best_score,
            parameter_count=sum(p.numel() for p in model.parameters()), elapsed_seconds=elapsed,
            test_evaluated=previous_summary.get('test_evaluated') is True
                           and previous_summary.get('epochs_completed') == len(history)))
    elif any((folder / n).exists() for n in ('best.pt', 'metrics.jsonl')):
        raise ValueError('Ambiguous existing run without resumable last.pt')
    plans = read_json(Path(root) / 'sampling_plan.json')[str(seed)]
    micro = config['micro_batch_sizes'][method]
    parameters = sum(p.numel() for p in model.parameters())
    for epoch in range(len(history) + 1, min(config['epochs'], stop_after_epoch or config['epochs']) + 1):
        draws = domain_draws(datasets['train'].groups, config['quota_per_domain'], seed, epoch)
        plan = plans[epoch - 1]
        image_hash = _digest([datasets['train'].groups[i][0]['image'] for i in draws])
        if draws != plan['indices'] or image_hash != plan['sample_sha256']:
            raise ValueError('Q04 query sequence differs from the frozen reference')
        model.train()
        started = time.monotonic()
        loss_sum, support_count, steps = 0., 0, 0
        support_hash = hashlib.sha256()
        for offset in range(0, len(draws), config['batch_size']):
            indices = draws[offset:offset + config['batch_size']]
            loss, count, fingerprint, norm = optimization_step(
                model, method, optimizer, datasets['train'], indices, device, micro, bank,
                seed=seed, epoch=epoch, offset=offset)
            loss_sum += loss * len(indices)
            support_count += count
            support_hash.update(fingerprint.encode())
            steps += 1
            if steps % 10 == 0 or offset + len(indices) == len(draws):
                write_json(folder / 'progress.json', dict(state='training', epoch=epoch,
                    optimizer_step=steps, optimizer_steps_per_epoch=282,
                    completed_epochs=len(history), updated_at=timestamp()))
        _synchronize(device)
        train_seconds = time.monotonic() - started
        started = time.monotonic()
        validation, _ = validate(model, method, datasets['val'], device, micro, bank)
        _synchronize(device)
        validation_seconds = time.monotonic() - started
        elapsed += train_seconds + validation_seconds
        score = validation['macro_positive_dsc_percent']
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        row = dict(epoch=epoch, method=method, seed=seed, train_loss=loss_sum / len(draws),
            sample_count=len(draws), sample_sha256=image_hash, optimizer_steps=steps,
            support_draws=support_count, support_draws_sha256=support_hash.hexdigest(),
            train_seconds=train_seconds, validation_seconds=validation_seconds,
            elapsed_seconds=elapsed, validation=validation)
        history.append(row)
        cp = dict(schema='q04_checkpoint_v1', protocol_sha256=protocol['protocol_sha256'],
            method=method, seed=seed, epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), rng=_rng_state(), history=history,
            best_score=best_score, best_epoch=best_epoch, elapsed_seconds=elapsed,
            parameter_count=parameters)
        _save_checkpoint(folder / 'last.pt', cp)
        if improved:
            _save_checkpoint(folder / 'best.pt', cp)
        _write_history(folder / 'metrics.jsonl', history)
        write_json(folder / 'run_summary.json', dict(method=method, seed=seed,
            protocol_sha256=protocol['protocol_sha256'], epochs_completed=epoch,
            best_epoch=best_epoch, best_validation_dsc_percent=best_score,
            parameter_count=parameters, elapsed_seconds=elapsed, test_evaluated=False))
        print(json.dumps(row), flush=True)
    return history


def verify_run(root, protocol, method, seed, require_test=False):
    folder = run_folder(root, method, seed)
    cp = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
    assert_checkpoint(cp, protocol, method, seed)
    if cp['epoch'] != protocol['config']['epochs']:
        raise ValueError('Training is incomplete for the frozen epoch budget')
    if [r['epoch'] for r in cp['history']] != list(range(1, cp['epoch'] + 1)):
        raise ValueError('Nonsequential epoch history')
    summary = read_json(folder / 'run_summary.json')
    if (summary.get('protocol_sha256') != protocol['protocol_sha256']
            or summary.get('method') != method or summary.get('seed') != seed):
        raise ValueError('Summary method/seed/protocol mismatch')
    if summary['epochs_completed'] != cp['epoch'] or summary['best_epoch'] != cp['best_epoch']:
        raise ValueError('Summary disagrees with checkpoint')
    plans = read_json(Path(root) / 'sampling_plan.json')[str(seed)]
    for actual, planned in zip(cp['history'], plans, strict=True):
        if (actual['sample_sha256'] != planned['sample_sha256'] or actual['sample_count'] != 4500
                or actual['optimizer_steps'] != 282):
            raise ValueError('Checkpoint training history violates frozen sampling budget')
    persisted_history = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if persisted_history != cp['history']:
        raise ValueError('Persisted metrics history differs from the checkpoint')
    expected_best = max(cp['history'], key=lambda row: row['validation']['macro_positive_dsc_percent'])
    if (cp['best_epoch'] != expected_best['epoch']
            or cp['best_score'] != expected_best['validation']['macro_positive_dsc_percent']):
        raise ValueError('Best checkpoint selection disagrees with validation history')
    best_epoch = cp['best_epoch']
    best_history = cp['history'][:best_epoch]
    del cp
    best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    assert_checkpoint(best, protocol, method, seed)
    if best['epoch'] != best_epoch:
        raise ValueError('Best checkpoint disagrees with selection')
    if best['history'] != best_history:
        raise ValueError('Best checkpoint history disagrees with the selected prefix')
    del best
    if require_test:
        expected_content = read_json(Path(root) / 'test_content_expected.json')
        if read_json(Path(root) / 'test_content_manifest.json') != expected_content:
            raise ValueError('Evaluation used different test content from frozen Q01')
        content_digest = _digest(expected_content)
        for selection in ('best', 'last'):
            for split in ('val', 'test'):
                result = read_json(folder / f'{selection}_{split}_metrics.json')
                if (not result['complete'] or result['protocol_sha256'] != protocol['protocol_sha256']
                        or result.get('method') != method or result.get('seed') != seed
                        or result.get('selection') != selection or result.get('split') != split
                        or result.get('test_content_sha256') != content_digest
                        or result['checkpoint_sha256'] != file_sha(folder / f'{selection}.pt')
                        or result['per_image_sha256'] != file_sha(folder / f'{selection}_{split}_per_image.jsonl')):
                    raise ValueError('Evaluation identity/contents mismatch')
        if not summary['test_evaluated']:
            raise ValueError('Evaluation summary incomplete')
    return summary


def evaluate_run(root, protocol, records, method, seed, device):
    assert_matrix_complete(root, protocol)
    folder = run_folder(root, method, seed)
    # Test content is opened only after all training and checkpoint selections.
    content = {}
    for path in sorted({r[k] for r in records if r['split'] == 'test' for k in ('image', 'mask')}):
        content[path] = dict(bytes=Path(path).stat().st_size, sha256=file_sha(path))
    if read_json(Path(root) / 'test_content_expected.json') != content:
        raise ValueError('Test images/masks changed since the frozen Q01 data protocol')
    test_path = Path(root) / 'test_content_manifest.json'
    if test_path.exists() and read_json(test_path) != content:
        raise ValueError('Test content changed after evaluation began')
    write_json(test_path, content)
    bank = _bank(root, records, method, protocol['config']['size'])
    for selection in ('best', 'last'):
        cp = torch.load(folder / f'{selection}.pt', map_location='cpu', weights_only=True)
        assert_checkpoint(cp, protocol, method, seed)
        model = make_model(root, protocol, method, seed)
        model.load_state_dict(cp['model'], strict=True)
        epoch = cp['epoch']
        del cp
        model.to(device)
        checkpoint_sha = file_sha(folder / f'{selection}.pt')
        for split in ('val', 'test'):
            result_path = folder / f'{selection}_{split}_metrics.json'
            dataset = UnionDataset([r for r in records if r['split'] == split], protocol['config']['size'])
            result, rows = validate(model, method, dataset, device,
                protocol['config']['micro_batch_sizes'][method], bank, surfaces=True)
            per_image = folder / f'{selection}_{split}_per_image.jsonl'
            _write_history(per_image, rows)
            result.update(complete=True, method=method, seed=seed, split=split, selection=selection,
                protocol_sha256=protocol['protocol_sha256'], checkpoint_sha256=checkpoint_sha,
                checkpoint_epoch=epoch, per_image_sha256=file_sha(per_image),
                test_content_sha256=_digest(content), updated_at=timestamp())
            write_json(result_path, result)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = read_json(folder / 'run_summary.json')
    summary['test_evaluated'] = True
    write_json(folder / 'run_summary.json', summary)


def summarize(root, protocol):
    results, groups = [], []
    for method in protocol['config']['methods']:
        for seed in protocol['config']['seeds']:
            verify_run(root, protocol, method, seed, require_test=True)
            for selection in ('best', 'last'):
                results.append(read_json(run_folder(root, method, seed) / f'{selection}_test_metrics.json'))
        for selection in ('best', 'last'):
            runs = [r for r in results if r['method'] == method and r['selection'] == selection]
            for cohort in COHORTS:
                group = dict(method=method, selection=selection, cohort=cohort, seeds=len(runs), metrics={})
                for field in FIELDS:
                    values = [r['surface_metrics'][0]['macro'][cohort][field] for r in runs]
                    group['metrics'][field] = dict(
                        mean=statistics.mean(values) if all(v is not None for v in values) else None,
                        sample_std=statistics.stdev(values) if len(values) > 1 and all(v is not None for v in values) else None)
                groups.append(group)
    write_json(Path(root) / 'summary.json', dict(complete=True,
        protocol_sha256=protocol['protocol_sha256'], results=results, seed_groups=groups, updated_at=timestamp()))
    write_reports(root, protocol, results, groups)


def write_reports(root, protocol, results, groups):
    """Export seed and cohort denominators alongside the three common metrics."""
    root = Path(root)
    fields = ['method', 'selection', 'cohort', 'seeds']
    fields += [f'{metric}_{statistic}' for metric in FIELDS for statistic in ('mean', 'sample_std')]
    with (root / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            row = {name: group[name] for name in fields[:4]}
            row.update({f'{metric}_{statistic}': group['metrics'][metric][statistic]
                        for metric in FIELDS for statistic in ('mean', 'sample_std')})
            writer.writerow({key: 'NA' if value is None else value for key, value in row.items()})
    fields = ['method', 'seed', 'selection', 'split', 'scope', 'group', 'cohort',
              *FIELDS, 'n_included', 'n_coverage_total', 'n_penalized',
              'n_target_boundary_undefined_excluded', 'distance_status']
    with (root / 'per_domain_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            union = next(item for item in result['surface_metrics'] if item['view'] == 'union')
            scopes = [('domain', union['groups']), ('abdomen_source', union['abdomen_by_source'])]
            for scope, entries in scopes:
                for name, cohorts in entries.items():
                    for cohort, metrics in cohorts.items():
                        row = {key: result[key] for key in ('method', 'seed', 'selection', 'split')}
                        row.update(scope=scope, group=name, cohort=cohort)
                        row.update({key: metrics.get(key) for key in fields[7:]})
                        writer.writerow({key: 'NA' if value is None else value for key, value in row.items()})

    def render(value):
        if value['mean'] is None:
            return 'NA'
        deviation = f"{value['sample_std']:.2f}" if value['sample_std'] is not None else 'NA'
        return f"{value['mean']:.2f} ± {deviation}"

    lines = ['# Q04 五种外部 baseline 统一并集评价', '',
        f"协议：`{protocol['protocol_sha256']}`。80轮、512×512、seed42/43/44，9数据域宏平均；下表为三种子均值 ± 样本标准差。", '',
        '`best` 按验证集阳性并集DSC选择；`last` 为固定第80轮。两种选择分别报告，不能混作一个结果。', '',
        '指标：DSC (%)、HD95 (pixel)、ASSD (pixel)。单侧空掩膜距离取图像对角线 hypot(511,511)，双方为空取100/0/0。', '',
        '`all`/`positive` 保留完整相应样本；`*_surface_defined` 只按GT边界是否可定义固定子集，仍保留空预测惩罚。不可定义距离报告NA，不通过删除预测失败样本产生有限均值。', '',
        'GPU preflight 是正式训练之前的准备阶段：两次真实有效batch16优化、显存和恢复检查；不计入80轮主训练，也不构成性能实验。MOFO每seed另有80轮仅训练集先验阶段；UniverSeg支持集计算另计。', '',
        '| 方法 | 选点 | 样本集合 | 种子 | DSC (%) | HD95 (pixel) | ASSD (pixel) |',
        '|---|---|---|---:|---:|---:|---:|']
    for group in groups:
        cells = [group['method'], group['selection'], group['cohort'], str(group['seeds'])]
        cells += [render(group['metrics'][metric]) for metric in FIELDS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '逐域、腹部来源、有效样本数与惩罚数量见 [per_domain_metrics.csv](per_domain_metrics.csv)；'
              '机器可读三种子统计见 [metrics.csv](metrics.csv)，完整结果及覆盖信息见 [summary.json](summary.json)。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--action', choices=('train', 'evaluate', 'verify', 'summarize', 'preflight', 'mofo-prior'), required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--require-test', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol, records = verify_protocol(args.output)
    if args.action == 'summarize':
        summarize(args.output, protocol)
    elif args.method is None:
        parser.error('--method is required')
    elif args.action == 'train':
        train(args.output, protocol, records, args.method, args.seed, args.device, args.stop_after_epoch)
    elif args.action == 'evaluate':
        evaluate_run(args.output, protocol, records, args.method, args.seed, args.device)
    elif args.action == 'verify':
        verify_run(args.output, protocol, args.method, args.seed, args.require_test)
    elif args.action == 'preflight':
        from .external_baselines.preflight import run_preflight
        run_preflight(args.output, protocol, records, args.method, args.device)
    elif args.action == 'mofo-prior':
        from .external_baselines.preflight import train_prior
        train_prior(args.output, protocol, args.seed, args.device)


if __name__ == '__main__':
    main()
