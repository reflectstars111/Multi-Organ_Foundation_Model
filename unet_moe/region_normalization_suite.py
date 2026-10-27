"""Frozen four-arm Region MoE normalization study, independent of Q01 training.

Every arm trains from scratch on the exact Q01 records and image draws. Domain
IDs come exclusively from image metadata, including during inference. Tests are
gated on all twelve training/validation runs, with no checkpoint substitution.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from . import fair_suite as fair
from . import organ_diagnostics as diagnostics
from .normalization import NORMALIZATION_TYPES, normalization_diagnostics, validate_domain_ids
from .organ_weighted_suite import (file_sha, read_json, state_digest, make_draw_plan,
                                   metric_scalars, moments)
from .region import RegionUNet, DOMAINS, PAPER_ALIGNED_TASK_IDS
from .single_channel_suite import UnionDataset
from .union_evaluate import union_prediction, score_image, summarize_run


VERSION = 'region_domain_normalization_v1'
ARMS = tuple(NORMALIZATION_TYPES)
GUARDRAIL_CONFIG = diagnostics.GUARDRAIL_CONFIG
DEFAULT_REFERENCE = Path('/ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007')
CONFIG = dict(size=512, base=24, epochs=80, batch_size=16, micro_batch_size=4,
              quota_per_domain=500, workers=0, lr=1e-4, weight_decay=1e-4,
              experts=9, top_k=1, seeds=[42, 43, 44], models=['region'], arms=list(ARMS),
              task_ids=list(PAPER_ALIGNED_TASK_IDS), dtype='float32', gradient_clip_norm=5.,
              lr_schedule='constant', augmentation='none', auxiliary_loss_weights=0.,
              threshold=.5, optimizer='AdamW', optimizer_betas=[.9, .999], optimizer_eps=1e-8,
              primary_selection='best_validation_macro_positive_dice',
              split_seed=42, split_policy='trainmore_v6', abdomen_split='ct7_2',
              loss='image_mean', norm_eps=1e-5, norm_momentum=.1,
              domain_names=list(DOMAINS), normalization_scope='dense_shared_paths',
              dense_normalization_sites=26, expert_normalization='shared_gn',
              batchnorm_statistics='physical_microbatch_domain_subset; no virtual accumulated BN')


class DomainDataset(torch.utils.data.Dataset):
    """One image, its available structure labels, and its explicit source domain."""
    def __init__(self, records, size=512, task_ids=None):
        self.task_ids = list(PAPER_ALIGNED_TASK_IDS if task_ids is None else task_ids)
        self.region = fair.RegionDataset(records, size, self.task_ids)
        self.groups = self.region.groups
        self.domain_ids = []
        for rows in self.groups:
            domains = {row.get('dataset') for row in rows}
            if len(domains) != 1 or not domains.issubset(DOMAINS):
                raise ValueError('Every image must have exactly one known metadata domain')
            self.domain_ids.append(DOMAINS.index(next(iter(domains))))

    def __len__(self):
        return len(self.region)

    def __getitem__(self, index):
        image, target = self.region[index]
        return image, target, torch.tensor(self.domain_ids[index], dtype=torch.long)


def create_model(config, arm):
    if arm not in ARMS:
        raise ValueError('Unknown normalization arm')
    return RegionUNet(base=config['base'], top_k=config['top_k'], task_ids=config['task_ids'],
                      normalization=arm, norm_eps=config['norm_eps'], norm_momentum=config['norm_momentum'])


def non_norm_digest(model):
    """Fingerprint every convolution, router and other non-normalization parameter."""
    prefixes = tuple(name + '.' for name, module in model.named_modules()
                     if isinstance(module, (torch.nn.GroupNorm, torch.nn.modules.batchnorm._BatchNorm)))
    return state_digest({name: value for name, value in model.named_parameters()
                         if not name.startswith(prefixes)})


def identities(protocol):
    c = protocol['config']
    return [('region', seed, arm) for seed in c['seeds'] for arm in c['arms']]


def run_folder(root, kind, seed, arm):
    if kind != 'region' or arm not in ARMS:
        raise ValueError('Only the declared Region normalization arms are supported')
    return Path(root) / arm / kind / f'seed_{seed}'


def source_manifest(root):
    root = Path(root)
    paths = sorted((root / 'unet_moe').rglob('*.py'))
    paths += [root / 'scripts' / name for name in (
        '__init__.py', 'run_region_normalization.py', 'run_organ_weighted.py', 'preflight_fair80.py',
        'fair_parallel.py')]
    return {str(path.relative_to(root)): file_sha(path) for path in paths}


def _verify_data_content(content):
    for path, recorded in content.items():
        if Path(path).stat().st_size != recorded['bytes'] or file_sha(path) != recorded['sha256']:
            raise ValueError(f'Data content changed: {path}')


def _verify_protocol_hash(protocol, version=None):
    expected = dict(protocol)
    digest = expected.pop('protocol_sha256', None)
    if (not digest or fair._digest(expected) != digest or
            (version is not None and protocol.get('protocol_version') != version)):
        raise ValueError('Frozen protocol hash/version differs')


def _verify_draws(dataset, draws, config):
    expected = make_draw_plan(dataset.groups, config['seeds'], config['epochs'],
                              config['quota_per_domain'], config['batch_size'])
    if draws != expected:
        raise ValueError('Sampling plan differs from the paired image/update budget')


def prepare(args):
    """Copy Q01's frozen input protocol, never its weights or result selections."""
    root, reference = Path(args.output).resolve(), Path(args.reference_root).resolve()
    if root.exists():
        raise FileExistsError('Prepare requires a new experiment directory')
    reference_protocol = read_json(reference / 'protocol.json')
    _verify_protocol_hash(reference_protocol, 'q01_organ_weighted_v1')
    # This reads the immutable Q01 contract only. Q01 need not have finished,
    # and live source edits must not invalidate its already frozen snapshot.
    ref_config = reference_protocol['config']
    fields = ('size', 'base', 'epochs', 'batch_size', 'micro_batch_size', 'quota_per_domain',
              'workers', 'lr', 'weight_decay', 'experts', 'top_k', 'seeds', 'task_ids',
              'dtype', 'gradient_clip_norm', 'augmentation', 'lr_schedule', 'threshold',
              'optimizer', 'optimizer_betas', 'optimizer_eps', 'split_seed', 'split_policy',
              'abdomen_split', 'auxiliary_loss_weights')
    if any(ref_config.get(key) != CONFIG[key] for key in fields):
        raise ValueError('Reference Q01 training/data configuration differs')
    artifact_fields = {'records.json': 'record_sha256', 'sampling_plan.json': 'draws_sha256',
                       'data_content.json': 'content_sha256', 'train_statistics.json': 'statistics_sha256'}
    artifacts = {name: read_json(reference / name) for name in artifact_fields}
    for name, key in artifact_fields.items():
        if fair._digest(artifacts[name]) != reference_protocol[key]:
            raise ValueError(f'Reference {name} fingerprint differs')
    records, draws = artifacts['records.json'], artifacts['sampling_plan.json']
    fair.validate_splits(records)
    if {r['task'] for r in records} != set(PAPER_ALIGNED_TASK_IDS):
        raise ValueError('Reference must contain exactly the sixteen paper-aligned structures')
    if {r[key] for r in records for key in ('image', 'mask')} != set(artifacts['data_content.json']):
        raise ValueError('Reference data content manifest differs from records')
    _verify_data_content(artifacts['data_content.json'])
    dataset = DomainDataset([r for r in records if r['split'] == 'train'], CONFIG['size'], CONFIG['task_ids'])
    if set(dataset.domain_ids) != set(range(len(DOMAINS))):
        raise ValueError('All nine training domains are required')
    _verify_draws(dataset, draws, CONFIG)
    initial, non_norm = {arm: {} for arm in ARMS}, {}
    for seed in CONFIG['seeds']:
        for arm in ARMS:
            fair._seed(seed)
            model = create_model(CONFIG, arm)
            initial[arm][str(seed)] = state_digest(model.state_dict())
            common = non_norm_digest(model)
            if seed in non_norm and non_norm[seed] != common:
                raise ValueError('Non-normalization initialization differs between arms')
            non_norm[seed] = common
            if arm == 'shared_gn' and initial[arm][str(seed)] != reference_protocol['initialization_sha256']['region'][str(seed)]:
                raise ValueError('Shared-GN initialization no longer reproduces the Q01 Region baseline')
            del model
    source_root = Path(__file__).resolve().parents[1]
    sources = source_manifest(source_root)
    config = {**CONFIG, 'data_root': ref_config['data_root']}
    protocol = dict(protocol_version=VERSION, config=config, config_sha256=fair._digest(config),
                    reference_root=str(reference), reference_protocol_sha256=reference_protocol['protocol_sha256'],
                    source_files=sources, source_sha256=fair._digest(sources),
                    initialization_sha256=initial,
                    non_norm_initialization_sha256={str(seed): digest for seed, digest in non_norm.items()},
                    guardrail=dict(GUARDRAIL_CONFIG),
                    normalization=dict(scope='26 dense norm sites: encoder 10, fusion 8, MoE shared 8',
                                       experts='all sparse expert blocks retain legacy GroupNorm',
                                       domain_ids='records[image].dataset; never inferred from targets',
                                       inference='known source domain required for domain-specific arms',
                                       batchnorm='running statistics from physical subbatches only',
                                       momentum=.1, eps=1e-5),
                    budget=dict(images_per_epoch=4500, optimizer_steps_per_epoch=282,
                                epochs=80, runs=12, images_per_run=360000),
                    loss_policy='image_mean for every arm; reference analytic weights are provenance only',
                    initialization_policy='from scratch; identical non-normalization parameters within each seed',
                    test_policy='all 12 training/validation runs complete before any test; historically inspected test',
                    selection_policy='structure validation macro selects best; union is downstream evaluation',
                    environment=dict(python=sys.version, torch=str(torch.__version__),
                                     numpy=np.__version__, cuda=torch.version.cuda))
    for name, key in artifact_fields.items():
        protocol[key] = fair._digest(artifacts[name])
    protocol['protocol_sha256'] = fair._digest(protocol)
    root.mkdir(parents=True)
    for name, artifact in artifacts.items():
        fair._write_json(root / name, artifact)
    fair._write_json(root / 'protocol.json', protocol)
    for relative, digest in sources.items():
        destination = root / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
        if file_sha(destination) != digest:
            raise ValueError('Source changed while freezing snapshot')
    print(f'Prepared {root}: {protocol["protocol_sha256"]}', flush=True)
    return protocol


def load_protocol(root, check_content=True):
    root = Path(root)
    protocol = read_json(root / 'protocol.json')
    _verify_protocol_hash(protocol, VERSION)
    if (any(protocol['config'].get(key) != value for key, value in CONFIG.items()) or
            fair._digest(protocol['config']) != protocol['config_sha256']):
        raise ValueError('Frozen normalization configuration differs')
    sources = source_manifest(Path(__file__).resolve().parents[1])
    if sources != protocol['source_files'] or fair._digest(sources) != protocol['source_sha256']:
        raise ValueError('Source changed; run the frozen source_snapshot entry point')
    artifacts = {}
    for name, key in (('records.json', 'record_sha256'), ('sampling_plan.json', 'draws_sha256'),
                      ('data_content.json', 'content_sha256'), ('train_statistics.json', 'statistics_sha256')):
        artifacts[name] = read_json(root / name)
        if fair._digest(artifacts[name]) != protocol[key]:
            raise ValueError(f'{name} differs from frozen protocol')
    if check_content:
        _verify_data_content(artifacts['data_content.json'])
    return protocol, artifacts['records.json'], artifacts['sampling_plan.json'], artifacts['train_statistics.json']


def loader(dataset, config, seed, sampler=None):
    return DataLoader(dataset, batch_size=config['micro_batch_size'], sampler=sampler,
                      shuffle=False, num_workers=config['workers'],
                      generator=torch.Generator().manual_seed(seed))


def train_effective_batches(model, batches, optimizer, args, image_total):
    count = steps = queries = window_count = micro_batches = 0
    total_loss = 0.
    domain_images = Counter({domain: 0 for domain in DOMAINS})
    subbatches = {domain: Counter() for domain in DOMAINS}
    for image, target, domain_ids in batches:
        image, target, domain_ids = image.to(args.device), target.to(args.device), domain_ids.to(args.device)
        validate_domain_ids(domain_ids, image, len(DOMAINS))
        if not target.ge(0).any((1, 2, 3)).all():
            raise ValueError('Training image without valid annotation')
        if window_count == 0:
            optimizer.zero_grad(set_to_none=True)
            window_size = min(args.batch_size, image_total - count)
        if window_size <= 0 or len(image) > window_size - window_count:
            raise ValueError('Microbatch crosses effective-batch boundary')
        output = model(image, domain_ids=domain_ids)
        loss = fair.common_loss(output['logits'], target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite image-mean training loss')
        (loss * (len(image) / window_size)).backward()
        total_loss += float(loss.detach()) * len(image)
        count += len(image)
        queries += int(target.ge(0).any((2, 3)).sum())
        micro_batches += 1
        for domain, number in Counter(domain_ids.cpu().tolist()).items():
            domain_images[DOMAINS[domain]] += number
            subbatches[DOMAINS[domain]][str(number)] += 1
        window_count += len(image)
        if window_count == window_size:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            window_count = 0
        del image, target, domain_ids, output, loss
    if count != image_total or window_count or steps != math.ceil(image_total / args.batch_size):
        raise ValueError('Training did not consume the exact paired image budget')
    return dict(train_loss=total_loss / count, sample_count=count, annotated_channel_queries=queries,
                optimizer_steps=steps, physical_micro_batches=micro_batches,
                domain_image_counts=dict(domain_images),
                domain_subbatch_sizes={key: dict(value) for key, value in subbatches.items()})


def _update_routing(totals, output, top_k):
    """Aggregate every image/proposal, without label-conditioned route selection."""
    routing = output.get('routing', [])
    if not totals:
        totals.extend(dict(proposals=0, images=0, entropy=0., confidence=0.,
                           probability=torch.zeros(9, dtype=torch.float64),
                           top1=torch.zeros(9, dtype=torch.int64),
                           topk=torch.zeros(9, dtype=torch.int64),
                           dispatched=torch.zeros(9, dtype=torch.int64)) for _ in routing)
    if len(totals) != len(routing):
        raise ValueError('Routing stages changed during evaluation')
    for row, scores in zip(totals, routing):
        probability = scores.detach().float().softmax(-1)
        selected = probability.topk(top_k, dim=-1).indices
        row['proposals'] += probability.shape[0] * probability.shape[1]
        row['images'] += probability.shape[0]
        row['entropy'] += float(-(probability * probability.clamp_min(1e-12).log()).sum())
        row['confidence'] += float(probability.max(-1).values.sum())
        row['probability'] += probability.double().sum((0, 1)).cpu()
        row['top1'] += torch.bincount(selected[..., 0].flatten(), minlength=9).cpu()
        row['topk'] += torch.bincount(selected.flatten(), minlength=9).cpu()
        for expert in range(9):
            row['dispatched'][expert] += int((selected == expert).any((1, 2)).sum())


def _finalize_routing(totals):
    return [dict(stage=index, proposals=row['proposals'], images=row['images'],
                 mean_entropy=row['entropy'] / row['proposals'],
                 mean_top1_probability=row['confidence'] / row['proposals'],
                 mean_expert_probability=(row['probability'] / row['proposals']).tolist(),
                 top1_expert_counts=row['top1'].tolist(), topk_expert_counts=row['topk'].tolist(),
                 dispatched_image_counts=row['dispatched'].tolist(),
                 experts_used=int(row['topk'].gt(0).sum()))
            for index, row in enumerate(totals)]


@torch.no_grad()
def evaluate(model, batches, device, groups=None, task_ids=None):
    """Structure/CT diagnostics with metadata IDs, preserving all model buffers."""
    task_ids = list(PAPER_ALIGNED_TASK_IDS if task_ids is None else task_ids)
    if groups is not None and isinstance(batches, DataLoader):
        if not isinstance(batches.sampler, SequentialSampler) or batches.drop_last:
            raise ValueError('groups require a sequential complete DataLoader')
        if len(groups) != len(batches.dataset):
            raise ValueError('groups length must match dataset')
    totals = {diagnostics.TASKS[task]: diagnostics._empty_totals() for task in task_ids}
    sources_totals, offset, routing_totals = {}, 0, []
    with diagnostics._read_only_model(model):
        buffers_before = state_digest(dict(model.named_buffers()))
        for image, target, domain_ids in batches:
            image, target, domain_ids = image.to(device), target.to(device), domain_ids.to(device)
            validate_domain_ids(domain_ids, image, len(DOMAINS))
            if target.shape[1] != len(task_ids):
                raise ValueError('Target channels do not match task_ids')
            output = model(image, domain_ids=domain_ids)
            logits = output['logits']
            top_k = model.moes[0].top_k if hasattr(model, 'moes') else 1
            _update_routing(routing_totals, output, top_k)
            if logits.shape != target.shape or not torch.isfinite(logits).all():
                raise ValueError('Invalid evaluation logits')
            valid, truth = target.ge(0), target.gt(0)
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
                current_groups = groups[offset:offset + len(image)]
                expected_domains = [DOMAINS.index(rows[0]['dataset']) for rows in current_groups]
                if domain_ids.cpu().tolist() != expected_domains:
                    raise ValueError('Evaluation domain IDs differ from image metadata order')
                sources = [diagnostics._source(rows) for rows in current_groups]
                for source in sources:
                    if source is not None:
                        sources_totals.setdefault(source, {name: diagnostics._empty_totals()
                                                          for name in diagnostics.ABDOMEN_STRUCTURES})
            for batch, channel in valid.any((2, 3)).cpu().nonzero().tolist():
                name = diagnostics.TASKS[task_ids[channel]]
                diagnostics._accumulate(totals[name], values[batch][channel])
                if groups is not None and name in diagnostics.ABDOMEN_STRUCTURES and sources[batch] is None:
                    raise ValueError('groups must identify the CT source of every abdomen target')
                if name in diagnostics.ABDOMEN_STRUCTURES and sources[batch] is not None:
                    diagnostics._accumulate(sources_totals[sources[batch]][name], values[batch][channel])
            offset += len(image)
        if state_digest(dict(model.named_buffers())) != buffers_before:
            raise RuntimeError('Evaluation changed normalization buffers')
    if groups is not None and offset != len(groups):
        raise ValueError('groups longer than loader image count')
    result = diagnostics._finalize(totals)
    result['abdomen_by_source'] = {source: diagnostics._finalize(rows)
                                   for source, rows in sorted(sources_totals.items())}
    result['normalization_buffers_unchanged'] = True
    result['routing'] = _finalize_routing(routing_totals)
    return result


@torch.no_grad()
def evaluate_union(model, dataset, device, config, seed):
    """Use the exact established union targets and scoring, with metadata IDs."""
    batches = loader(dataset, config, seed)
    rows, offset = [], 0
    with diagnostics._read_only_model(model):
        buffers_before = state_digest(dict(model.named_buffers()))
        for image, target, domains, sources, multi_target in batches:
            image, multi_target = image.to(device), multi_target.to(device)
            if any(domain not in DOMAINS for domain in domains):
                raise ValueError('Unknown union evaluation metadata domain')
            domain_ids = torch.tensor([DOMAINS.index(domain) for domain in domains], device=device)
            output = model(image, domain_ids=domain_ids)
            if not torch.isfinite(output['logits']).all():
                raise ValueError('Nonfinite union evaluation logits')
            predictions = union_prediction(output['logits'], multi_target).cpu()
            for index in range(len(image)):
                row = score_image(predictions[index, 0], target[index, 0])
                row.update(domain=domains[index], source=sources[index],
                           image=dataset.groups[offset + index][0]['image'])
                rows.append(row)
            offset += len(image)
        if state_digest(dict(model.named_buffers())) != buffers_before:
            raise RuntimeError('Union evaluation changed normalization buffers')
    if offset != len(dataset):
        raise ValueError('Union evaluation did not consume the complete dataset')
    return dict(metrics=summarize_run(rows), per_image=rows, normalization_buffers_unchanged=True)


def _identity(protocol, seed, arm):
    return dict(protocol_sha256=protocol['protocol_sha256'], kind='region', seed=seed, arm=arm,
                initialization_sha256=protocol['initialization_sha256'][arm][str(seed)],
                non_norm_initialization_sha256=protocol['non_norm_initialization_sha256'][str(seed)])


def check_identity(artifact, protocol, kind, seed, arm):
    if kind != 'region' or any(artifact.get(key) != value
                               for key, value in _identity(protocol, seed, arm).items()):
        raise ValueError('Run identity, normalization initialization, or frozen protocol differs')


def check_history(history, protocol, draws, kind, seed, arm):
    if (len(history) > len(draws[str(seed)]) or
            [row['epoch'] for row in history] != list(range(1, len(history) + 1))):
        raise ValueError('History is not a contiguous paired epoch sequence')
    for row, draw in zip(history, draws[str(seed)]):
        check_identity(row, protocol, kind, seed, arm)
        if (row['sample_sha256'] != draw['sample_sha256'] or row['sample_count'] != len(draw['indices'])
                or row['optimizer_steps'] != len(draw['effective_batch_sizes'])
                or not math.isfinite(row['macro_positive_dice'])
                or sum(row['domain_image_counts'].values()) != row['sample_count']
                or not row['normalization_buffers_unchanged']):
            raise ValueError('History differs from paired draw/update budget or read-only evaluation')
        if set(row['domain_image_counts']) != set(DOMAINS) or set(row['domain_subbatch_sizes']) != set(DOMAINS):
            raise ValueError('History domain exposure keys differ')
        for domain in DOMAINS:
            counts = row['domain_subbatch_sizes'][domain]
            if (any(not str(size).isdigit() or int(size) < 1 or
                    int(size) > protocol['config']['micro_batch_size'] or type(number) is not int or number < 1
                    for size, number in counts.items()) or
                    sum(int(size) * number for size, number in counts.items()) != row['domain_image_counts'][domain]):
                raise ValueError('History domain microbatch exposure differs')


def _empty_exposure():
    return dict(physical_micro_batches=0, domain_image_counts={domain: 0 for domain in DOMAINS},
                domain_subbatch_sizes={domain: {} for domain in DOMAINS})


def _accumulate_exposure(exposure, training):
    result = _empty_exposure()
    result['physical_micro_batches'] = exposure['physical_micro_batches'] + training['physical_micro_batches']
    for domain in DOMAINS:
        result['domain_image_counts'][domain] = exposure['domain_image_counts'][domain] + training['domain_image_counts'][domain]
        result['domain_subbatch_sizes'][domain] = dict(Counter(exposure['domain_subbatch_sizes'][domain])
                                                      + Counter(training['domain_subbatch_sizes'][domain]))
    return result


def _normalization_report(model, exposure):
    report = normalization_diagnostics(model)
    for layer in report['layers']:
        domain_index = layer['domain_index']
        updates = (exposure['physical_micro_batches'] if domain_index is None else
                   sum(exposure['domain_subbatch_sizes'][DOMAINS[domain_index]].values()))
        samples = (sum(exposure['domain_image_counts'].values()) if domain_index is None else
                   exposure['domain_image_counts'][DOMAINS[domain_index]])
        layer.update(training_forward_batches=updates, training_image_exposure=samples)
        if 'num_batches_tracked' in layer and layer['num_batches_tracked'] != updates:
            raise ValueError('BN running-statistics updates differ from metadata subbatch exposure')
    return report


def run_model(args, protocol, records, draws, statistics=None):
    c, seed, arm = protocol['config'], args.seed, args.arm
    if (getattr(args, 'kind', 'region'), seed, arm) not in identities(protocol):
        raise ValueError('Run is outside the frozen normalization matrix')
    folder = run_folder(args.output, 'region', seed, arm)
    if folder.exists() and not args.resume:
        raise FileExistsError('Existing run requires --resume')
    folder.mkdir(parents=True, exist_ok=True)
    fair._seed(seed)
    model = create_model(c, arm).to(args.device)
    identity = {**_identity(protocol, seed, arm), 'initialization_sha256': state_digest(model.state_dict()),
                'non_norm_initialization_sha256': non_norm_digest(model)}
    check_identity(identity, protocol, 'region', seed, arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'],
                                 betas=tuple(c['optimizer_betas']), eps=c['optimizer_eps'])
    history, best_score, best_epoch, guard_score, guard_epoch, elapsed = [], -math.inf, None, -math.inf, None, 0.
    exposure = _empty_exposure()
    if (folder / 'last.pt').exists():
        checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        check_identity(checkpoint, protocol, 'region', seed, arm)
        history = checkpoint['history']
        if checkpoint['epoch'] != len(history):
            raise ValueError('Checkpoint history is not contiguous')
        check_history(history, protocol, draws, 'region', seed, arm)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        exposure = checkpoint['normalization_exposure']
        if _normalization_report(model, exposure) != checkpoint['normalization_diagnostics']:
            raise ValueError('Checkpoint normalization buffers/diagnostics differ')
        best_score, best_epoch = checkpoint['best_score'], checkpoint['best_epoch']
        guard_score, guard_epoch = checkpoint['guardrail_score'], checkpoint['guardrail_epoch']
        elapsed = checkpoint['elapsed_seconds']
        for name, epoch in (('best', best_epoch), ('guardrail', guard_epoch)):
            if epoch == checkpoint['epoch']:
                fair._save_checkpoint(folder / f'{name}.pt', checkpoint)
            elif epoch is not None and not (folder / f'{name}.pt').is_file():
                raise FileNotFoundError(f'Missing selected {name} checkpoint')
        fair._restore_rng(checkpoint['rng'])
        fair._write_history(folder / 'metrics.jsonl', history)
        del checkpoint
    elif any((folder / name).exists() for name in ('best.pt', 'guardrail.pt', 'metrics.jsonl', 'run_summary.json')):
        raise ValueError('Run artifacts exist without last checkpoint')
    datasets = {split: DomainDataset([r for r in records if r['split'] == split], c['size'], c['task_ids'])
                for split in ('train', 'val')}
    batches_val = loader(datasets['val'], c, seed)
    training_args = argparse.Namespace(device=args.device, batch_size=c['batch_size'])
    final_epoch = min(c['epochs'], args.stop_after_epoch or c['epochs'])
    if final_epoch < len(history):
        raise ValueError('Requested stop epoch precedes the restored checkpoint')
    for epoch in range(len(history) + 1, final_epoch + 1):
        draw = draws[str(seed)][epoch - 1]
        actual_hash = fair._digest([datasets['train'].groups[index][0]['image'] for index in draw['indices']])
        if actual_hash != draw['sample_sha256']:
            raise ValueError('Draw no longer refers to the same training images')
        batches = loader(datasets['train'], c, seed + epoch, draw['indices'])
        model.train()
        fair._synchronize(args.device)
        started = time.perf_counter()
        training = train_effective_batches(model, batches, optimizer, training_args, len(draw['indices']))
        fair._synchronize(args.device)
        train_seconds = time.perf_counter() - started
        exposure = _accumulate_exposure(exposure, training)
        norm_report = _normalization_report(model, exposure)
        started = time.perf_counter()
        metrics = evaluate(model, batches_val, args.device, groups=datasets['val'].groups, task_ids=c['task_ids'])
        fair._synchronize(args.device)
        validation_seconds = time.perf_counter() - started
        score = metrics['macro_positive_dice']
        if score is None or not math.isfinite(score):
            raise ValueError('Validation score is not finite')
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        passed = diagnostics.guardrail_passes(metrics, protocol['guardrail'])
        guard_improved = passed and score > guard_score
        if guard_improved:
            guard_score, guard_epoch = score, epoch
        elapsed += train_seconds + validation_seconds
        metrics.update(**identity, **training, epoch=epoch, lr=c['lr'], sample_sha256=actual_hash,
                       train_seconds=train_seconds, validation_seconds=validation_seconds,
                       elapsed_seconds=elapsed, guardrail_passed=passed,
                       normalization_diagnostics=norm_report,
                       normalization_buffer_sha256=state_digest(dict(model.named_buffers())))
        history.append(metrics)
        checkpoint = dict(**identity, protocol_version=VERSION, config=c, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), rng=fair._rng_state(), epoch=epoch,
                          history=history, metrics=metrics, best_score=best_score, best_epoch=best_epoch,
                          guardrail_score=guard_score, guardrail_epoch=guard_epoch, elapsed_seconds=elapsed,
                          normalization_exposure=exposure, normalization_diagnostics=norm_report,
                          buffer_names=list(dict(model.named_buffers())))
        fair._save_checkpoint(folder / 'last.pt', checkpoint)
        if improved:
            fair._save_checkpoint(folder / 'best.pt', checkpoint)
        if guard_improved:
            fair._save_checkpoint(folder / 'guardrail.pt', checkpoint)
        fair._write_history(folder / 'metrics.jsonl', history)
        result = dict(**identity, epochs_completed=epoch, best_epoch=best_epoch,
                      guardrail_epoch=guard_epoch, guardrail_selected=guard_epoch is not None,
                      best_validation_macro_positive_dice=best_score, elapsed_seconds=elapsed,
                      test_evaluated=False, parameter_count=sum(p.numel() for p in model.parameters()),
                      normalization_exposure=exposure)
        fair._write_json(folder / 'run_summary.json', result)
        print(f'{arm} seed={seed} epoch={epoch}/{c["epochs"]} validation={score:.6f} '
              f'abdomen={metrics["abdomen_macro_positive_dice"]}', flush=True)
    if not history:
        raise ValueError('No training epoch was completed')
    result = dict(**identity, epochs_completed=len(history), best_epoch=best_epoch,
                  guardrail_epoch=guard_epoch, guardrail_selected=guard_epoch is not None,
                  best_validation_macro_positive_dice=best_score, elapsed_seconds=elapsed,
                  test_evaluated=False, parameter_count=sum(p.numel() for p in model.parameters()),
                  normalization_exposure=exposure)
    existing = read_json(folder / 'run_summary.json') if (folder / 'run_summary.json').exists() else {}
    if existing.get('test_evaluated') and existing.get('epochs_completed') == len(history):
        check_identity(existing, protocol, 'region', seed, arm)
        result = existing
    fair._write_json(folder / 'run_summary.json', result)
    return result


def validate_run(root, protocol, draws, kind, seed, arm, require_test=False):
    folder = run_folder(root, kind, seed, arm)
    result = read_json(folder / 'run_summary.json')
    check_identity(result, protocol, kind, seed, arm)
    epochs = protocol['config']['epochs']
    if result['epochs_completed'] != epochs:
        raise ValueError('Run training is incomplete')
    import json
    history = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    if len(history) != epochs:
        raise ValueError('Incomplete epoch history')
    check_history(history, protocol, draws, kind, seed, arm)
    best = max(history, key=lambda row: row['macro_positive_dice'])['epoch']
    guarded = [row for row in history if diagnostics.guardrail_passes(row, protocol['guardrail'])]
    guard = max(guarded, key=lambda row: row['macro_positive_dice'])['epoch'] if guarded else None
    if result['best_epoch'] != best or result['guardrail_epoch'] != guard:
        raise ValueError('Checkpoint selection violates frozen validation rules')
    selected = {'best': best, 'last': epochs, 'guardrail': guard}
    exposure = _empty_exposure()
    for row in history:
        exposure = _accumulate_exposure(exposure, row)
    if exposure != result['normalization_exposure']:
        raise ValueError('Recorded normalization exposure differs from training history')
    shas = {}
    for name, epoch in selected.items():
        path = folder / f'{name}.pt'
        if epoch is None:
            if path.exists():
                raise ValueError('Unselected guardrail checkpoint unexpectedly exists')
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        check_identity(checkpoint, protocol, kind, seed, arm)
        if checkpoint['epoch'] != epoch or checkpoint['history'] != history[:epoch]:
            raise ValueError(f'{name} checkpoint does not match recorded history')
        buffer_digest = state_digest({key: checkpoint['model'][key] for key in checkpoint['buffer_names']})
        if (buffer_digest != history[epoch - 1]['normalization_buffer_sha256'] or
                checkpoint['normalization_diagnostics'] != history[epoch - 1]['normalization_diagnostics']):
            raise ValueError('Checkpoint normalization buffers or diagnostics differ')
        if name == 'last' and (checkpoint['best_epoch'] != best or checkpoint['guardrail_epoch'] != guard
                               or checkpoint['normalization_exposure'] != exposure):
            raise ValueError('Final checkpoint selections/exposure differ')
        shas[name] = file_sha(path)
        del checkpoint
    if require_test:
        if result.get('test_evaluated') is not True:
            raise ValueError('Test not evaluated')
        for name, epoch in selected.items():
            metric = result.get('test', {}).get(name)
            if epoch is None:
                if metric is not None:
                    raise ValueError('Unselected guardrail has test metrics')
                continue
            if (not metric or metric.get('checkpoint_epoch') != epoch or
                    metric.get('checkpoint_sha256') != shas[name] or
                    metric.get('protocol_sha256') != protocol['protocol_sha256'] or
                    metric != read_json(folder / f'test_{name}.json') or not metric.get('union')):
                raise ValueError('Test result differs from selected checkpoint/protocol or lacks union metrics')
            union = read_json(folder / f'test_union_{name}.json')
            if (any(union.get(key) != metric.get(key) for key in ('checkpoint_epoch', 'checkpoint_sha256',
                                                               'checkpoint_selection', 'protocol_sha256')) or
                    union.get('metrics') != metric['union'] or
                    not union.get('normalization_buffers_unchanged') or
                    summarize_run(union.get('per_image', [])) != metric['union']):
                raise ValueError('Union test result differs from its checkpoint or per-image scores')
    return result


def evaluate_test(args, protocol, records, draws):
    # Authoritative barrier precedes construction/access of either test dataset.
    for kind, seed, arm in identities(protocol):
        validate_run(args.output, protocol, draws, kind, seed, arm)
    if ('region', args.seed, args.arm) not in identities(protocol):
        raise ValueError('Test run is outside the frozen matrix')
    folder = run_folder(args.output, 'region', args.seed, args.arm)
    result = read_json(folder / 'run_summary.json')
    c = protocol['config']
    test_records = [row for row in records if row['split'] == 'test']
    dataset = DomainDataset(test_records, c['size'], c['task_ids'])
    union_dataset = UnionDataset(test_records, c['size'], return_multi_target=True)
    batches = loader(dataset, c, args.seed)
    model = create_model(c, args.arm).to(args.device)
    test = {}
    for name in ('best', 'last', 'guardrail'):
        if name == 'guardrail' and result['guardrail_epoch'] is None:
            test[name] = None
            continue
        path = folder / f'{name}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        check_identity(checkpoint, protocol, 'region', args.seed, args.arm)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        del checkpoint
        metric = evaluate(model, batches, args.device, groups=dataset.groups, task_ids=c['task_ids'])
        union = evaluate_union(model, union_dataset, args.device, c, args.seed)
        metadata = dict(checkpoint_epoch=epoch, checkpoint_selection=name,
                        checkpoint_sha256=file_sha(path), protocol_sha256=protocol['protocol_sha256'])
        metric.update(**metadata, union=union['metrics'])
        fair._write_json(folder / f'test_union_{name}.json', {**metadata, **union})
        fair._write_json(folder / f'test_{name}.json', metric)
        test[name] = metric
    result.update(test_evaluated=True, test=test)
    fair._write_json(folder / 'run_summary.json', result)
    return result


def paired_comparison(runs, selection='best'):
    by_arm = {arm: {row['seed']: row.get('test', {}).get(selection) for row in runs if row['arm'] == arm}
              for arm in ARMS}
    result = {}
    for right, left in (('domain_bn', 'shared_bn'), ('domain_gn', 'shared_gn'), ('shared_bn', 'shared_gn'),
                        ('domain_bn', 'shared_gn'), ('domain_bn', 'domain_gn')):
        pairs = []
        for seed in sorted(set(by_arm[left]) & set(by_arm[right])):
            if by_arm[left][seed] is not None and by_arm[right][seed] is not None:
                a, b = metric_scalars(by_arm[left][seed]), metric_scalars(by_arm[right][seed])
                pairs.append((seed, {key: b[key] - a[key] for key in a.keys() & b.keys()}))
        fields = sorted(set().union(*(row.keys() for _, row in pairs)))
        result[f'{right}_minus_{left}'] = {
            key: {**moments([row[key] for _, row in pairs if key in row]),
                  'pairs': [dict(seed=seed, delta=row[key]) for seed, row in pairs if key in row]}
            for key in fields}
    return result


def summarize(root, protocol, draws):
    runs = [validate_run(root, protocol, draws, kind, seed, arm, True)
            for kind, seed, arm in identities(protocol)]
    aggregate = {}
    for arm in protocol['config']['arms']:
        aggregate[arm] = {}
        for name in ('best', 'last', 'guardrail'):
            scalars = [metric_scalars(row['test'][name]) for row in runs
                       if row['arm'] == arm and row['test'][name] is not None]
            fields = sorted(set().union(*(row.keys() for row in scalars)))
            aggregate[arm][name] = {key: moments([row[key] for row in scalars if key in row]) for key in fields}
    result = dict(protocol_sha256=protocol['protocol_sha256'], complete=True, runs=runs,
                  primary_comparison='domain_bn_minus_shared_bn',
                  test_across_seeds=aggregate,
                  paired_test={name: paired_comparison(runs, name) for name in ('best', 'last', 'guardrail')},
                  interpretation='paired normalization effects, three seeds; fixed image_mean loss; '
                                 'domain arms require source-domain metadata; guardrail absence remains missing')
    fair._write_json(Path(root) / 'summary.json', result)
    return result


def preflight(args, protocol, records, statistics=None):
    """Two real 512 training updates followed by read-only mixed-domain eval."""
    from scripts.preflight_fair80 import worst_batch
    if torch.device(args.device).type != 'cuda':
        raise ValueError('Memory preflight requires a CUDA device')
    c = protocol['config']
    dataset = DomainDataset([row for row in records if row['split'] == 'train'], c['size'], c['task_ids'])
    selection = worst_batch(dataset.groups, seeds=c['seeds'], epochs=c['epochs'],
                            quota=c['quota_per_domain'], batch_size=c['batch_size'],
                            micro_batch_size=c['micro_batch_size'])
    fair._seed(selection['seed'])
    torch.cuda.mem_get_info(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    model = create_model(c, args.arm).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'],
                                 betas=tuple(c['optimizer_betas']), eps=c['optimizer_eps'])
    training_args = argparse.Namespace(batch_size=c['batch_size'], device=args.device)
    exposure = _empty_exposure()
    for _ in range(2):
        model.train()
        training = train_effective_batches(model, loader(dataset, c, 42, selection['indices']), optimizer,
                                            training_args, len(selection['indices']))
        exposure = _accumulate_exposure(exposure, training)
        if training['optimizer_steps'] != 1:
            raise ValueError('Preflight selection must produce exactly one effective update per repetition')
    probe_indices = [dataset.domain_ids.index(domain) for domain in range(len(DOMAINS))]
    probes = torch.utils.data.Subset(dataset, probe_indices)
    metric = evaluate(model, loader(probes, c, 42), args.device, task_ids=c['task_ids'])
    norm_report = _normalization_report(model, exposure)
    torch.cuda.synchronize(args.device)
    peak = math.ceil(torch.cuda.max_memory_reserved(args.device) / 2 ** 20)
    report = dict(kind='region', arm=args.arm, protocol_sha256=protocol['protocol_sha256'],
                  passed=True, optimizer_updates=2, selection=selection,
                  peak_reserved_mib=peak,
                  peak_allocated_mib=math.ceil(torch.cuda.max_memory_allocated(args.device) / 2 ** 20),
                  required_free_mib=peak + 3072, normalization_diagnostics=norm_report,
                  normalization_exposure=exposure, validation=metric)
    fair._write_json(Path(args.output) / 'preflight' / f'{args.arm}.json', report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--verify-run', action='store_true')
    modes.add_argument('--summarize', action='store_true')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--kind', choices=['region'], default='region')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=CONFIG['seeds'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', args.device):
        parser.error('--device must be cpu, cuda or cuda:N')
    if not 0 <= args.stop_after_epoch <= CONFIG['epochs']:
        parser.error('--stop-after-epoch must be within [0, 80]')
    if args.reference_root is not None and not args.prepare:
        parser.error('--reference-root is only valid with --prepare')
    if args.prepare or args.summarize:
        if args.arm is not None or args.seed is not None or args.resume or args.evaluate_test or args.stop_after_epoch:
            parser.error('Preparation/summary cannot select a run or training/test options')
        if args.device != 'cpu':
            parser.error('Preparation/summary is CPU-only')
    elif args.preflight:
        if args.arm is None or args.seed is not None or args.resume or args.evaluate_test or args.stop_after_epoch:
            parser.error('Preflight requires --arm and no seed/resume/test/stop options')
        if not args.device.startswith('cuda'):
            parser.error('Memory preflight requires --device cuda:N')
    else:
        if args.arm is None or args.seed is None:
            parser.error('--arm and --seed are required')
        if (args.verify_run or args.evaluate_test) and (args.resume or args.stop_after_epoch):
            parser.error('Verification/test cannot resume or stop training')
        if args.verify_run and args.device != 'cpu':
            parser.error('Verification is CPU-only')
    if args.prepare and args.reference_root is None:
        args.reference_root = DEFAULT_REFERENCE
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.prepare:
        prepare(args)
        return
    protocol, records, draws, statistics = load_protocol(args.output, check_content=not args.verify_run)
    if args.verify_run:
        result = validate_run(args.output, protocol, draws, 'region', args.seed, args.arm, args.evaluate_test)
        print(f'valid: {args.arm} seed={args.seed} epochs={result["epochs_completed"]}', flush=True)
    elif args.summarize:
        summarize(args.output, protocol, draws)
    elif args.preflight:
        preflight(args, protocol, records, statistics)
    elif args.evaluate_test:
        evaluate_test(args, protocol, records, draws)
    else:
        run_model(args, protocol, records, draws, statistics)


if __name__ == '__main__':
    main()
