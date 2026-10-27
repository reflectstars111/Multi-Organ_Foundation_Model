"""Real-input admission checks and MOFO's separate, train-only prior stage."""
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .protocol import file_sha, read_json, write_json
from ..fair_suite import _restore_rng, _rng_state, _save_checkpoint, _seed, _synchronize
from ..region import DOMAINS


def run_preflight(root, protocol, records, method, device):
    from ..external_baseline_suite import (_bank, _datasets, batch_forward, make_model,
                                          optimization_step, timestamp)
    root = Path(root)
    if torch.device(device).type != 'cuda':
        raise ValueError('Queue admission requires actual CUDA memory/optimizer preflight')
    config = protocol['config']
    _seed(42)
    datasets = _datasets(records, config['size'])
    train = datasets['train']
    bank = _bank(root, records, method, config['size'])
    # Cover all domains in the two effective batches, then use real frozen draws.
    representatives = [next(i for i, g in enumerate(train.groups) if g[0]['dataset'] == name)
                       for name in DOMAINS]
    frozen = read_json(root / 'sampling_plan.json')['42'][0]['indices']
    indices = (representatives + frozen)[:2 * config['batch_size']]
    model = make_model(root, protocol, method, 42).to(device)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    rows = []
    parameter = next(p for p in model.parameters() if p.requires_grad)
    before = parameter.detach().clone()
    micro = config['micro_batch_sizes'][method]
    for step in range(2):
        batch = indices[step * config['batch_size']:(step + 1) * config['batch_size']]
        loss, supports, support_hash, norm = optimization_step(
            model, method, optimizer, train, batch, device, micro, bank,
            seed=42, epoch=1, offset=step * config['batch_size'])
        rows.append(dict(loss=loss, support_draws=supports, support_sha256=support_hash, gradient_norm=norm))
    if torch.equal(before, parameter):
        raise ValueError('Real optimization did not update the first trainable parameter')
    del before
    # Serialize and strictly reload the full optimizer/model/RNG, then verify
    # inference numerically using the same fixed supports and actual image.
    model.eval()
    with torch.no_grad():
        expected = batch_forward(model, method, train, indices[:1], device, bank)[0]['logits'].cpu()
    probe_path = root / 'preflight' / f'{method}_resume.pt'
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(probe_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=_rng_state()))
    saved_rng = _rng_state()
    del model, optimizer, parameter
    gc.collect()
    torch.cuda.empty_cache()
    model = make_model(root, protocol, method, 42).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    cp = torch.load(probe_path, map_location=device, weights_only=True)
    model.load_state_dict(cp['model'], strict=True)
    optimizer.load_state_dict(cp['optimizer'])
    _restore_rng(cp['rng'])
    if not torch.equal(torch.get_rng_state(), saved_rng['torch']):
        raise ValueError('RNG restoration mismatch')
    del cp
    model.eval()
    with torch.no_grad():
        actual = batch_forward(model, method, train, indices[:1], device, bank)[0]['logits'].cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _synchronize(device)
    peak = int(torch.cuda.max_memory_reserved(device) / 1024 ** 2)
    allocated = int(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
    required = max(8192, peak + 4096)
    total = int(torch.cuda.get_device_properties(device).total_memory / 1024 ** 2)
    if required > total:
        raise ValueError(f'Preflight needs {required} MiB including safety margin, above GPU capacity {total} MiB')
    report = dict(method=method, passed=True, protocol_sha256=protocol['protocol_sha256'],
        size=config['size'], dtype='float32', physical_batch=micro, effective_batch=config['batch_size'],
        real_optimizer_updates=2, real_query_images=len(indices), domains=DOMAINS,
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_reserved_mib=peak, peak_allocated_mib=allocated,
        required_free_mib=required, device_total_mib=total,
        strict_model_optimizer_rng_reload=True, identical_reload_prediction=True,
        steps=rows, elapsed_seconds=time.monotonic() - started, updated_at=timestamp())
    write_json(root / 'preflight' / f'{method}.json', report)
    probe_path.unlink()
    print(report, flush=True)


def train_prior(root, protocol, seed, device):
    root = Path(root)
    folder = root / 'mofo_prior' / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    config = protocol['config']['mofo_prior']
    command = [sys.executable, '-m', 'unet_moe.external_baselines.mofo_prior',
        '--records', str(root / 'records.json'), '--output', str(folder),
        '--pool-manifest', str(root / 'support_manifest.json'),
        '--seed', str(seed), '--device', device, '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']), '--micro-batch-size', str(config['micro_batch_size']),
        '--size', str(protocol['config']['size']),
        '--quota-per-domain', str(protocol['config']['quota_per_domain']),
        '--lr', str(protocol['config']['lr']), '--weight-decay', str(protocol['config']['weight_decay'])]
    if (folder / 'last.pt').exists():
        command += ['--resume', str(folder / 'last.pt')]
    inherited = tuple(int(fd) for fd in os.environ.get('Q04_LOCK_FDS', '').split(',') if fd)
    subprocess.run(command, check=True, pass_fds=inherited)
    cp = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
    expected = dict(epochs=config['epochs'], seed=seed, size=protocol['config']['size'],
                    records_sha256=file_sha(root / 'records.json'),
                    support_manifest_sha256=file_sha(root / 'support_manifest.json'))
    if (cp.get('complete') is not True or cp.get('epoch') != config['epochs']
            or any(cp.get('protocol', {}).get(key) != value for key, value in expected.items())):
        raise ValueError('MOFO prior exited without a complete prior checkpoint')
    write_json(folder / 'prior_complete.json', dict(complete=True, seed=seed,
        epochs_completed=config['epochs'], protocol_sha256=protocol['protocol_sha256'],
        checkpoint_sha256=file_sha(folder / 'last.pt')))
