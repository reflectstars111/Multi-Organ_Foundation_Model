"""Q07 cost measurements on fixed validation inputs and selected checkpoints.

Inputs reside on the GPU during timing. Single task and all annotated structures
are separate workloads; the latter includes query expansion and union assembly.
Other GPU processes are reported, never silently treated as an isolated device.
"""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch


def benchmark_indices(groups):
    if any(row.get('split') != 'val' for group in groups for row in group):
        raise ValueError('Benchmark inputs must be validation only')
    indices = [i for i, group in enumerate(groups)
               if group and all(row.get('dataset') == 'AbdomenUS' for row in group)]
    if len(indices) < 4:
        raise ValueError('Benchmark requires four fixed validation abdomen images')
    return indices[:4]


def latency_statistics(samples):
    if not samples or any(not math.isfinite(v) or v < 0 for v in samples):
        raise ValueError('Invalid latency samples')
    ordered = sorted(samples)
    return dict(median_ms=statistics.median(ordered),
                p95_ms=ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
                minimum_ms=min(ordered), maximum_ms=max(ordered), samples_ms=list(samples))


def invoked_parameter_count(model, forward):
    seen, handles = set(), []
    parameters = {id(p): p.numel() for p in model.parameters()}
    def record(module, _):
        seen.update(id(p) for p in module.parameters(recurse=False))
    try:
        for module in model.modules():
            handles.append(module.register_forward_pre_hook(record))
        result = forward()
        del result
    finally:
        for handle in handles:
            handle.remove()
    return sum(parameters[p] for p in seen)


def gpu_processes():
    """Best-effort observed processes; absence of telemetry is not isolation."""
    try:
        value = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True,
                               timeout=10, check=True)
        rows = []
        for line in value.stdout.splitlines():
            fields = [v.strip() for v in line.split(',')]
            if len(fields) == 3:
                rows.append(dict(gpu_uuid=fields[0], pid=int(fields[1]), memory_mib=fields[2]))
        return dict(available=True, processes=rows,
                    other_processes=[row for row in rows if row['pid'] != os.getpid()])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(available=False, error=str(error))


def profiler_flops(forward):
    """Partial profiler estimate, not a complete end-to-end operation count."""
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=True, with_flops=True) as prof:
            result = forward()
            del result
        count = sum(event.flops for event in prof.key_averages() if event.flops)
        return dict(available=count > 0, value=count if count > 0 else None,
                    coverage='PyTorch profiler supported convolution/matmul ops only; excludes normalization, activation, routing, pooling and union operations')
    except (RuntimeError, AttributeError) as error:
        return dict(available=False, value=None, error=str(error), coverage='Unavailable')


def benchmark_run(root, protocol, records, kind, seed, arm, device):
    from . import fair_suite as fair
    from . import mechanism_suite as suite
    from .mechanism_models import create_model

    if protocol.get('stage') != 'q07' or kind != 'task_id' or seed != 42:
        raise ValueError('This benchmark is only defined for Q07 Task-ID seed42')
    if torch.device(device).type != 'cuda':
        raise ValueError('Cost measurement requires CUDA')
    root = Path(root)
    c = protocol['config']
    folder = suite.run_folder(root, kind, seed, arm)
    path = folder / 'best.pt'
    cp = torch.load(path, map_location='cpu', weights_only=True)
    suite.check_identity(cp, protocol, kind, seed, arm)
    model = create_model(kind, c['base'], c['top_k'], c['experts'], arm=arm).to(device)
    model.load_state_dict(cp['model'], strict=True)
    epoch = cp['epoch']
    del cp
    model.eval()
    dataset = fair.RegionDataset([r for r in records if r['split'] == 'val'], c['size'], c['task_ids'])
    indices = benchmark_indices(dataset.groups)
    pairs = [dataset[i] for i in indices]
    images = torch.stack([p[0] for p in pairs]).to(device)
    targets = torch.stack([p[1] for p in pairs]).to(device)
    del pairs
    active = targets.ge(0).any((2, 3))
    query_counts = active.sum(1).tolist()
    if query_counts != [5, 5, 5, 5]:
        raise ValueError('Fixed abdomen workload must have five annotated structures per image')
    task_channel = int(active[0].nonzero()[0])
    task = torch.tensor([c['task_ids'][task_channel]], device=device)
    workloads = [('single_task_batch1', 1, 1), ('complete_union_batch1', 1, 5),
                 ('complete_union_batch4', 4, 20)]
    results = []
    original_rng = fair._rng_state()
    original_threads = torch.get_num_threads()
    try:
        with torch.inference_mode():
            for mode, batch, queries in workloads:
                x, target, available = images[:batch], targets[:batch], active[:batch]
                def forward():
                    if mode == 'single_task_batch1':
                        return model(x, task)['logits'].sigmoid()
                    output = fair.forward_model(model, kind, x, target, task_ids=c['task_ids'])['logits']
                    return (output.sigmoid() * available[:, :, None, None]).amax(1)
                torch.cuda.empty_cache()
                for _ in range(5):
                    value = forward()
                    del value
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                latencies = []
                processes_before = gpu_processes()
                for _ in range(20):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    value = forward()
                    torch.cuda.synchronize(device)
                    latencies.append((time.perf_counter() - start) * 1000)
                    del value
                peak = dict(peak_allocated_mib=math.ceil(torch.cuda.max_memory_allocated(device) / 2**20),
                            peak_reserved_mib=math.ceil(torch.cuda.max_memory_reserved(device) / 2**20))
                invoked = invoked_parameter_count(model, forward)
                flops = profiler_flops(forward)
                results.append(dict(workload=mode, image_batch=batch, annotated_queries=queries,
                    warmup=5, repeats=20, input_residency='cuda_before_timing',
                    output='single_task_probability' if queries == 1 else 'annotated_union_probability',
                    **latency_statistics(latencies), **peak, invoked_parameters=invoked,
                    partial_flops=flops, gpu_processes_before=processes_before,
                    gpu_processes_after=gpu_processes(), isolated_device_claimed=False))
    finally:
        fair._restore_rng(original_rng)
        torch.set_num_threads(original_threads)
    report = dict(schema='q07_cost_benchmark_v1', complete=True,
        protocol_sha256=protocol['protocol_sha256'], kind=kind, seed=seed, arm=arm,
        checkpoint_selection='best', checkpoint_epoch=epoch, checkpoint_sha256=suite.file_sha(path),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        dtype='float32', size=c['size'], device=torch.cuda.get_device_name(device),
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cpu_threads=original_threads,
        input_indices=indices, input_split='val', input_domain='AbdomenUS',
        input_image_sha256=fair._digest([dataset.groups[i][0]['image'] for i in indices]),
        query_counts=query_counts, workloads=results,
        scope='Fixed validation abdomen subset; actual conditional expert usage can vary with input',
        isolated_device_claimed=False)
    fair._write_json(folder / 'benchmark.json', report)
    del model, images, targets
    gc.collect()
    torch.cuda.empty_cache()
    return report
