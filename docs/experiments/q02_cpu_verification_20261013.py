"""Functional 512px Q02 check using real TRAIN data; not formal GPU admission.

Run from repository root with CUDA_VISIBLE_DEVICES='' .conda/bin/python -m
docs.experiments.q02_cpu_verification_20261013 --output /path/to/report.json
"""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from unet_moe import fair_suite as fair
from unet_moe.organ_weighted_suite import state_digest
from unet_moe.routing_diagnostics import diagnose
from unet_moe.routing_objective import compute_objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference = Path('/ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007')
    records = json.loads((reference / 'records.json').read_text())
    weights = json.loads((reference / 'train_statistics.json').read_text())
    dataset = fair.RegionDataset([r for r in records if r['split'] == 'train'], 512,
                                 fair.PAPER_ALIGNED_TASK_IDS)
    index = weights['representative_indices']['abdomen_vessels']['positive']
    image, target = dataset[index]
    image, target = image[None], target[None]
    results = []
    for kind in ('region', 'task_id'):
        initial = None
        for arm in ('control', 'auxiliary'):
            fair._seed(42)
            model = fair.create_model(kind, base=24, experts=9, top_k=1)
            digest = state_digest(model.state_dict())
            if initial is not None and digest != initial:
                raise AssertionError('Paired initialization differs')
            initial = digest
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            terms, started = [], time.perf_counter()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss, components = compute_objective(model, kind, image, target, arm,
                                                      'organ_exposure_fg_bg', weights)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                terms.append({**components, 'gradient_norm': float(norm)})
            if state_digest(model.state_dict()) == digest:
                raise AssertionError('Optimizer did not change model')
            saved = copy.deepcopy(model.state_dict())
            rng = torch.get_rng_state().clone()
            report = diagnose(model, kind, dataset, [index], 'cpu', 'organ_exposure_fg_bg', weights, arm)
            assert all(torch.equal(saved[key], value) for key, value in model.state_dict().items())
            assert torch.equal(rng, torch.get_rng_state())
            row = dict(kind=kind, arm=arm, initialization_sha256=digest, updates=terms,
                       expert_gradient_probe=report, elapsed_seconds=time.perf_counter() - started)
            results.append(row)
            print(json.dumps(dict(kind=kind, arm=arm, passed=True, seconds=row['elapsed_seconds'])), flush=True)
            del model, optimizer, saved, report, loss
    result = dict(passed=True, device='cpu', size=512, base=24, seed=42,
                  main_loss_arm_for_functional_check='organ_exposure_fg_bg',
                  main_loss_selection_frozen=False, physical_batch=1, effective_batch=1,
                  queries_per_image=int(target.ge(0).any((2, 3)).sum()),
                  scope='Real train image, two optimizer updates per kind/arm; not formal GPU/memory preflight',
                  image=dataset.groups[index][0]['image'], results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
