"""Q05 real 512 training-image CPU check, separate from formal GPU admission.

Reads only frozen training records/statistics. It never starts a controller,
loads test images, writes training checkpoints, or changes historical results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from unet_moe import fair_suite as fair
from unet_moe.refinement_models import create_model
from unet_moe.routing_objective import compute_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new verification output; do not overwrite evidence')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    records_path = args.reference_root / 'records.json'
    records = json.loads(records_path.read_text())
    weights = json.loads((args.reference_root / 'train_statistics.json').read_text())
    selection = json.loads((args.reference_root / 'main_loss_selection.json').read_text())
    assert selection['test_used'] is False
    loss_arm = selection['main_loss_arm']
    dataset = fair.RegionDataset([row for row in records if row['split'] == 'train'],
                                512, fair.PAPER_ALIGNED_TASK_IDS)
    index = next(i for i, rows in enumerate(dataset.groups)
                 if rows[0]['dataset'] == 'AbdomenUS')
    image, target = dataset[index]
    image, target = image[None], target[None]
    report = dict(schema='q05_real_cpu_512_verification_v1',
                  created_at=datetime.now(timezone.utc).isoformat(),
                  device='cpu', cuda_initialized=torch.cuda.is_initialized(),
                  threads=2, image_size=512, base=24, experts=9, top_k=1,
                  physical_image_batch=1, optimizer_updates_per_arm=2,
                  formal_batch_16_gpu_preflight=False, split='train',
                  reference_root=str(args.reference_root), training_index=index,
                  record_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest(),
                  image=dataset.groups[index][0]['image'],
                  active_queries=int(target.ge(0).any((2, 3)).sum()),
                  main_loss_arm=loss_arm, objective='q02_auxiliary_in_both_arms',
                  models={})
    for kind in ('region', 'task_id'):
        initial_logits = None
        control_state = None
        arms = {}
        for arm in ('control', 'refined'):
            started = time.monotonic()
            fair._seed(42)
            model = create_model(kind, base=24, top_k=1, experts=9, arm=arm)
            if arm == 'control':
                control_state = {name: value.clone() for name, value in model.state_dict().items()}
            else:
                state = model.state_dict()
                assert all(torch.equal(value, state[name]) for name, value in control_state.items())
                added = state.keys() - control_state.keys()
                assert added and all(torch.count_nonzero(state[name]) == 0 for name in added)
            model.eval()
            with torch.no_grad():
                prediction = fair.forward_model(model, kind, image, target)['logits']
            if arm == 'control':
                initial_logits = prediction.clone()
            else:
                assert torch.equal(initial_logits, prediction), 'Initial 512 prediction differs'
            del prediction
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            updates = []
            for step in range(2):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, terms = compute_objective(model, kind, image, target, 'auxiliary', loss_arm, weights)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                added_grad = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                                 if name not in control_state and p.grad is not None)
                router_grad = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                                  if '.router.' in name and p.grad is not None)
                assert router_grad > 0
                if arm == 'refined':
                    assert added_grad > 0, 'Refinement gets no gradient on real image'
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                updates.append(dict(step=step + 1, loss=float(loss.detach()),
                                    components=terms, preclip_gradient_norm=float(norm),
                                    added_parameter_gradient_l1=added_grad,
                                    router_gradient_l1=router_grad))
                del loss
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                expected = fair.forward_model(model, kind, image, target)['logits'].clone()
            with tempfile.TemporaryDirectory(prefix='q05_cpu_restore_') as directory:
                checkpoint = Path(directory) / 'checkpoint.pt'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                kind=kind, arm=arm), checkpoint)
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                restored = create_model(kind, base=24, top_k=1, experts=9, arm=arm)
                restored.load_state_dict(payload['model'], strict=True)
                restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=1e-4)
                restored_optimizer.load_state_dict(payload['optimizer'])
                restored.eval()
                with torch.no_grad():
                    actual = fair.forward_model(restored, kind, image, target)['logits']
                assert torch.equal(expected, actual)
                assert len(restored_optimizer.state) == len(optimizer.state)
                del restored, restored_optimizer, payload, actual
            arms[arm] = dict(parameters=sum(p.numel() for p in model.parameters()),
                             initial_equivalence=True, updates=updates,
                             strict_checkpoint_reload=True, reload_prediction_equal=True,
                             elapsed_seconds=time.monotonic() - started)
            print(json.dumps(dict(kind=kind, arm=arm, result=arms[arm]), ensure_ascii=False), flush=True)
            del model, optimizer, expected
        report['models'][kind] = arms
        del control_state, initial_logits
    assert not torch.cuda.is_initialized()
    report['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
