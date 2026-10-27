"""Q02 objectives. No change to either MoE's architecture or routing rule."""
import torch

from .fair_suite import forward_model
from .organ_weighting import weighted_loss


def objective_from_output(output, target, kind, arm, main_loss_arm, weights):
    if kind not in ('region', 'task_id') or arm not in ('control', 'auxiliary'):
        raise ValueError('Unknown Q02 architecture or paired arm')
    main = weighted_loss(output['logits'], target, main_loss_arm, weights)
    coarse, balance = main.new_zeros(()), main.new_zeros(())
    if arm == 'auxiliary' and kind == 'region':
        proposals = output.get('coarse', [])
        if len(proposals) != 4:
            raise ValueError('Q02 requires four full-resolution proposal layers')
        coarse = torch.stack([weighted_loss(p, target, main_loss_arm, weights)
                              for p in proposals]).mean()
    elif arm == 'auxiliary':
        balance = output['balance']
        if balance.numel() != 1:
            raise ValueError('Task-ID balance must be the scalar four-layer mean')
    loss = main + .1 * coarse + .01 * balance
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite Q02 objective')
    return loss, {name: float(value.detach()) for name, value in
                  [('main', main), ('coarse', coarse), ('balance', balance), ('total', loss)]}


def compute_objective(model, kind, image, target, arm, main_loss_arm, weights):
    """Balance is computed over queries in THIS microbatch, not all 16 images.

    The caller accumulates this scalar by image count / effective batch size.
    Region's proposals reuse exactly the main objective's masks and weights.
    Control never attaches auxiliary tensors to the backward graph.
    """
    return objective_from_output(forward_model(model, kind, image, target), target,
                                 kind, arm, main_loss_arm, weights)
