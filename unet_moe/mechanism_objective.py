"""Q06 last-layer marginal routing regularization and Q07 expert-count losses.

The regularizer is training supervision only. Truth chooses positive annotated
queries for its reduction; neither truth nor organ/domain IDs enter the Region
router, and inference is identical in the paired Q06 arms.
"""
import math

import torch

from .fair_suite import forward_model
from .mechanism_models import expert_count_for_arm
from .organ_weighting import weighted_loss
from .routing_objective import objective_from_output as q02_objective


REGION_ROUTING_WEIGHT = .1
REGION_ROUTING_LAYER = 3


def positive_reverse_kl(scores, target):
    """KL(U_9 || mean softmax(scores)) over this microbatch's positive queries.

Scores have shape [image, structure, 9]. Only annotated structures containing
at least one positive target pixel participate; an empty positive set returns
a connected zero. Compute log(mean p) using log-softmax and logsumexp, avoiding
epsilon clipping which would suppress gradients after severe early collapse.
This balances the marginal assignment, not every individual query's entropy.
"""
    if (scores.ndim != 3 or scores.shape[-1] != 9 or target.ndim != 4
            or tuple(scores.shape[:2]) != tuple(target.shape[:2])):
        raise ValueError('Expected matching B,C,9 routing scores and B,C,H,W targets')
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError('Nonfinite Q06 routing scores')
    positive = (target.ge(0) & target.gt(0)).any((2, 3))
    selected = scores[positive].float()
    if not len(selected):
        return scores.sum() * 0
    log_probabilities = selected.log_softmax(-1)
    log_marginal = log_probabilities.logsumexp(0) - math.log(len(selected))
    return -log_marginal.mean() - math.log(scores.shape[-1])


def objective_from_output(output, target, kind, arm, main_loss_arm, weights):
    count = expert_count_for_arm(kind, arm)
    if kind == 'region' or count == 9:
        loss, components = q02_objective(output, target, kind, 'auxiliary', main_loss_arm, weights)
    else:
        # Zero/one experts have no meaningful routing choice. In particular,
        # do not add a constant .01 to the one-expert objective or imply that
        # it receives the same effective routing gradient as nine experts.
        loss = weighted_loss(output['logits'], target, main_loss_arm, weights)
        components = dict(main=float(loss.detach()), coarse=0., balance=0., total=float(loss.detach()))
    routing = loss.new_zeros(())
    if kind == 'region' and arm == 'routing':
        routes = output.get('routing', [])
        if len(routes) != 4:
            raise ValueError('Q06 requires exactly four Region routing layers')
        routing = positive_reverse_kl(routes[REGION_ROUTING_LAYER], target)
        loss = loss + REGION_ROUTING_WEIGHT * routing
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite Q06/Q07 objective')
    return loss, {**components, 'routing': float(routing.detach()), 'total': float(loss.detach())}


def compute_objective(model, kind, image, target, arm, main_loss_arm, weights):
    """Preserve Q05 image/query expansion and effective-batch accumulation.

The new Q06 marginal is formed within the physical microbatch, matching the
existing Q02 balance reduction scope. The caller weights each microbatch loss
by its image count within the effective batch of sixteen images.
"""
    expert_count_for_arm(kind, arm)
    return objective_from_output(forward_model(model, kind, image, target), target,
                                 kind, arm, main_loss_arm, weights)
