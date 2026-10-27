"""Independent, zero-initialized Q05 visual and task-conditional refinements.

The control factory is exactly the existing fair-comparison model. Candidates
first construct that entire model, then transplant its initialized modules and
add only zero parameters. Neither candidate construction consumes extra random
numbers or changes the original state-dict keys. No frozen training entry uses
this factory unless it explicitly opts in.

Region remains purely image conditioned: each predicted foreground pool is
augmented by a learned projection of its contrast with predicted background.
Task-ID gains task-conditioned FiLM after each existing decoder's shared-plus-
expert output. Selected softmax probabilities remain unnormalized, preserving
the original top-1 segmentation gradient to each router.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from . import fair_suite
from .model import DecoderMoE
from .normalization import forward_normalized_block
from .region import RegionMoE


class _ZeroLinear(nn.Linear):
    """Linear layer whose construction never samples random initialization."""

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class VisualContrastRegionMoE(RegionMoE):
    """Add visual foreground/background contrast, without any organ/domain ID.

    For a predicted mask m_r, z_f=sum(m_r*x)/sum(m_r) and
    z_b=sum((1-m_r)*x)/sum(1-m_r). The original router receives
    z_f + A(z_f-z_b), with one zero-initialized, bias-free A shared by all
    regions in this layer. If either mass is <= 1e-6, the contrast is zero;
    the original foreground pooling and spatial fusion are left intact.
    """

    def __init__(self, original):
        # Do not invoke RegionMoE.__init__: its parameters are already initialized.
        nn.Module.__init__(self)
        self.top_k = original.top_k
        self.proposals = original.proposals
        self.router = original.router
        self.shared = original.shared
        self.experts = original.experts
        weight = original.proposals.weight
        channels = weight.shape[1]
        self.contrast_projection = _ZeroLinear(
            channels, channels, bias=False, device=weight.device, dtype=weight.dtype)

    def forward(self, x, domain_ids=None):
        coarse = self.proposals(x)
        masks = coarse.sigmoid()
        foreground_mass = masks.sum((2, 3))
        pooled = (torch.einsum('brhw,bchw->brc', masks, x)
                  / foreground_mass.clamp_min(1e-6)[..., None])
        background_masks = 1 - masks
        background_mass = background_masks.sum((2, 3))
        background = (torch.einsum('brhw,bchw->brc', background_masks, x)
                      / background_mass.clamp_min(1e-6)[..., None])
        has_contrast = (foreground_mass > 1e-6) & (background_mass > 1e-6)
        contrast = torch.where(has_contrast[..., None], pooled - background,
                               torch.zeros_like(pooled))
        scores = self.router(pooled + self.contrast_projection(contrast)).float()
        probs = scores.softmax(-1)
        values, indices = probs.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, indices, values)
        weights = torch.einsum('brhw,bre->behw', masks, gates.to(x.dtype))
        weights = weights / masks.sum(1, keepdim=True).clamp_min(1e-6)
        output = forward_normalized_block(self.shared, x, domain_ids)
        for i, expert in enumerate(self.experts):
            batch = torch.where((indices == i).any((1, 2)))[0]
            if batch.numel():
                output = output.index_add(0, batch, expert(x[batch]) * weights[batch, i:i+1])
        load = F.one_hot(indices, len(self.experts)).float().mean((0, 1, 2))
        return output, coarse, scores, load


class TaskFiLMDecoder(DecoderMoE):
    """Condition the existing decoder output by (1 + gamma(task))*h + beta(task)."""

    def __init__(self, original):
        # Transfer the existing modules without consuming random numbers again.
        nn.Module.__init__(self)
        self.top_k = original.top_k
        self.experts = original.experts
        self.shared = original.shared
        self.router = original.router
        weight = original.shared[0].weight
        channels = weight.shape[0]
        context = original.router.in_features - channels
        self.film = _ZeroLinear(context, 2 * channels,
                                device=weight.device, dtype=weight.dtype)

    def forward(self, x, context):
        output, balance, z_loss, load = super().forward(x, context)
        gamma, beta = self.film(context).chunk(2, dim=1)
        output = (1 + gamma[:, :, None, None]) * output + beta[:, :, None, None]
        return output, balance, z_loss, load


def create_model(kind, base=24, top_k=1, experts=9, arm='control'):
    """Create an unchanged control or independently enabled refinement.

    Region has the original fixed nine experts. Both refined variants keep the
    original forward contract and remain compatible with fair.forward_model,
    router-score hooks and expert-only residual ablation.
    """
    if kind not in ('region', 'task_id'):
        raise ValueError('Refinement models require kind region or task_id')
    if arm not in ('control', 'refined'):
        raise ValueError('Refinement models require arm control or refined')
    if kind == 'region' and experts != 9:
        raise ValueError('Region retains exactly nine experts')
    model = fair_suite.create_model(kind, base=base, top_k=top_k, experts=experts)
    if arm == 'refined':
        if kind == 'region':
            model.moes = nn.ModuleList(VisualContrastRegionMoE(layer) for layer in model.moes)
        else:
            model.decoders = nn.ModuleList(TaskFiLMDecoder(layer) for layer in model.decoders)
    return model


_ADDED_PARAMETER_SUFFIXES = (
    '.contrast_projection.weight', '.film.weight', '.film.bias',
)


def added_parameter_names(model):
    """Names of only the refinement parameters, suitable for gradient probes."""
    return tuple(name for name, _ in model.named_parameters()
                 if name.endswith(_ADDED_PARAMETER_SUFFIXES))


def added_parameter_count(model):
    """Return 48,960 for Region / 23,760 for Task-ID at base=24 (zero for controls)."""
    parameters = dict(model.named_parameters())
    return sum(parameters[name].numel() for name in added_parameter_names(model))


def original_state_dict(model):
    """Project candidate state onto original keys, without copying tensor storage.

    This is for initialization evidence / explicit projection checks. Candidate
    checkpoints must save the full state_dict and load it with strict=True.
    """
    added = set(added_parameter_names(model))
    state = model.state_dict()
    projected = OrderedDict((name, value) for name, value in state.items() if name not in added)
    if hasattr(state, '_metadata'):
        projected._metadata = OrderedDict(
            (name, value) for name, value in state._metadata.items()
            if not name.endswith(('.contrast_projection', '.film')))
    return projected
