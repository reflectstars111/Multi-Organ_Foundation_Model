"""Explicit domain normalization for Region MoE's dense shared paths."""
import math

import torch
from torch import nn


NORMALIZATION_TYPES = ('shared_gn', 'domain_gn', 'shared_bn', 'domain_bn')


def validate_normalization(normalization, eps, momentum):
    if normalization not in NORMALIZATION_TYPES:
        raise ValueError(f'Unknown normalization: {normalization}')
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Normalization epsilon must be finite and positive')
    if (not isinstance(momentum, (int, float)) or not math.isfinite(momentum)
            or not 0 <= momentum <= 1):
        raise ValueError('BatchNorm momentum must be finite and within [0, 1]')


def validate_domain_ids(domain_ids, image, num_domains):
    """Require explicit domain metadata, never infer a domain from target masks."""
    if not isinstance(domain_ids, torch.Tensor):
        raise TypeError('Domain normalization requires a tensor of domain IDs')
    integers = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if domain_ids.dtype not in integers:
        raise TypeError('Domain IDs must use an integer dtype (not bool)')
    if domain_ids.ndim != 1 or domain_ids.shape[0] != image.shape[0]:
        raise ValueError('Domain IDs must have shape [batch]')
    if domain_ids.device != image.device:
        raise ValueError('Domain IDs and image must be on the same device')
    if domain_ids.numel() == 0:
        raise ValueError('Domain normalization requires a nonempty image batch')
    if int(domain_ids.min()) < 0 or int(domain_ids.max()) >= num_domains:
        raise ValueError(f'Domain IDs must be in [0, {num_domains - 1}]')


class DomainSpecificNorm2d(nn.Module):
    """Independent affine parameters and, for BN, running statistics per domain.

    Only present domains execute a branch; missing domains' parameters and
    buffers are untouched. Splitting/reassembly preserves image order and
    autograd. BatchNorm uses each *physical* domain subbatch, not accumulated
    optimizer batches, and always keeps running statistics for evaluation.
    """

    def __init__(self, channels, num_domains=9, kind='batch', eps=1e-5, momentum=.1):
        super().__init__()
        if kind not in ('batch', 'group'):
            raise ValueError('Domain norm kind must be batch or group')
        validate_normalization('domain_bn' if kind == 'batch' else 'domain_gn', eps, momentum)
        if not isinstance(num_domains, int) or isinstance(num_domains, bool) or num_domains < 1:
            raise ValueError('Require a positive integer number of domains')
        if not isinstance(channels, int) or channels < 1 or (kind == 'group' and channels % 4):
            raise ValueError('Require positive channels, divisible by four for GroupNorm')
        self.channels = channels
        self.num_domains = num_domains
        self.kind = kind
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(channels, eps=eps, momentum=momentum, track_running_stats=True)
            if kind == 'batch' else nn.GroupNorm(4, channels, eps=eps)
            for _ in range(num_domains)])

    def forward(self, x, domain_ids):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('Domain norm expects [batch, channels, height, width]')
        validate_domain_ids(domain_ids, x, self.num_domains)
        output = torch.zeros_like(x)
        for domain in torch.unique(domain_ids).tolist():
            indices = torch.where(domain_ids == domain)[0]
            normalized = self.norms[domain](x.index_select(0, indices))
            output = output.index_copy(0, indices, normalized)
        return output


class DomainSequential(nn.Sequential):
    """Keep legacy numeric state keys while forwarding explicit domain metadata."""

    def forward(self, x, domain_ids):
        for module in self:
            x = module(x, domain_ids) if isinstance(module, DomainSpecificNorm2d) else module(x)
        return x


def normalization_block(cin, cout, normalization='shared_gn', eps=1e-5,
                        momentum=.1, num_domains=9):
    """Conv/norm construction preserves legacy initialization order and state keys."""
    validate_normalization(normalization, eps, momentum)

    def make_norm():
        if normalization == 'shared_gn':
            return nn.GroupNorm(4, cout, eps=eps)
        if normalization == 'shared_bn':
            return nn.BatchNorm2d(cout, eps=eps, momentum=momentum, track_running_stats=True)
        return DomainSpecificNorm2d(cout, num_domains,
                                    'group' if normalization == 'domain_gn' else 'batch',
                                    eps=eps, momentum=momentum)

    sequence = DomainSequential if normalization.startswith('domain_') else nn.Sequential
    return sequence(nn.Conv2d(cin, cout, 3, padding=1), make_norm(), nn.SiLU(),
                    nn.Conv2d(cout, cout, 3, padding=1), make_norm(), nn.SiLU())


def forward_normalized_block(module, x, domain_ids=None):
    return module(x, domain_ids) if isinstance(module, DomainSequential) else module(x)


@torch.no_grad()
def normalization_diagnostics(model):
    """Summarize dense normalization buffers without forwards or mode changes."""
    def moments(tensor):
        values = tensor.detach().float()
        return dict(mean=float(values.mean()), std=float(values.std(unbiased=False)),
                    min=float(values.min()), max=float(values.max()))

    def describe(name, module, domain):
        result = dict(name=name, type=type(module).__name__, domain_index=domain,
                      num_channels=module.num_features if isinstance(module, nn.BatchNorm2d)
                      else module.num_channels)
        if isinstance(module, nn.BatchNorm2d):
            result.update(num_batches_tracked=int(module.num_batches_tracked),
                          running_mean=moments(module.running_mean),
                          running_var=moments(module.running_var))
        return result

    rows = []
    domain_prefixes = []
    for name, module in model.named_modules():
        if 'experts' in name.split('.') or any(name.startswith(prefix) for prefix in domain_prefixes):
            continue
        if isinstance(module, DomainSpecificNorm2d):
            rows.extend(describe(name, norm, domain) for domain, norm in enumerate(module.norms))
            domain_prefixes.append(name + '.')
        elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
            rows.append(describe(name, module, None))
    return dict(normalization=getattr(model, 'normalization', None),
                scope='dense_shared_paths', layers=rows)
