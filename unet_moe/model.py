import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout),
                         nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1),
                         nn.GroupNorm(4, cout), nn.SiLU())


class DecoderMoE(nn.Module):
    def __init__(self, channels, context, experts=4, top_k=2):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError('Require 1 <= top_k <= experts')
        self.top_k = top_k
        self.experts = nn.ModuleList([block(channels, channels) for _ in range(experts)])
        self.shared = block(channels, channels)
        self.router = nn.Linear(channels + context, experts)

    def forward(self, x, context):
        scores = self.router(torch.cat([x.mean((2, 3)), context], 1)).float()
        probabilities = scores.softmax(-1)
        values, indices = probabilities.topk(self.top_k, dim=-1)
        # Unnormalized selected probabilities preserve task gradients even for top-1.
        output = self.shared(x)
        for index, expert in enumerate(self.experts):
            batch, slot = torch.where(indices == index)
            if batch.numel():
                contribution = expert(x[batch]) * values[batch, slot, None, None, None].to(x.dtype)
                output = output.index_add(0, batch, contribution)
        load = F.one_hot(indices, len(self.experts)).float().mean((0, 1))
        balance = len(self.experts) * (load.detach() * probabilities.mean(0)).sum()
        z_loss = scores.logsumexp(-1).square().mean()
        return output, balance, z_loss, load.detach()


class ConditionalDecoder(nn.Module):
    """Shared convolutional control retaining task conditioning, without routing."""
    def __init__(self, channels, context):
        super().__init__()
        self.condition = nn.Linear(context, channels)
        self.shared = block(channels, channels)

    def forward(self, x, context):
        output = self.shared(x + self.condition(context)[:, :, None, None])
        return output, x.new_zeros(()), x.new_zeros(()), x.new_ones(1)


class UNetMoE(nn.Module):
    def __init__(self, tasks, base=24, experts=4, top_k=2, decoder='moe'):
        super().__init__()
        if base <= 0 or base % 4:
            raise ValueError('base must be divisible by 4')
        if decoder not in ('moe', 'shared'):
            raise ValueError('decoder must be moe or shared')
        widths = [base * 2**i for i in range(5)]
        self.embedding = nn.Embedding(tasks, 32)
        self.encoders = nn.ModuleList([block(1 if i == 0 else widths[i-1], w)
                                       for i, w in enumerate(widths)])
        self.fusion = nn.ModuleList([block(widths[i+1] + widths[i], widths[i])
                                    for i in reversed(range(4))])
        self.decoders = nn.ModuleList([(DecoderMoE(widths[i], 32, experts, top_k)
                                      if decoder == 'moe' else ConditionalDecoder(widths[i], 32))
                                      for i in reversed(range(4))])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, image, task):
        features = []
        x = image
        for i, encoder in enumerate(self.encoders):
            x = encoder(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        context = self.embedding(task)
        balances, zs, loads = [], [], []
        for skip, fusion, moe in zip(reversed(features[:-1]), self.fusion, self.decoders):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = fusion(torch.cat([x, skip], 1))
            x, balance, z, load = moe(x, context)
            balances.append(balance)
            zs.append(z)
            loads.append(load)
        return {'logits': self.head(x), 'balance': torch.stack(balances).mean(),
                'router_z': torch.stack(zs).mean(), 'expert_load': torch.stack(loads)}


def segmentation_loss(logits, target):
    valid = target >= 0
    target = target.clamp_min(0)
    probability = logits.sigmoid() * valid
    dice = (2 * (probability * target).sum((1, 2, 3)) + 1) / (
        probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (bce * valid).sum() / valid.sum().clamp_min(1) + 1 - dice.mean()
