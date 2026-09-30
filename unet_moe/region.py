"""Image-only, soft-region pooled routing. No task IDs enter forward()."""
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from .model import block
from .data import TASKS, SegmentationDataset

DOMAINS = ['BUSI', 'MMOTU', 'DDTI', 'AUL', 'FALLMUD', 'Fetal_HC', 'CCA', 'CAMUS', 'AbdomenUS']
TASK_IDS = [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12] + list(range(13, 21))
STRUCTURES = [TASKS[i] for i in TASK_IDS]
EXPERT_FOR_REGION = [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7] + [8]*8
PAPER_ALIGNED_TASK_IDS = [task for task in TASK_IDS if task not in (15, 17, 19)]


class RegionDataset(torch.utils.data.Dataset):
    """One image with all available targets; absent annotations stay -1."""
    def __init__(self, records, size=256, task_ids=None):
        self.task_ids = list(TASK_IDS if task_ids is None else task_ids)
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids) or not set(self.task_ids).issubset(TASK_IDS):
            raise ValueError('Invalid region task IDs')
        groups = defaultdict(list)
        for r in records:
            if r['task'] in self.task_ids:
                groups[r['image']].append(r)
        self.groups = list(groups.values())
        self.size = size

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        rows = self.groups[index]
        targets = torch.full((len(self.task_ids), self.size, self.size), -1.)
        for r, (x, y, _) in zip(rows, SegmentationDataset(rows, self.size)):
            targets[self.task_ids.index(r['task'])] = y[0]
        return x, targets


class RegionMoE(nn.Module):
    def __init__(self, channels, regions, top_k):
        super().__init__()
        self.top_k = top_k
        self.proposals = nn.Conv2d(channels, regions, 1)
        self.router = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 9))
        self.shared = block(channels, channels)
        self.experts = nn.ModuleList([block(channels, channels) for _ in range(9)])

    def forward(self, x):
        coarse = self.proposals(x)
        masks = coarse.sigmoid()
        # Each predicted region independently pools its visual features.
        pooled = torch.einsum('brhw,bchw->brc', masks, x) / masks.sum((2, 3)).clamp_min(1e-6)[..., None]
        scores = self.router(pooled).float()
        probs = scores.softmax(-1)
        values, indices = probs.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, indices, values)
        # Region probabilities localize expert contributions spatially.
        weights = torch.einsum('brhw,bre->behw', masks, gates.to(x.dtype))
        weights = weights / masks.sum(1, keepdim=True).clamp_min(1e-6)
        output = self.shared(x)
        for i, expert in enumerate(self.experts):
            batch = torch.where((indices == i).any((1, 2)))[0]
            if batch.numel():
                output = output.index_add(0, batch, expert(x[batch]) * weights[batch, i:i+1])
        load = F.one_hot(indices, 9).float().mean((0, 1, 2))
        return output, coarse, scores, load


class RegionUNet(nn.Module):
    def __init__(self, base=16, top_k=2, task_ids=None):
        super().__init__()
        if base <= 0 or base % 4 or not 1 <= top_k <= 9:
            raise ValueError('Require positive base divisible by 4 and 1 <= top_k <= 9')
        self.task_ids = list(TASK_IDS if task_ids is None else task_ids)
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids) or not set(self.task_ids).issubset(TASK_IDS):
            raise ValueError('Invalid region task IDs')
        self.structures = [TASKS[i] for i in self.task_ids]
        self.expert_for_region = [EXPERT_FOR_REGION[TASK_IDS.index(i)] for i in self.task_ids]
        widths = [base*2**i for i in range(5)]
        self.encoders = nn.ModuleList([block(1 if i == 0 else widths[i-1], w) for i, w in enumerate(widths)])
        self.fusions = nn.ModuleList([block(widths[i+1]+widths[i], widths[i]) for i in reversed(range(4))])
        self.moes = nn.ModuleList([RegionMoE(widths[i], len(self.task_ids), top_k) for i in reversed(range(4))])
        self.head = nn.Conv2d(base, len(self.task_ids), 1)

    def forward(self, image):
        features, x = [], image
        for i, encoder in enumerate(self.encoders):
            x = encoder(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        coarse, routing, loads = [], [], []
        for skip, fusion, moe in zip(reversed(features[:-1]), self.fusions, self.moes):
            x = fusion(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
            x, c, r, load = moe(x)
            coarse.append(F.interpolate(c, size=image.shape[-2:], mode='bilinear', align_corners=False))
            routing.append(r); loads.append(load)
        return dict(logits=self.head(x), coarse=coarse, routing=routing, expert_load=torch.stack(loads))


def masked_loss(logits, target, positive_weights=None):
    """Average annotated channels per image; upweight images containing rare targets."""
    valid = target >= 0
    truth = target.clamp_min(0)
    probability = logits.sigmoid() * valid
    active = valid.any((2, 3))
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction='none') * valid).sum((2, 3))
    bce = bce / valid.sum((2, 3)).clamp_min(1)
    dice = 1 - (2*(probability*truth).sum((2, 3))+1)/(probability.sum((2, 3))+truth.sum((2, 3))+1)
    if positive_weights is None:
        weights = torch.ones_like(bce)
    else:
        if positive_weights.shape != (logits.shape[1],):
            raise ValueError('Expected one positive-image weight per output channel')
        weights = torch.where(truth.gt(0).any((2, 3)),
                              positive_weights.to(device=logits.device, dtype=logits.dtype)[None, :], 1.)
    per_image = ((bce + dice) * weights * active).sum(1) / active.sum(1).clamp_min(1)
    return per_image.mean()


def region_loss(output, target, positive_weights=None, expert_for_region=None,
                coarse_weight=.3, route_weight=1):
    if coarse_weight < 0 or route_weight < 0:
        raise ValueError('Auxiliary loss weights must be nonnegative')
    segmentation = masked_loss(output['logits'], target, positive_weights)
    coarse = torch.stack([masked_loss(x, target, positive_weights) for x in output['coarse']]).mean()
    present = (target > 0).any((2, 3))
    mapping = EXPERT_FOR_REGION if expert_for_region is None else expert_for_region
    if len(mapping) != target.shape[1]:
        raise ValueError('Expert mapping must match target channels')
    labels = torch.tensor(mapping, device=target.device).expand(target.shape[0], -1)
    route = []
    for scores in output['routing']:
        ce = F.cross_entropy(scores.transpose(1, 2), labels, reduction='none')
        route.append((ce*present).sum()/present.sum().clamp_min(1))
    routing = torch.stack(route).mean()
    return segmentation + coarse_weight*coarse + route_weight*routing

