"""Plain U-Net baselines with the same image-only inputs as the region model."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import block
from .region import DOMAINS, EXPERT_FOR_REGION, STRUCTURES, masked_loss


DOMAIN_CHANNELS = {
    domain: [i for i, owner in enumerate(EXPERT_FOR_REGION) if owner == index]
    for index, domain in enumerate(DOMAINS)
}


class PlainUNet(nn.Module):
    def __init__(self, outputs=19, base=16):
        super().__init__()
        if base <= 0 or base % 4 or outputs < 1:
            raise ValueError('base must be divisible by 4; outputs must be positive')
        widths = [base * 2**i for i in range(5)]
        self.encoders = nn.ModuleList([
            block(1 if i == 0 else widths[i-1], width)
            for i, width in enumerate(widths)
        ])
        self.fusions = nn.ModuleList([
            block(widths[i+1] + widths[i], widths[i])
            for i in reversed(range(4))
        ])
        self.head = nn.Conv2d(base, outputs, 1)

    def forward(self, image):
        features = []
        x = image
        for i, encoder in enumerate(self.encoders):
            x = encoder(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        for skip, fusion in zip(reversed(features[:-1]), self.fusions):
            x = fusion(torch.cat([
                F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False),
                skip,
            ], 1))
        return self.head(x)


def image_balanced_loss(logits, target, positive_weights=None):
    """Use the same primary segmentation objective as the region model."""
    return masked_loss(logits, target, positive_weights)


@torch.no_grad()
def evaluate(model, loader, device, channels, structures=None):
    structures = STRUCTURES if structures is None else structures
    model.eval()
    totals = {structures[c]: dict(dice_sum=0., positive_sum=0., samples=0,
                                   positive_samples=0, empty_samples=0,
                                   empty_false_positives=0)
              for c in channels}
    for image, target in loader:
        valid = target.to(device) >= 0
        truth = target.to(device) > 0
        pred = (model(image.to(device)).sigmoid() >= .5) & valid
        intersection = (pred & truth).sum((2, 3))
        denominator = pred.sum((2, 3)) + truth.sum((2, 3))
        dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.)
        for b, channel in valid.any((2, 3)).nonzero().tolist():
            row = totals[structures[channels[channel]]]
            score = dice[b, channel].item()
            row['samples'] += 1
            row['dice_sum'] += score
            if truth[b, channel].any():
                row['positive_samples'] += 1
                row['positive_sum'] += score
            else:
                row['empty_samples'] += 1
                row['empty_false_positives'] += int(pred[b, channel].any().item())
    per_structure = {}
    for name, row in totals.items():
        per_structure[name] = {
            'dice': row['dice_sum']/row['samples'] if row['samples'] else None,
            'positive_dice': row['positive_sum']/row['positive_samples'] if row['positive_samples'] else None,
            'samples': row['samples'], 'positive_samples': row['positive_samples'],
            'empty_samples': row['empty_samples'],
            'empty_false_positive_rate': row['empty_false_positives']/row['empty_samples'] if row['empty_samples'] else None,
        }
    values = [r['positive_dice'] for r in per_structure.values() if r['positive_dice'] is not None]
    all_values = [r['dice'] for r in per_structure.values() if r['dice'] is not None]
    empty_values = [r['empty_false_positive_rate'] for r in per_structure.values()
                    if r['empty_false_positive_rate'] is not None]
    return dict(macro_positive_dice=sum(values)/len(values) if values else None,
                macro_all_dice=sum(all_values)/len(all_values) if all_values else None,
                macro_empty_false_positive_rate=sum(empty_values)/len(empty_values) if empty_values else None,
                per_structure=per_structure)
