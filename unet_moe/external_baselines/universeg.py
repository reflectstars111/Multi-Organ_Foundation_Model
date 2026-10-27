"""Official four-level, width-64 UniverSeg, with activation recomputation.

Upstream JJGO/UniverSeg commit 833a0c34c65e38d675e21bd48ddec6797cc03259.
The vendored model and CrossConv retain their original parameters and math.
Recomputation changes memory use, not the support/target interaction.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .vendor.universeg.model import UniverSeg
from .vendor.universeg.nn import vmap


class UniverSegAdapter(nn.Module):
    def __init__(self, activation_checkpointing=True):
        super().__init__()
        self.network = UniverSeg(encoder_blocks=[64, 64, 64, 64])
        self.activation_checkpointing = activation_checkpointing

    def _block(self, block, target, support):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, target, support, use_reentrant=False)
        return block(target, support)

    def forward(self, images, domain_ids=None, *, support_images, support_labels):
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError('UniverSeg requires B,1,H,W query images')
        if (support_images.shape != support_labels.shape or support_images.ndim != 5
                or support_images.shape[0] != images.shape[0]
                or support_images.shape[2:] != images.shape[1:]
                or support_images.shape[1] < 1):
            raise ValueError('Expected matching B,S,1,H,W support images and labels')
        if not ((support_labels == 0) | (support_labels == 1)).all():
            raise ValueError('Unknown/ignored support labels cannot be converted to background')
        network = self.network
        target = images.unsqueeze(1)
        support = torch.cat((support_images, support_labels), dim=2)
        skips = []
        for i, block in enumerate(network.enc_blocks):
            target, support = self._block(block, target, support)
            if i < len(network.enc_blocks) - 1:
                skips.append((target, support))
                target = vmap(network.downsample, target)
                support = vmap(network.downsample, support)
        for block in network.dec_blocks:
            target_skip, support_skip = skips.pop()
            target = torch.cat((vmap(network.upsample, target), target_skip), dim=2)
            support = torch.cat((vmap(network.upsample, support), support_skip), dim=2)
            target, support = self._block(block, target, support)
        return dict(logits=network.out_conv(target[:, 0]), auxiliary_logits=[])


def build_model():
    return UniverSegAdapter()
