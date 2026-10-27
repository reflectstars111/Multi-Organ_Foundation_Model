"""AImageLab finalist fusion architecture, adapted to nine union tasks.

Input is the common dataset's grayscale image in [0, 1]. Both branches use
the same spatial grid: the segmentation branch receives RGB copies of x,
and the classification-feature branch receives 4*x-2, matching its original
intensity convention. No competition cropping or postprocessing is applied.
"""
from __future__ import annotations

import torch
from torch import nn

from .vendor.aimagelab.net import FusionWrapper


class AImageLabUnion(nn.Module):
    """Full ResNet18/CBAM + FiLM U-Net with a trainable fusion gate."""

    def __init__(self, num_domains: int = 9):
        super().__init__()
        if num_domains < 1:
            raise ValueError("num_domains must be positive")
        self.num_domains = num_domains
        self.network = FusionWrapper(
            resnet_size="18",
            use_cbam=True,
            use_film=False,  # Classifier FiLM only; segmentation FiLM stays on.
            n_organs_cls=9,
            num_classes_cls=10,
            predict_bboxes=False,
            mlp_organ=False,
            in_channels=3,
            num_classes=1,
            n_organs=num_domains,
            size=32,
            depth=5,
        )

    def forward(
        self, images: torch.Tensor, domain_ids: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("images must have shape [B,1,H,W]")
        if not images.is_floating_point():
            raise TypeError("images must be floating point in [0,1]")
        if domain_ids.shape != (images.shape[0],) or domain_ids.dtype != torch.long:
            raise ValueError("domain_ids must be torch.long with shape [B]")
        if domain_ids.device != images.device:
            raise ValueError("images and domain_ids must be on the same device")
        if torch.any((domain_ids < 0) | (domain_ids >= self.num_domains)):
            raise ValueError("domain_ids outside configured domain range")
        rgb = images.expand(-1, 3, -1, -1)
        logits = self.network(4.0 * rgb - 2.0, rgb, domain_ids)
        return {"logits": logits, "auxiliary_logits": []}


def build_model(num_domains: int = 9) -> AImageLabUnion:
    """Create the complete published-width network without pretrained weights."""
    return AImageLabUnion(num_domains=num_domains)
