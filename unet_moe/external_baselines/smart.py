"""SMART/QMOS segmentation adaptation, preserving the released network widths.

See docs/baselines/SMART_IMPLEMENTATION.md for provenance, training scope and
the window-padding correction. No pretrained checkpoint is loaded here.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.smart.QMOS import QMOS


class SMARTAdapter(nn.Module):
    """Image-only foreground union model with the published two dynamic heads.

    The classifier has no forward dependency back into segmentation. It is
    preserved in the state dict, frozen, and omitted from the default forward
    because the dataset has no original diagnostic labels. The 16 shared query
    tokens and every module contributing to segmentation remain trainable.
    """

    auxiliary_loss_weight = 0.4

    def __init__(self, image_size: int = 512, use_checkpoint: bool = True):
        super().__init__()
        if image_size < 32 or image_size % 32:
            raise ValueError("SMART image_size must be a positive multiple of 32, at least 32")
        self.image_size = image_size
        self.network = QMOS(img_size=image_size, use_checkpoint=use_checkpoint)
        self.network.q_cls_head.requires_grad_(False)
        self.network.q_cls_head.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.network.q_cls_head.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
        *,
        compute_classification: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if images.ndim != 4 or images.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError(f"SMART expects [B,1,{self.image_size},{self.image_size}] grayscale images")
        if domain_ids is not None and domain_ids.shape != (images.shape[0],):
            raise ValueError("domain_ids must have one entry per image")
        full_features, quarter_features, encoder_features = self.network.swin_unet(images.repeat(1, 3, 1, 1))
        logits, quarter_logits = self.network.q_seg_head(
            self.network.query_tokens.expand(images.shape[0], -1, -1),
            encoder_features[-1], full_features, quarter_features,
        )
        # Upsample logits, not binary target masks: unknown pixels remain unknown
        # in the common loss and small structures retain their original labels.
        auxiliary = F.interpolate(quarter_logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        result = {"logits": logits, "auxiliary_logits": [auxiliary]}
        if compute_classification:
            with torch.no_grad():
                binary, four_class = self.network.q_cls_head(
                    encoder_features, quarter_features, self.network.query_tokens,
                )
            result["classification_binary_logits"] = binary
            result["classification_four_logits"] = four_class
        return result


def build_model(*, image_size: int = 512, use_checkpoint: bool = True) -> SMARTAdapter:
    """Build the released SMART widths/depths with fresh random initialization."""
    return SMARTAdapter(image_size=image_size, use_checkpoint=use_checkpoint)
