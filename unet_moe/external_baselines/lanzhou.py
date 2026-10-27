"""Published ExactFusedSwinUNet with the common binary union interface."""
from __future__ import annotations

import torch
from torch import nn

from .vendor.lanzhou.exact_model import ExactFusedSwinUNet


# Retain the published diagnostic heads, without making up diagnostic labels
# or losses for this segmentation-only retraining protocol.
CLASSIFICATION_TASK_INFO = {
    "Appendix": 2,
    "BUS-BRA": 2,
    "BUSI": 2,
    "Breast": 2,
    "Breast_luminal": 4,
    "Fatty-Liver": 2,
    "Liver": 2,
}


def foreground_logit(two_class_logits: torch.Tensor) -> torch.Tensor:
    """sigmoid(z_fg-z_bg) is exactly softmax([z_bg,z_fg])_fg."""
    if two_class_logits.ndim != 4 or two_class_logits.shape[1] != 2:
        raise ValueError("two_class_logits must have shape [B,2,H,W]")
    return two_class_logits[:, 1:2] - two_class_logits[:, 0:1]


class LanzhouUnion(nn.Module):
    """Finalist BN/dilated-block model, including fixed deep supervision fusion."""

    def __init__(self, num_domains: int = 9):
        super().__init__()
        if num_domains < 1:
            raise ValueError("num_domains must be positive")
        self.num_domains = num_domains
        self.network = ExactFusedSwinUNet(
            seg_classes=2,
            classification_task_info=CLASSIFICATION_TASK_INFO,
            use_gn=False,
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self, images: torch.Tensor, domain_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("images must have shape [B,1,H,W]")
        if not images.is_floating_point():
            raise TypeError("images must be floating point in [0,1]")
        if domain_ids is not None and domain_ids.shape != (images.shape[0],):
            raise ValueError("domain_ids must have shape [B] when supplied")
        rgb = images.expand(-1, 3, -1, -1)
        normalized = (rgb - self.image_mean) / self.image_std
        # domain_ids deliberately never enter this unconditional model.
        two_class, _ = self.network(normalized)
        # The vendor output already contains final + .4*ds1 + .3*ds2 + .3*ds3.
        # Returning those branches again as losses would change the method.
        return {"logits": foreground_logit(two_class), "auxiliary_logits": []}


def build_model(num_domains: int = 9) -> LanzhouUnion:
    """Create the final published architecture; no ImageNet/contest weights."""
    return LanzhouUnion(num_domains=num_domains)
