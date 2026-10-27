"""Pure tensor helpers extracted from the published AImageLab utility file."""
import torch
import torch.nn.functional as F

organ_to_class_dict = {
    "appendix": 0,
    "breast": 1,
    "cardiac": 2,
    "thyroid": 3,
    "fetal": 4,
    "kidney": 5,
    "liver": 6,
    "testicles": 7,
    "breast_luminal": 8,
}


def pad_to_2d(x: torch.Tensor, stride: int):
    h, w = x.shape[-2:]

    new_h = h if h % stride == 0 else h + stride - (h % stride)
    new_w = w if w % stride == 0 else w + stride - (w % stride)

    top = (new_h - h) // 2
    bottom = (new_h - h) - top
    left = (new_w - w) // 2
    right = (new_w - w) - left

    pads = (left, right, top, bottom)
    x_pad = F.pad(x, pads, mode="constant", value=0)
    return x_pad, pads


def unpad_2d(x: torch.Tensor, pads):
    left, right, top, bottom = pads

    if top or bottom:
        end_h = -bottom if bottom > 0 else None
        x = x[:, :, top:end_h, :]

    if left or right:
        end_w = -right if right > 0 else None
        x = x[:, :, :, left:end_w]

    return x
