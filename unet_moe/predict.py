import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .model import UNetMoE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    task_id = checkpoint['tasks'].index(args.task)
    if 'trained_tasks' in checkpoint and task_id not in checkpoint['trained_tasks']:
        raise ValueError(f'Task has no training records: {args.task}')
    model = UNetMoE(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with Image.open(args.image) as image:
        size = image.size
        pixels = np.array(image.convert('L').resize((checkpoint['size'],) * 2, Image.Resampling.BILINEAR), dtype=np.float32) / 255
    with torch.no_grad():
        logits = model(torch.from_numpy(pixels[None, None]), torch.tensor([task_id]))['logits']
        logits = F.interpolate(logits, size=size[::-1], mode='bilinear', align_corners=False)
        mask = (logits.sigmoid()[0, 0].numpy() >= .5).astype(np.uint8) * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.output)


if __name__ == '__main__':
    main()
