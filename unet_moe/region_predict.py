"""Automatic image-only routing; export independent structure masks."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .region import RegionUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if cp.get('architecture') != 'region_unet_v1':
        raise ValueError('Requires a region_unet_v1 checkpoint, not the task-ID model')
    model = RegionUNet(**cp['config']); model.load_state_dict(cp['model']); model.eval()
    with Image.open(args.image) as image:
        size = image.size
        x = np.array(image.convert('L').resize((cp['size'],)*2, Image.Resampling.BILINEAR), dtype=np.float32)/255
    with torch.no_grad():
        output = model(torch.from_numpy(x[None, None]))
        probs = F.interpolate(output['logits'], size=size[::-1], mode='bilinear', align_corners=False).sigmoid()[0]
    args.output.mkdir(parents=True)
    for name, probability in zip(cp['structures'], probs):
        Image.fromarray((probability.numpy() >= .5).astype(np.uint8)*255).save(args.output/f'{name}.png')
    report = dict(structures=cp['structures'], experts=cp['domains'],
                  routing_probabilities=[x.softmax(-1)[0].tolist() for x in output['routing']],
                  note='Independent sigmoid masks may overlap. Probabilities are uncalibrated. No task ID or dataset ID was provided.')
    (args.output/'routing.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
