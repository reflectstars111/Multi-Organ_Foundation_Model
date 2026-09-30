"""Save illustrative failures from an existing region checkpoint on held-out AUS."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .region import RegionUNet, STRUCTURES


CLASSES = {'abdomen_vessels': 4, 'abdomen_adrenal': 5, 'abdomen_bone': 7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint.get('architecture') != 'region_unet_v1':
        raise ValueError('Region checkpoint required')
    model = RegionUNet(**checkpoint['config']).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    records = json.loads(args.records.read_text(encoding='utf-8'))['records']
    unique = {r['image']:r for r in records if r['dataset']=='AbdomenUS' and r['split']=='test'}
    size = checkpoint['size']
    choices = {}
    for name, value in CLASSES.items():
        candidates = []
        for row in unique.values():
            with Image.open(row['mask']) as im:
                mask = np.asarray(im.resize((size,size),Image.Resampling.NEAREST))
            pixels = int((mask == value).sum())
            if pixels:
                candidates.append((pixels,row))
        candidates.sort(key=lambda x:x[0])
        choices[name] = candidates[len(candidates)//2]
    args.output.mkdir(parents=True)
    report = {}
    for name,(pixels,row) in choices.items():
        with Image.open(row['image']) as im:
            gray = np.asarray(im.convert('L').resize((size,size),Image.Resampling.BILINEAR))
        with Image.open(row['mask']) as im:
            mask = np.asarray(im.resize((size,size),Image.Resampling.NEAREST))
        x = torch.from_numpy(gray.astype(np.float32)[None,None]/255).to(args.device)
        with torch.no_grad():
            probability = model(x)['logits'].sigmoid()[0,STRUCTURES.index(name)].cpu().numpy()
        truth = mask == CLASSES[name]
        prediction = (probability >= .5) & (mask != 255)
        base = np.repeat(gray[...,None],3,axis=2)
        overlay = base.copy()
        overlay[truth] = [255,80,45]
        overlay[prediction] = [45,220,120]
        overlay[truth & prediction] = [255,240,40]
        heat = np.zeros_like(base)
        heat[...,0] = (probability*255).astype(np.uint8)
        heat[...,2] = ((1-probability)*255).astype(np.uint8)
        canvas = Image.fromarray(np.concatenate([base,overlay,heat],axis=1))
        draw = ImageDraw.Draw(canvas)
        draw.text((5,5),'IMAGE',fill='white',stroke_width=1,stroke_fill='black')
        draw.text((size+5,5),'GT red / pred green / overlap yellow',fill='white',stroke_width=1,stroke_fill='black')
        draw.text((2*size+5,5),'PROB red=high',fill='white',stroke_width=1,stroke_fill='black')
        canvas.save(args.output/f'{name}.png')
        report[name] = dict(image=row['image'], ground_truth_pixels=pixels,
                            predicted_pixels=int(prediction.sum()), overlap_pixels=int((prediction & truth).sum()),
                            max_probability=float(probability.max()),
                            mean_probability_on_truth=float(probability[truth].mean()))
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
