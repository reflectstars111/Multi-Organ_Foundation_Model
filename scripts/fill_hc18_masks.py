"""Fill closed HC18 contours without altering source masks."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes


def fill_contour(mask, return_method=False):
    contour = np.asarray(mask) >= 128
    filled = binary_fill_holes(contour)
    method = 'fill_holes'
    if not contour.any():
        raise ValueError('Empty contour')
    if filled.sum() <= contour.sum():
        ys, xs = np.nonzero(contour)
        if len(xs) < 5:
            raise ValueError('Too few contour points')
        ellipse = cv2.fitEllipse(np.column_stack((xs, ys)).astype(np.float32))
        (cx, cy), (w, h), angle = ellipse
        theta = np.deg2rad(angle)
        u = (xs-cx)*np.cos(theta)+(ys-cy)*np.sin(theta)
        v = -(xs-cx)*np.sin(theta)+(ys-cy)*np.cos(theta)
        residual = np.abs(np.sqrt((u/(w/2))**2+(v/(h/2))**2)-1)
        if not np.isfinite(residual).all() or np.quantile(residual, .95) > .05:
            raise ValueError('Contour is not a reliable ellipse fit')
        canvas = np.zeros(contour.shape, np.uint8)
        cv2.ellipse(canvas, ellipse, 255, thickness=-1)
        filled = (canvas > 0) | contour
        method = 'ellipse_fit_clipped_to_image'
    result = filled.astype(np.uint8) * 255
    return (result, method) if return_method else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=Path, required=True)
    args = parser.parse_args()
    root = args.processed
    manifest = root/'manifest_filled.csv'
    output = root/'masks_filled'
    if manifest.exists() or output.exists():
        raise FileExistsError('Filled outputs already exist; refusing overwrite')
    with (root/'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    # Validate every contour before creating outputs; do not invent ellipses for gaps.
    stats = []
    for row in rows:
        if row['pairing_status'] != 'paired':
            continue
        with Image.open(root/row['mask']) as im:
            arr = np.array(im.convert('L'))
        try:
            filled, method = fill_contour(arr, return_method=True)
        except ValueError as exc:
            raise ValueError(f"{row['mask']}: {exc}") from exc
        with Image.open(root/row['image']) as im:
            if im.size != (arr.shape[1], arr.shape[0]):
                raise ValueError('Image/mask size mismatch')
        stats.append(dict(mask=row['mask'], method=method, contour_pixels=int((arr >= 128).sum()),
                          filled_pixels=int((filled > 0).sum())))
    for row in rows:
        if row['pairing_status'] != 'paired':
            continue
        old = Path(row['mask'])
        if old.parts[0] != 'masks' or '..' in old.parts:
            raise ValueError(f'Unexpected mask path: {old}')
        new = Path('masks_filled', *old.parts[1:])
        with Image.open(root/old) as im:
            filled = fill_contour(np.array(im.convert('L')))
        (root/new).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(filled).save(root/new)
        row['mask'] = new.as_posix()
    with manifest.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    report = dict(method='fill holes; validated ellipse fit for open contours; original resolution',
                  count=len(stats), foreground=255, background=0, original_masks_preserved=True,
                  masks=stats)
    (root/'report_filled.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Filled {len(stats)} masks; original contours preserved. Manifest: {manifest}')


if __name__ == '__main__':
    main()
