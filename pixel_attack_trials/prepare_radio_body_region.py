"""Prepare an image-specific radio-body mask for the existing pixel optimizer.

Coordinates refer to the saved 364x252 model input, not the 500x334 source.
This creates mask/preview files only and never launches training.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/shape_location_reproduction_20261103/run/clean_resized.png'
OUT = ROOT / 'output/radio_body_region'

# Conservative hand-traced boundary inside the radio's visible outer casing.
# Excludes surrounding floor, storage boxes, and the wires beyond the casing.
BOUNDARY = [(17,108), (23,97), (34,91), (52,86), (77,83), (99,81),
            (109,77), (118,76), (151,76), (177,77), (199,76),
            (226,77), (242,79), (264,82), (282,85), (305,89),
            (326,92), (337,97), (343,106), (347,121), (349,143),
            (348,159), (345,169), (340,173), (330,175), (310,176),
            (284,178), (253,179), (221,180), (190,179), (158,179),
            (123,178), (89,178), (56,177), (33,176), (23,173),
            (18,166), (16,155), (16,131)]


def main():
    with Image.open(SOURCE) as source:
        image = source.convert('RGB')
    if image.size != (364, 252):
        raise ValueError('Mask coordinates require the original 364x252 training grid')
    OUT.mkdir(parents=True, exist_ok=True)
    mask = Image.new('L', image.size, 0)
    ImageDraw.Draw(mask).polygon(BOUNDARY, fill=255)
    mask.save(OUT / 'radio_body_mask.png')
    preview = Image.blend(image, Image.new('RGB', image.size, (0, 220, 130)), .3)
    preview = Image.composite(preview, image, mask)
    ImageDraw.Draw(preview).line(BOUNDARY + [BOUNDARY[0]], fill=(255, 70, 30), width=1)
    preview.save(OUT / 'radio_body_region_preview.png')
    pixels = np.asarray(mask) > 0
    payload = dict(training_size=list(image.size), mask='radio_body_mask.png',
        editable_pixels=int(pixels.sum()), editable_fraction=float(pixels.mean()),
        boundary=BOUNDARY, bounding_box=list(mask.getbbox()),
        description='Hand-traced radio casing; green overlay is editable. Full-frame mask coordinates.',
        training_changes='Only mask and placement box. Original optimizer/default files untouched.')
    (OUT / 'region.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
