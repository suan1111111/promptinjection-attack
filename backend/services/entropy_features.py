"""Python port of Jedi's entropy candidates, excluding AE and inpainting."""
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from .frequency_features import FrequencyFeatureStore, _atomic, _now

CONFIG = {
    'extractor': 'jedi-entropy-candidates-v1',
    'source': 'https://github.com/ihsenLab/jedi-CVPR2023/blob/main/Jedi-main/jedi.m',
    'patch_mean': [5.32, 6.06, 6.47], 'patch_std': [0.32, 0.34, 0.35],
    'clean_mean': [2.75, 3.19, 3.47], 'clean_std': [1.16, 1.13, 1.07],
    'gray': 'uint8 round RGB*[0.298936021293775,0.587043074451121,0.114020904255103]',
    'decode': 'EXIF transpose; alpha on white; RGB8; no resize',
    'windows': 'upstream size-adaptive 3 scales; inclusive endpoints; symmetric padding',
    'heatmap': 'scan row then column; last window overwrites; MATLAB growth and crop',
    'components': '8-connectivity; original area filter, including absolute max Area > .01',
    'numpy': np.__version__, 'pillow': Image.__version__,
}
CONFIG_ID = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]


def window_sizes(shape):
    sizes = [math.ceil(length / 100) for length in shape]
    base = max(8, *(value + value % 2 for value in sizes))
    middle = int(base * 1.5)
    return (base, middle + middle % 2, base * 2)


def shannon_entropy(window):
    counts = np.bincount(window.ravel(), minlength=256)
    p = counts[counts > 0] / window.size
    return float(-(p * np.log2(p)).sum())


def compute_scale(gray, winsize, scale):
    height, width = gray.shape
    half = winsize // 2
    padded = np.pad(gray, ((half, half), (half, half)), mode='symmetric')
    ys = range(0, padded.shape[0] - winsize, half)
    xs = range(0, padded.shape[1] - winsize, half)
    # MATLAB assignment expands its initially HxW array only as required.
    heat = np.zeros((max(height, ys[-1] + winsize + 1),
                     max(width, xs[-1] + winsize + 1)), dtype=np.float64)
    values = np.empty((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            value = shannon_entropy(padded[y:y + winsize + 1, x:x + winsize + 1])
            values[i, j] = value
            heat[y:y + winsize + 1, x:x + winsize + 1] = value
    # MATLAB positive round ties away from zero, and indices are one-based.
    cy = math.floor((heat.shape[0] - height) / 2 + .5) - 1
    cx = math.floor((heat.shape[1] - width) / 2 + .5) - 1
    heat = heat[cy:cy + height, cx:cx + width]
    pm, ps = CONFIG['patch_mean'][scale], CONFIG['patch_std'][scale]
    cm, cs = CONFIG['clean_mean'][scale], CONFIG['clean_std'][scale]
    mean = float(values.mean())
    threshold = pm + 1.5 * ps - ((pm - cm) / pm * 2) * ps + ((mean - cm) / cs) * ps
    summary = {'entropy_mean': mean, 'entropy_std': float(values.std()),
               'entropy_max': float(values.max()), 'threshold': threshold,
               'high_entropy_area_ratio': float((heat > threshold).mean())}
    return summary, {'window_entropies': values, 'heatmap': heat, 'mask': heat > threshold,
                     'window_parameter': np.array(winsize), 'stride': np.array(half)}


def filter_candidates(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]
    keep = np.ones(count + 1, dtype=bool)
    keep[0] = False
    if count:
        # Preserve upstream's absolute area comparison rather than silently fixing it.
        keep[1:] = ~((areas / mask.size < .005) & ((count > 1) | (areas.max() > .01)))
    cleaned = keep[labels]
    retained = areas[keep[1:]]
    return {'candidate_area_ratio': float(cleaned.mean()),
            'candidate_region_count': int(len(retained)),
            'largest_candidate_area_ratio': float(retained.max() / mask.size) if len(retained) else 0.0}, cleaned


class EntropyFeatureStore(FrequencyFeatureStore):
    def extract(self, image_path):
        path = Path(image_path)
        if not path.is_file():
            raise ValueError('Please provide an image file, not a directory: ' + str(path))
        content = path.read_bytes()
        image_id = hashlib.sha256(content).hexdigest()
        directory = self.root / image_id / CONFIG_ID
        with Image.open(io.BytesIO(content)) as source:
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Animated/multi-frame images are not supported')
            rgba = ImageOps.exif_transpose(source).convert('RGBA')
            rgb = np.asarray(Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB'), dtype=np.float64)
        gray = np.floor(rgb @ np.array([.298936021293775, .587043074451121, .114020904255103]) + .5).clip(0, 255).astype(np.uint8)
        output, masks = {}, []
        directory.mkdir(parents=True, exist_ok=True)
        original = directory.parent / 'original.bin'
        if not original.exists():
            _atomic(original, lambda stream: stream.write(content))
        for scale, size in enumerate(window_sizes(gray.shape)):
            name = 'entropy_scale_' + str(scale)
            artifact = directory / (name + '.npz')
            with self._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT payload FROM features WHERE image_id=? AND config_id=? AND group_name=?', (image_id, CONFIG_ID, name)).fetchone()
                if row and artifact.exists():
                    summary = json.loads(row[0])['values']
                    with np.load(artifact, allow_pickle=False) as cached:
                        mask = cached['mask']
                else:
                    summary, details = compute_scale(gray, size, scale)
                    mask = details['mask']
                    _atomic(artifact, lambda stream: np.savez_compressed(stream, **details))
                    db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?)', (image_id, CONFIG_ID, name, json.dumps({'status': 'completed', 'values': summary, 'completed_at': _now(), 'configuration': CONFIG}, allow_nan=False)))
            # Each scale commits separately, surviving a later error or interruption.
            output.update({f'scale_{scale + 1}_{key}': value for key, value in summary.items()})
            masks.append(mask)
        summary, mask = filter_candidates(np.logical_or.reduce(masks))
        _atomic(directory / 'candidate_mask.npy', lambda stream: np.save(stream, mask, allow_pickle=False))
        with self._connect() as db:
            db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?)', (image_id, CONFIG_ID, 'entropy_candidates', json.dumps({'status': 'completed', 'values': summary, 'completed_at': _now()}, allow_nan=False)))
        output.update(summary)
        return output
