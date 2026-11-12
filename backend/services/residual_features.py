"""Author SRM KB/KV residuals with project-specific scalar summaries.

Residual formula port from SRM.m (DDE Lab, October 2011); see
docs/references/SRM_LICENSE.txt for the original copyright and permission.
No quantization, co-occurrences, classifier or attack score is computed.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .frequency_features import FrequencyFeatureStore, _atomic, _now


KERNELS = {
    'kb': np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64),
    'kv': np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]], dtype=np.float64),
}
CONFIG = {
    'extractor': 'srm-kb-kv-residual-summary-v1',
    'source': 'https://dde.binghamton.edu/download/feature_extractors/download/SRM.zip',
    'source_function': 'SRM.m Residual, KB order=2 and KV order=3',
    'kernels': {name: kernel.tolist() for name, kernel in KERNELS.items()},
    'decode': 'EXIF transpose; alpha on white; Pillow L uint8 converted to float64, 0..255',
    'border': 'valid: KB excludes 1 pixel, KV excludes 2 pixels, no padding',
    'scale': 'raw author filter coefficients, no division or quantization',
    'summaries': 'project extension: mean(abs(D)), std(D,ddof=0), mean(D**2)',
    'numpy_version': np.__version__, 'pillow_version': Image.__version__,
}
CONFIG_ID = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
GROUPS = tuple(KERNELS)


def compute_residual(gray, kind):
    """Valid correlation equivalent to the author's symmetric KB/KV formulas."""
    kernel = KERNELS[kind]
    size = kernel.shape[0]
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2 or not np.isfinite(gray).all():
        raise ValueError('Expected a finite two-dimensional grayscale image')
    height, width = gray.shape
    if min(height, width) < size:
        raise ValueError(f'{kind.upper()} residual requires image dimensions of at least {size} pixels')
    result = np.zeros((height - size + 1, width - size + 1), dtype=np.float64)
    for y in range(size):
        for x in range(size):
            weight = kernel[y, x]
            if weight:
                result += weight * gray[y:y + result.shape[0], x:x + result.shape[1]]
    return result


def summarize(residual, kind):
    return {kind + '_residual_mean_abs': float(np.mean(np.abs(residual))),
            kind + '_residual_std': float(np.std(residual)),
            kind + '_residual_energy': float(np.mean(np.square(residual)))}


class ResidualFeatureStore(FrequencyFeatureStore):
    def extract(self, image_path, groups=GROUPS):
        groups = tuple(dict.fromkeys(groups))
        if not groups or set(groups) - set(GROUPS):
            raise ValueError('Residual groups must be kb and/or kv')
        content = Path(image_path).read_bytes()
        image_id = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as source:
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Multi-frame images are not supported')
            rgba = ImageOps.exif_transpose(source).convert('RGBA')
            image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('L')
        gray = np.asarray(image, dtype=np.float64)
        directory = self.root / image_id / CONFIG_ID
        directory.mkdir(parents=True, exist_ok=True)
        original = directory.parent / 'original.bin'
        output = {}
        for kind in groups:
            # Each group has its own transaction; subsequent failure preserves it.
            error = None
            with self._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT payload FROM features WHERE image_id=? AND config_id=? AND group_name=?',
                                 (image_id, CONFIG_ID, 'residual_' + kind)).fetchone()
                cached = json.loads(row[0]) if row else None
                if cached and cached['status'] == 'completed':
                    output.update(cached['values'])
                    continue
                if not original.exists():
                    _atomic(original, lambda stream: stream.write(content))
                try:
                    values = summarize(compute_residual(gray, kind), kind)
                    payload = {'status': 'completed', 'values': values, 'configuration': CONFIG,
                               'completed_at': _now(), 'analysis_size': list(image.size)}
                    output.update(values)
                except Exception as caught:
                    error = caught
                    payload = {'status': 'failed', 'error': str(caught), 'completed_at': _now()}
                db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?)',
                           (image_id, CONFIG_ID, 'residual_' + kind, json.dumps(payload, allow_nan=False)))
            if error is not None:
                raise error
        return output
