"""Source-based Peak Spectrum and down/up PSNR; incremental image cache.

Scoring adaptation from EQuiw/2023-detection-scalingattacks (GPL-3.0).
See third_party/scaling_detection/SOURCE.md and LICENSE.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import percentileofscore
from PIL import Image, ImageOps

from third_party.scaling_detection import f_utils
from .frequency_features import FrequencyFeatureStore, _atomic, _now

ALGORITHMS = {'nearest': Image.Resampling.NEAREST, 'bilinear': Image.Resampling.BILINEAR,
              'bicubic': Image.Resampling.BICUBIC, 'lanczos': Image.Resampling.LANCZOS}


def peak_spectrum(rgb, target, radius=5):
    spectrum = f_utils.fourier(rgb)
    mask = f_utils.get_peaks_mask(spectrum.shape, target, f_utils.get_main_peaks, radius)
    if not mask.any():
        return None  # No predicted non-DC peaks; upstream would return NaN.
    return float(percentileofscore(spectrum.flatten(), np.average(spectrum[mask])))


def down_up_psnr(rgb, target, algorithm):
    image = Image.fromarray(rgb)
    restored = image.resize((target[1], target[0]), ALGORITHMS[algorithm]).resize(image.size, ALGORITHMS[algorithm])
    error = rgb.astype(np.float64) - np.asarray(restored, dtype=np.float64)
    mse = float(np.mean(error ** 2))
    return float(20 * np.log10(255) - 10 * np.log10(mse)) if mse else float('inf')


class ScalingFeatureStore(FrequencyFeatureStore):
    def extract(self, image_path, target=(224, 224), algorithm='nearest', radius=5):
        if len(target) != 2 or any(not isinstance(v, int) or v < 1 for v in target):
            raise ValueError('Target height and width must be positive integers')
        if algorithm not in ALGORITHMS or not isinstance(radius, int) or radius < 0:
            raise ValueError('Invalid scaling algorithm or sampling radius')
        content = Path(image_path).read_bytes()
        image_id = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as source:
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Multi-frame images are not supported')
            rgba = ImageOps.exif_transpose(source).convert('RGBA')
            rgb = np.asarray(Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB'))
        if target[0] > rgb.shape[0] or target[1] > rgb.shape[1] or tuple(target) == rgb.shape[:2]:
            raise ValueError('Target must downscale at least one axis and must not upscale either axis')
        config = {'extractor': 'scaling-source-v1', 'target_hw': target, 'library': 'Pillow',
                  'algorithm': algorithm, 'sampling_radius': radius, 'peaks': 'main',
                  'decode': 'RGB8; EXIF transpose; alpha on white',
                  'upstream_utils_sha256': hashlib.sha256(Path(f_utils.__file__).read_bytes()).hexdigest(),
                  'pillow': Image.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}
        config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.root / image_id / config_id
        directory.mkdir(parents=True, exist_ok=True)
        original = directory.parent / 'original.bin'
        if not original.exists():
            _atomic(original, lambda stream: stream.write(content))
        results, failures = {}, {}
        groups = {'peak_spectrum_percentile': lambda: peak_spectrum(rgb, target, radius),
                  'down_up_psnr_db': lambda: down_up_psnr(rgb, target, algorithm)}
        for name, compute in groups.items():
            with self._connect() as db:
                row = db.execute('SELECT payload FROM features WHERE image_id=? AND config_id=? AND group_name=?', (image_id, config_id, name)).fetchone()
            if row:
                payload = json.loads(row[0])
                if payload['status'] in ('completed', 'not_applicable'):
                    results[name] = float('inf') if payload.get('infinite') else payload['value']
                    continue
            try:
                value = compute()
                infinite = value is not None and np.isposinf(value)
                if value is not None and not infinite and not np.isfinite(value):
                    raise ValueError('Non-finite score')
                payload = {'status': 'not_applicable' if value is None else 'completed',
                           'value': None if infinite else value, 'infinite': bool(infinite),
                           'configuration': config, 'completed_at': _now()}
                with self._connect() as db:
                    db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?)', (image_id, config_id, name, json.dumps(payload, allow_nan=False)))
                results[name] = value
            except Exception as error:
                failures[name] = str(error)
                results[name] = None
        return results, failures
