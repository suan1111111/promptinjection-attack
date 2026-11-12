"""LGS-inspired local image gradients, with three additional summary statistics."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .frequency_features import FrequencyFeatureStore, _atomic, _now


CONFIG = {
    "extractor": "lgs-gradient-summary-v1",
    "window": 15, "stride": 10, "threshold": 0.1,
    "decode": "EXIF transpose; alpha on white; Pillow L grayscale /255",
    "derivative": "numpy.gradient central difference, one-sided boundaries",
    "normalize": "global min-max; constant magnitude maps become zero",
    "edges": "append last full window to cover edges; small images use entire axis",
    "resize": None,
    "numpy_version": np.__version__, "pillow_version": Image.__version__,
    "reference": "https://arxiv.org/abs/1807.01216",
    "summary": "project extension, not a validated attack score",
}
CONFIG_ID = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]


def _starts(length, window):
    end = max(0, length - window)
    starts = list(range(0, end + 1, CONFIG["stride"]))
    if starts[-1] != end:
        starts.append(end)
    return np.asarray(starts)


def compute_gradient(gray):
    height, width = gray.shape
    gy = np.gradient(gray, axis=0) if height > 1 else np.zeros_like(gray)
    gx = np.gradient(gray, axis=1) if width > 1 else np.zeros_like(gray)
    magnitude = np.hypot(gx, gy)
    span = float(np.ptp(magnitude))
    normalized = (magnitude - magnitude.min()) / span if span > 1e-15 else np.zeros_like(magnitude)
    wh, ww = min(15, height), min(15, width)
    ys, xs = _starts(height, wh), _starts(width, ww)
    y, x = np.meshgrid(ys, xs, indexing="ij")
    integral = np.pad(normalized.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    means = (integral[y + wh, x + ww] - integral[y, x + ww]
             - integral[y + wh, x] + integral[y, x]) / (wh * ww)
    selected = means > CONFIG["threshold"]
    # Rectangle difference map: union coverage without counting overlap twice.
    coverage = np.zeros((height + 1, width + 1), dtype=np.int32)
    sy, sx = y[selected], x[selected]
    for yy, xx, delta in ((sy, sx, 1), (sy + wh, sx, -1),
                          (sy, sx + ww, -1), (sy + wh, sx + ww, 1)):
        np.add.at(coverage, (yy, xx), delta)
    covered = coverage.cumsum(0).cumsum(1)[:height, :width] > 0
    summary = {"max_window_gradient": float(means.max()),
               "high_gradient_window_ratio": float(selected.mean()),
               "high_gradient_area_ratio": float(covered.mean())}
    return summary, {"window_y": ys, "window_x": xs, "window_means": means,
                     "selected_windows": selected, "window_height": np.array(wh),
                     "window_width": np.array(ww)}


class GradientFeatureStore(FrequencyFeatureStore):
    def extract(self, image_path):
        path = Path(image_path)
        if not path.is_file():
            raise ValueError("Please provide an image file, not a directory: " + str(path))
        content = path.read_bytes()
        image_id = hashlib.sha256(content).hexdigest()
        directory = self.root / image_id / CONFIG_ID
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM features WHERE image_id=? AND config_id=? AND group_name='gradient'",
                             (image_id, CONFIG_ID)).fetchone()
            if row and json.loads(row[0])["status"] == "completed":
                return json.loads(row[0])["values"]
            with Image.open(io.BytesIO(content)) as source:
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Animated/multi-frame images are not supported")
                rgba = ImageOps.exif_transpose(source).convert("RGBA")
                image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("L")
                gray = np.asarray(image, dtype=np.float64) / 255.0
            summary, windows = compute_gradient(gray)
            directory.mkdir(parents=True, exist_ok=True)
            original = directory.parent / "original.bin"
            if not original.exists():
                _atomic(original, lambda stream: stream.write(content))
            _atomic(directory / "windows.npz", lambda stream: np.savez_compressed(stream, **windows))
            payload = {"status": "completed", "values": summary, "configuration": CONFIG,
                       "completed_at": _now(), "analysis_size": list(image.size)}
            db.execute("INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?)",
                       (image_id, CONFIG_ID, "gradient", json.dumps(payload, allow_nan=False)))
            return summary
