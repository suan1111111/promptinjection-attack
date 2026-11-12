"""Persistent, incremental RGB Fourier features; no model inference required."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps

GROUPS = ("bands", "statistics")
CONFIG = {
    "extractor": "rgb-frequency-v1",
    "decode": "Pillow RGB8, EXIF transpose, alpha composite on white",
    "resize": None,
    "window": "none",
    "fft": "numpy.fft.fft2 norm=ortho, per RGB channel",
    "power": "abs(FFT)**2; DC excluded from distribution features",
    "radius": "sqrt(fx**2+fy**2)/sqrt(0.5**2+0.5**2), cycles/pixel",
    "band_edges": [0.0, 0.15, 0.5, 1.0],
    "numpy_version": np.__version__,
    "pillow_version": Image.__version__,
}
CONFIG_ID = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic(path, writer):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _radius(height, width):
    return np.hypot(np.fft.fftfreq(height)[:, None],
                    np.fft.fftfreq(width)[None, :]) / np.sqrt(0.5)


def _distribution(power):
    ac = power.copy()
    ac[0, 0, :] = 0
    totals = ac.sum(axis=(0, 1))
    # FFT roundoff on a constant channel is not meaningful AC energy.
    valid = totals > np.maximum(power.sum(axis=(0, 1)) * 1e-24, 1e-30)
    ac[:, :, ~valid] = 0
    totals[~valid] = 0
    return ac, totals


def _extract(group, power):
    height, width, _ = power.shape
    radius = _radius(height, width)
    ac, totals = _distribution(power)
    output = {}
    for index, channel in enumerate(("red", "green", "blue")):
        total = float(totals[index])
        values = ac[:, :, index]
        if group == "bands":
            channel_data = {"ac_energy": total}
            for name, low, high in zip(("low", "mid", "high"),
                                       CONFIG["band_edges"][:-1], CONFIG["band_edges"][1:]):
                mask = (radius >= low) & (radius < high if high < 1 else radius <= high)
                energy = float(values[mask].sum())
                channel_data[name + "_energy"] = energy
                channel_data[name + "_ratio"] = energy / total if total else None
        else:
            full_total = float(power[:, :, index].sum())
            channel_data = {"total_energy": full_total,
                            "dc_energy_ratio": float(power[0, 0, index]) / full_total if full_total else None}
            probabilities = values.ravel() / total if total else None
            if probabilities is None:
                channel_data.update(centroid=None, bandwidth=None, entropy_normalized=None,
                                    peak_energy_ratio=None, radius_q50=None, radius_q90=None)
            else:
                centroid = float((values * radius).sum() / total)
                positive = probabilities[probabilities > 0]
                count = height * width - 1
                entropy = float(-(positive * np.log(positive)).sum())
                order = np.argsort(radius.ravel())
                cumulative = np.cumsum(probabilities[order])
                sorted_radius = radius.ravel()[order]
                channel_data.update(
                    centroid=centroid,
                    bandwidth=float(np.sqrt((values * (radius - centroid)**2).sum() / total)),
                    entropy_normalized=entropy / np.log(count) if count > 1 else 0.0,
                    peak_energy_ratio=float(probabilities.max()),
                    radius_q50=float(sorted_radius[min(np.searchsorted(cumulative, 0.5), len(order)-1)]),
                    radius_q90=float(sorted_radius[min(np.searchsorted(cumulative, 0.9), len(order)-1)]),
                )
        output[channel] = channel_data
    return output


class FrequencyFeatureStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "image_features"
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS features (image_id TEXT, config_id TEXT, group_name TEXT, payload TEXT, PRIMARY KEY(image_id, config_id, group_name))")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.root / "features.sqlite3", timeout=120)
        try:
            db.execute("PRAGMA busy_timeout=120000")
            with db:
                yield db
        finally:
            db.close()

    def extract(self, image_path, groups=GROUPS):
        return self.extract_bytes(Path(image_path).read_bytes(), Path(image_path).name, groups)

    def extract_de_five(self, image_path):
        """Equations (4)-(10), r_low=8, epsilon=1e-8.

        Paper leaves RGB aggregation unspecified: average per-channel scores.
        Preserve the original spectrum/groups; cache this interpretation separately.
        """
        result = self.extract(image_path, groups=("bands",))
        if result["failed_groups"]:
            raise ValueError(result["report"]["groups"]["bands"]["error"])
        image_id = result["report"]["image_id"]
        score_config = CONFIG_ID + "-defive-r8-rgbmean-v1"
        with self._connect() as db:
            row = db.execute("SELECT payload FROM features WHERE image_id=? AND config_id=? AND group_name=?",
                             (image_id, score_config, "de_five")).fetchone()
            if row:
                return json.loads(row[0])["score"]
            power = np.load(Path(result["cache_path"]) / "power.npy", allow_pickle=False)
            values = de_five_values(power)
            db.execute("INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?)",
                       (image_id, score_config, "de_five", json.dumps(values, allow_nan=False)))
            return values["score"]

    def extract_de_five_components(self, image_path):
        """Return six separate RGB components without calculating combined scores."""
        content = Path(image_path).read_bytes()
        image_id = hashlib.sha256(content).hexdigest()
        config_id = CONFIG_ID + "-defive-components-r8-v1"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM features WHERE image_id=? AND config_id=? AND group_name=?",
                             (image_id, config_id, "de_five_components")).fetchone()
            if row:
                return json.loads(row[0])
            with Image.open(io.BytesIO(content)) as source:
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Animated/multi-frame images are not supported")
                rgba = ImageOps.exif_transpose(source).convert("RGBA")
                image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
                rgb = np.asarray(image, dtype=np.float64) / 255.0
            directory = self.root / image_id / CONFIG_ID
            directory.mkdir(parents=True, exist_ok=True)
            original = directory.parent / "original.bin"
            if not original.exists():
                _atomic(original, lambda stream: stream.write(content))
            spectrum_path = directory / "power.npy"
            if spectrum_path.exists():
                power = np.load(spectrum_path, allow_pickle=False)
                if power.shape != rgb.shape or not np.isfinite(power).all() or (power < 0).any():
                    raise ValueError("Invalid cached power spectrum")
            else:
                power = np.abs(np.fft.fft2(rgb, axes=(0, 1), norm="ortho"))**2
                _atomic(spectrum_path, lambda stream: np.save(stream, power, allow_pickle=False))
            channels = de_five_values(power, include_score=False)["channels"]
            values = {f"{channel}_de_five_{name}": value
                      for channel, components in zip(("red", "green", "blue"), channels)
                      for name, value in components.items()}
            db.execute("INSERT OR REPLACE INTO features VALUES (?,?,?,?)",
                       (image_id, config_id, "de_five_components", json.dumps(values, allow_nan=False)))
            return values

    def extract_bytes(self, content, filename="image", groups=GROUPS):
        requested = tuple(dict.fromkeys(groups))
        if not requested or set(requested) - set(GROUPS):
            raise ValueError("groups must contain bands and/or statistics")
        image_id = hashlib.sha256(content).hexdigest()
        # Validate and decode before creating the persistent image record.
        with Image.open(io.BytesIO(content)) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Animated/multi-frame images are not supported")
            source_format, original_mode = source.format, source.mode
            source_size = source.size
            oriented = ImageOps.exif_transpose(source)
            rgba = oriented.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            rgb = np.asarray(image, dtype=np.float64) / 255.0
        directory = self.root / image_id / CONFIG_ID
        directory.mkdir(parents=True, exist_ok=True)
        original = directory.parent / "original.bin"
        reused, computed, failed = [], [], []
        with self._connect() as db:
            # Single writer guards cache creation and per-group commits across processes.
            db.execute("BEGIN IMMEDIATE")
            if not original.exists():
                _atomic(original, lambda stream: stream.write(content))
            spectrum_path = directory / "power.npy"
            power = None
            for group in requested:
                exists = db.execute("SELECT payload FROM features WHERE image_id=? AND config_id=? AND group_name=?", (image_id, CONFIG_ID, group)).fetchone()
                if exists and json.loads(exists[0])["status"] == "completed":
                    reused.append(group)
                    continue
                started = time.perf_counter()
                try:
                    spectrum_reused = spectrum_path.exists()
                    if power is None:
                        if spectrum_reused:
                            power = np.load(spectrum_path, allow_pickle=False)
                            if power.shape != rgb.shape or not np.isfinite(power).all() or (power < 0).any():
                                raise ValueError("Invalid cached power spectrum")
                        else:
                            power = np.abs(np.fft.fft2(rgb, axes=(0, 1), norm="ortho"))**2
                            _atomic(spectrum_path, lambda stream: np.save(stream, power, allow_pickle=False))
                    payload = {"status": "completed", "values": _extract(group, power),
                               "completed_at": _now(), "duration_ms": round((time.perf_counter()-started)*1000, 3),
                               "spectrum_reused": spectrum_reused}
                    computed.append(group)
                except Exception as error:
                    payload = {"status": "failed", "values": None, "error": str(error), "completed_at": _now()}
                    failed.append(group)
                db.execute("INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?)",
                           (image_id, CONFIG_ID, group, json.dumps(payload, allow_nan=False)))
                db.commit()  # Completed work survives a later failure or interrupted request.
                db.execute("BEGIN IMMEDIATE")
            report = self._report(db, image_id, filename, source_format, original_mode, source_size, image.size)
        return {"report": report, "cache_path": str(directory.resolve()),
                "computed_groups": computed, "reused_groups": reused, "failed_groups": failed}

    def _report(self, db, image_id, filename, image_format, mode, source_size, analysis_size):
        entries = dict(db.execute("SELECT group_name,payload FROM features WHERE image_id=? AND config_id=?", (image_id, CONFIG_ID)))
        report = {"schema_version": 1, "image_id": image_id, "config_id": CONFIG_ID,
                  "image": {"filename": filename, "format": image_format, "original_mode": mode,
                            "original_size": list(source_size), "analysis_size": list(analysis_size)},
                  "configuration": CONFIG,
                  "groups": {name: json.loads(entries[name]) if name in entries else {"status": "not_computed", "values": None} for name in GROUPS}}
        return report


def de_five_values(power, include_score=True):
    """Input is the existing orthonormal RGB power spectrum, including DC."""
    height, width, _ = power.shape
    # FFT bin distances, not the original extractor's normalized radial bands.
    yy = np.fft.fftfreq(height) * height
    xx = np.fft.fftfreq(width) * width
    low = np.hypot(yy[:, None], xx[None, :]) <= 8
    # Undo norm='ortho' to match the unnormalized DFT in the paper.
    raw_power = power * (height * width)
    channels = []
    for index in range(3):
        spectrum = raw_power[:, :, index]
        ratio = float(spectrum[~low].sum() / (spectrum[low].sum() + 1e-8))
        magnitude = np.sqrt(spectrum)
        total = float(magnitude.sum())
        if total:
            q = magnitude[magnitude > 0] / total
            entropy = float(np.sum(q * np.log(q)))
        else:
            entropy = 0.0  # continuous zero-signal convention; avoids NaN
        component = {"high_low_ratio": ratio, "entropy_surrogate": entropy}
        if include_score:
            component["score"] = (ratio + entropy) / 2
        channels.append(component)
    result = {"channels": channels, "low_radius_bins": 8, "epsilon": 1e-8,
            "rgb_aggregation": "mean of channel scores", "dc_included": True}
    if include_score:
        result["score"] = float(np.mean([c["score"] for c in channels]))
    else:
        result["rgb_aggregation"] = "none; separate components"
    return result
