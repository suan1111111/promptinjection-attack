import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.frequency_features import FrequencyFeatureStore, de_five_values
from services.frequency_api import frequency_bp
from flask import Flask


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class FrequencyTests(unittest.TestCase):
    def test_de_five_matches_direct_dft(self):
        array = np.random.default_rng(9).random((23, 27, 3))
        raw = np.abs(np.fft.fft2(array, axes=(0, 1)))
        yy = np.minimum(np.arange(23), 23 - np.arange(23))
        xx = np.minimum(np.arange(27), 27 - np.arange(27))
        low = yy[:, None] ** 2 + xx[None, :] ** 2 <= 64
        expected = []
        for i in range(3):
            m = raw[:, :, i]
            q = m / m.sum()
            expected.append((np.sum(m[~low] ** 2) / (np.sum(m[low] ** 2) + 1e-8)
                             + np.sum(q * np.log(q))) / 2)
        result = de_five_values(np.abs(np.fft.fft2(array, axes=(0, 1), norm="ortho")) ** 2)
        self.assertAlmostEqual(result["score"], np.mean(expected), places=12)

    def test_de_five_cache_and_cli_components(self):
        import subprocess
        path = Path(self.temp.name) / "image.png"
        path.write_bytes(png(np.random.default_rng(8).integers(0, 256, (25, 27, 3), dtype=np.uint8)))
        first = self.store.extract_de_five(path)
        with patch("numpy.fft.fft2", side_effect=AssertionError("cached")):
            self.assertEqual(first, FrequencyFeatureStore(self.temp.name).extract_de_five(path))
        output = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "extract_frequency.py"),
                                 str(path), "--store", self.temp.name], capture_output=True, text=True, check=True)
        values = dict(line.split(': ', 1) for line in output.stdout.splitlines())
        self.assertEqual(len(values), 6)
        self.assertFalse(any(key.endswith('_score') for key in values))
        components = self.store.extract_de_five_components(path)
        self.assertEqual(len(components), 6)
        self.assertAlmostEqual(float(values['red_de_five_high_low_ratio']), components['red_de_five_high_low_ratio'], places=10)
        with patch('numpy.fft.fft2', side_effect=AssertionError('cached')):
            self.assertEqual(components, self.store.extract_de_five_components(path))

    def test_components_do_not_compute_legacy_statistics(self):
        path = Path(self.temp.name) / 'components.png'
        path.write_bytes(png(np.full((19, 21, 3), 128, dtype=np.uint8)))
        with patch('services.frequency_features._extract', side_effect=AssertionError('legacy groups must not run')):
            values = self.store.extract_de_five_components(path)
        self.assertEqual(len(values), 6)
        with self.store._connect() as db:
            groups = [row[0] for row in db.execute('SELECT group_name FROM features')]
        self.assertEqual(groups, ['de_five_components'])

    def setUp(self):
        test_root = Path(__file__).resolve().parents[2] / "tmp"
        test_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.store = FrequencyFeatureStore(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_constant_has_no_ac_distribution(self):
        result = self.store.extract_bytes(png(np.full((17, 19, 3), 128, dtype=np.uint8)))
        bands = result["report"]["groups"]["bands"]["values"]["red"]
        self.assertEqual(bands["ac_energy"], 0)
        self.assertIsNone(bands["high_ratio"])
        stats = result["report"]["groups"]["statistics"]["values"]["red"]
        self.assertAlmostEqual(stats["dc_energy_ratio"], 1)

    def test_known_frequency_and_parseval(self):
        # Alternating columns concentrate AC at fx=0.5, fy=0.
        array = np.tile(np.array([0, 255], dtype=np.uint8), (16, 8))
        content = png(array)
        result = self.store.extract_bytes(content)
        groups = result["report"]["groups"]
        self.assertAlmostEqual(groups["bands"]["values"]["red"]["high_ratio"], 1)
        stats = groups["statistics"]["values"]["red"]
        self.assertAlmostEqual(stats["centroid"], 1 / np.sqrt(2))
        self.assertAlmostEqual(stats["total_energy"], float(((array / 255.)**2).sum()))

    def test_incremental_persistence_and_fft_reuse(self):
        content = png(np.random.default_rng(4).integers(0, 256, (13, 15, 3), dtype=np.uint8))
        first = self.store.extract_bytes(content, groups=["bands"])
        self.assertEqual(first["report"]["groups"]["statistics"]["status"], "not_computed")
        restarted = FrequencyFeatureStore(self.temp.name)
        with patch("numpy.fft.fft2", side_effect=AssertionError("FFT should be cached")):
            second = restarted.extract_bytes(content)
        self.assertEqual(second["reused_groups"], ["bands"])
        self.assertEqual(second["computed_groups"], ["statistics"])
        self.assertEqual(first["report"]["groups"]["bands"], second["report"]["groups"]["bands"])
        self.assertTrue(Path(second["cache_path"]).exists())
        self.assertFalse(list(Path(self.temp.name).rglob("*.json")))
        for channel in second["report"]["groups"]["bands"]["values"].values():
            self.assertAlmostEqual(sum(channel[name + "_ratio"] for name in ("low", "mid", "high")), 1)

    def test_failure_keeps_completed_and_retries(self):
        from services import frequency_features as module
        content = png(np.arange(64, dtype=np.uint8).reshape(8, 8))
        original = module._extract
        def fail_statistics(group, power):
            if group == "statistics":
                raise RuntimeError("simulated interruption")
            return original(group, power)
        with patch.object(module, "_extract", side_effect=fail_statistics):
            failed = self.store.extract_bytes(content)
        self.assertEqual(failed["failed_groups"], ["statistics"])
        recovered = self.store.extract_bytes(content)
        self.assertEqual(recovered["reused_groups"], ["bands"])
        self.assertEqual(recovered["computed_groups"], ["statistics"])

    def test_invalid_input_and_upload_api(self):
        with self.assertRaises(ValueError):
            self.store.extract_bytes(b"invalid", groups=["unknown"])
        app = Flask(__name__)
        app.register_blueprint(frequency_bp)
        with app.test_client() as client:
            self.assertEqual(client.post("/api/image-features/frequency").status_code, 400)
            with patch("services.frequency_api.FrequencyFeatureStore", return_value=self.store):
                response = client.post("/api/image-features/frequency", data={"file": (io.BytesIO(png(np.zeros((2, 2), dtype=np.uint8))), "sample.png"), "groups": "bands"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["computed_groups"], ["bands"])


if __name__ == "__main__":
    unittest.main()
