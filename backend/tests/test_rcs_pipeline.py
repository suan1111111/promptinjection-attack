"""CPU tests: upstream equivalence, caching/retries, labels, projection round trip."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcs_features import (PROJECT_ROOT, load_feature, parse_layers, read_manifest, run_batch)
from rcs_projection import load_projection, project_vectors, train_projection, training_arrays
from rcs_projection_core import LearnedProjection, compute_contrastive_loss


class FakeExtractor:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def extract(self, row, layers):
        self.calls.append(row["id"])
        if row["id"] == self.fail:
            raise RuntimeError("deliberate test failure")
        seed = sum(row["id"].encode())
        return np.random.default_rng(seed).normal(size=(len(layers), 2048)).astype(np.float32), {
            "token_index": 9, "token_id": 198, "token_text": "\n", "input_token_count": 10}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        root = PROJECT_ROOT / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="rcs_test_", dir=root)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "inputs.json"
        self.output = self.root / "features"
        image = str(PROJECT_ROOT / "picture/medical-page-09.png")
        self.rows = [{"id": f"s{i}", "image": image, "question": f"test question {i}",
                      "label": i % 2, "dataset": f"source_{i%2}", "split": "train"}
                     for i in range(4)]
        self.rows.append({"id": "heldout", "image": image, "question": "heldout question",
                          "label": None, "dataset": None, "split": "test"})
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.manifest.write_text(json.dumps(self.rows), encoding="utf-8")

    def test_retry_skip_change_and_corruption(self):
        fake = FakeExtractor(fail="s1")
        self.assertEqual(run_batch(self.manifest, self.output, [0, 18, 36], extractor=fake), 1)
        fake = FakeExtractor()
        self.assertEqual(run_batch(self.manifest, self.output, [0, 18, 36], extractor=fake), 0)
        self.assertEqual(fake.calls, ["s1"])
        index = json.loads((self.output / "index.json").read_text(encoding="utf-8"))
        vectors, meta = load_feature(self.output / index["records"][0]["file"])
        self.assertEqual(vectors.shape, (3, 2048))
        self.assertEqual(meta["token_index"], 9)
        self.rows[0]["question"] = "changed"
        self.write()
        (self.output / index["records"][2]["file"]).write_bytes(b"corrupt")
        fake = FakeExtractor()
        run_batch(self.manifest, self.output, [0, 18, 36], extractor=fake)
        self.assertEqual(fake.calls, ["s0", "s2"])
        with self.assertRaises(ValueError):
            run_batch(self.manifest, self.output, [18], extractor=fake)

    def test_train_load_and_label_guards(self):
        run_batch(self.manifest, self.output, [18], extractor=FakeExtractor())
        x, d, y, mapping, _, _ = training_arrays(self.output, 18)
        self.assertEqual(x.shape, (4, 2048))  # excludes held-out unlabelled row
        model, metadata = train_projection(self.output, self.root / "projection.pt", 18,
                                           epochs=2, batch_size=4, device="cpu")
        loaded, restored = load_projection(self.root / "projection.pt")
        np.testing.assert_allclose(project_vectors(model, x), project_vectors(loaded, x), atol=1e-6)
        self.assertEqual(project_vectors(loaded, x[:1]).shape, (1, 256))
        self.assertEqual(restored["layer"], 18)
        self.rows[0]["label"] = None
        self.write()
        run_batch(self.manifest, self.output, [18], extractor=FakeExtractor())
        with self.assertRaisesRegex(ValueError, "缺少"):
            training_arrays(self.output, 18)

    def test_upstream_definitions_unchanged(self):
        names = {"ProjectionConfig", "LearnedProjection", "compute_contrastive_loss"}
        base = PROJECT_ROOT / "backend"
        def definitions(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names}
        self.assertEqual(definitions(base / "third_party/rcs/kcd.py.txt"),
                         definitions(base / "rcs_projection_core.py"))
        model = LearnedProjection(input_dim=2048)
        x = torch.randn(4, 2048)
        total, source, safety = compute_contrastive_loss(model(x), torch.tensor([0,0,1,1]),
                                                        torch.tensor([0,0,1,1]))
        torch.testing.assert_close(total, source + 5 * safety)
        total.backward()
        self.assertTrue(torch.isfinite(model.projection[0].weight.grad).all())

    def test_manifest_and_layers(self):
        self.assertEqual(len(parse_layers("all")), 37)
        self.assertEqual(parse_layers("18,12,18"), [12,18])
        self.rows[0]["label"] = None
        self.write()
        self.assertIsNone(read_manifest(self.manifest)[0]["label"])
        self.rows[1]["id"] = self.rows[0]["id"]
        self.write()
        with self.assertRaises(ValueError):
            read_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
