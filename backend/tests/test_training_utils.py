"""Tests for the projection-training batch guard."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcs_training_utils import ensure_full_batch


class TrainingGuardTests(unittest.TestCase):
    def test_rejects_fewer_samples_than_batch(self):
        with self.assertRaises(ValueError):
            ensure_full_batch(4, 64)

    def test_accepts_exactly_one_full_batch(self):
        ensure_full_batch(64, 64)

    def test_rejects_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            ensure_full_batch(10, 1)


if __name__ == "__main__":
    unittest.main()
