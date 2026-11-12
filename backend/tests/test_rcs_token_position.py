"""Padding cases that differ from upstream's unconditional last position."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_rcs_hidden_state import last_valid_token_index


class LastValidTokenTests(unittest.TestCase):
    def test_padding_and_mask_positions(self):
        for mask, expected in [
            ([1, 1, 0, 0], 1),
            ([0, 0, 1, 1], 3),
            ([1, 1, 1], 2),
            ([1, 0, 1, 0], 2),
        ]:
            with self.subTest(mask=mask):
                self.assertEqual(last_valid_token_index(torch.tensor([mask])), expected)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            last_valid_token_index(torch.tensor([[0, 0]]))


if __name__ == "__main__":
    unittest.main()
