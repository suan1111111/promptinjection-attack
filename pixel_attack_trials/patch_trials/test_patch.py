"""CPU checks for patch gradient routing, support and training-data isolation."""
import csv
import tempfile
import unittest
from pathlib import Path

import torch
import train_qwen_patch as train


class PatchTests(unittest.TestCase):
    def test_gradients_and_boundaries(self):
        clean = torch.rand(1, 3, 12, 16)
        original = clean.clone()
        for x, y in [(0, 0), (12, 9), (6, 4)]:
            patch = torch.rand(1, 3, 3, 4, requires_grad=True)
            output = train.compose(clean, patch, x, y)
            weights = torch.arange(output.numel()).reshape_as(output).float()
            (output * weights).sum().backward()
            self.assertTrue(torch.equal(patch.grad, weights[:, :, y:y+3, x:x+4]))
            self.assertEqual(train.outside_difference(clean, output, x, y, 4, 3), 0)
            self.assertTrue(torch.equal(clean, original))
        with self.assertRaises(ValueError):
            train.compose(clean, patch, 13, 0)

    def test_only_train_rows_and_optional_short_target(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'rows.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['split', 'question', 'target_answer', 'short_target'])
                w.writeheader()
                w.writerows([
                    dict(split='test', question='do not train', target_answer='held out'),
                    dict(split='train', question='question', target_answer='full', short_target='short'),
                    dict(split='val', question='do not train', target_answer='held out'),
                ])
            self.assertEqual(len(train.read_training_rows(path, 'full', 0)), 1)
            self.assertEqual(train.read_training_rows(path, 'short', 0)[0]['training_target'], 'short')


if __name__ == '__main__':
    unittest.main()
