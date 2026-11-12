import csv
import tempfile
import unittest
from pathlib import Path

from attack_dataset import read_rows


class DatasetTests(unittest.TestCase):
    def test_soft_prompts_split_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['Question', 'normal', 'injection', 'spam'])
                writer.writeheader()
                for i in range(60):
                    writer.writerow(dict(Question=f'q{i}', normal=f'n{i}',
                        injection=f'a{i} <maliciousurl>', spam='wrong target'))
            rows = read_rows(path)
            self.assertEqual([sum(r['split'] == s for r in rows) for s in ['train','val','test']], [40,10,10])
            self.assertEqual(rows[39]['question'], 'q39')
            self.assertEqual(rows[50]['target_answer'], 'a50 <maliciousurl>')
            self.assertEqual(rows[50]['source_csv_line'], '52')
            self.assertEqual(rows[50]['target_contains'], '<maliciousurl>')

    def test_existing_schema_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.csv'
            path.write_text('split,question,target_answer\ntrain,q,answer\n', encoding='utf-8')
            self.assertEqual(read_rows(path), [dict(split='train', question='q', target_answer='answer')])


if __name__ == '__main__':
    unittest.main()
