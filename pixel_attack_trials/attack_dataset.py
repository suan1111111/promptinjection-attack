"""Read native Qwen CSVs or image-matched Soft-Prompts URL-injection CSVs.

Native Soft-Prompts rows 1-40 train, 41-50 validate, and 51+ test.
The original questions and complete injection answers are preserved verbatim.
"""
import csv
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if {'split', 'question'}.issubset(fields):
        return rows
    if not {'Question', 'normal', 'injection'}.issubset(fields):
        raise ValueError('Expected split/question CSV or Soft-Prompts Question/normal/injection CSV')
    if len(rows) < 51:
        raise ValueError('Soft-Prompts requires at least 51 rows for 40 train / 10 val / remaining test')
    converted = []
    for index, row in enumerate(rows):
        question, target = row['Question'].strip(), row['injection'].strip()
        if not question or not target:
            raise ValueError(f'Empty question/injection on CSV line {index + 2}')
        converted.append(dict(split='train' if index < 40 else 'val' if index < 50 else 'test',
            category='injection', question=question, normal_answer=row['normal'],
            target_answer=target, target_contains='<maliciousurl>' if '<maliciousurl>' in target else '',
            source_csv_line=str(index + 2)))
    return converted
