"""Read existing attack outputs and compare existing CPU image features.

No attack training, model inference, thresholds, or detector fitting is performed.
All feature formulae are imported from the project's existing implementations.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'backend'))

import numpy as np
from PIL import Image, ImageOps
from services.gradient_features import compute_gradient
from services.entropy_features import window_sizes, compute_scale, filter_candidates
from services.residual_features import compute_residual, summarize
from services.frequency_features import de_five_values


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


memo = {}


def features(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in memo:
        return memo[digest]
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
        rgb_image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
    rgb = np.asarray(rgb_image, dtype=np.float64)
    gray = np.asarray(rgb_image.convert('L'), dtype=np.float64)
    values, _ = compute_gradient(gray / 255)
    entropy_gray = np.floor(rgb @ np.array([.298936021293775, .587043074451121, .114020904255103]) + .5).clip(0, 255).astype(np.uint8)
    masks = []
    for scale, size in enumerate(window_sizes(entropy_gray.shape)):
        summary, details = compute_scale(entropy_gray, size, scale)
        values.update({f'scale_{scale+1}_{key}': value for key, value in summary.items()})
        masks.append(details['mask'])
    summary, _ = filter_candidates(np.logical_or.reduce(masks))
    values.update(summary)
    for kind in ('kb', 'kv'):
        values.update(summarize(compute_residual(gray, kind), kind))
    power = np.abs(np.fft.fft2(rgb / 255, axes=(0, 1), norm='ortho')) ** 2
    for channel, component in zip(('red', 'green', 'blue'), de_five_values(power, include_score=False)['channels']):
        values.update({f'{channel}_de_five_{key}': value for key, value in component.items()})
    result = dict(sha256=digest, size=list(rgb_image.size), values=values)
    memo[digest] = result
    return result


cases = []
seven = ROOT / 'outputs/seven_attack_revised_20261027'
for directory in sorted(seven.iterdir()):
    if not directory.is_dir() or not directory.name[:2].isdigit():
        continue
    evaluations = list(directory.glob('*/evaluation.json'))
    if len(evaluations) != 1:
        continue
    ev = evaluations[0]
    run = ev.parent
    cases.append((directory.name, run / 'clean_resized.png', run / 'adversarial_final.png', ev))

scene = ROOT / 'outputs/scene_region_top10_20261103/jobs'
for job in sorted(scene.iterdir()):
    ev = job / 'test_evaluation.json'
    if not ev.exists():
        continue
    # The evaluation uses the validation-selected image, not necessarily final step.
    attack = job / 'best_validation.png'
    cases.append(('scene_' + job.name, job / 'run/clean_resized.png', attack, ev))

records = []
for name, clean_path, attack_path, evaluation_path in cases:
    evaluation = read(evaluation_path)
    rows = evaluation['records']
    clean_hits = sum('<maliciousurl>' in row.get('clean_answer', '') for row in rows)
    attack_hits = sum('<maliciousurl>' in row.get('adversarial_answer', '') for row in rows)
    clean, attack = features(clean_path), features(attack_path)
    assert clean['size'] == attack['size'], (name, clean['size'], attack['size'])
    record = dict(name=name, clean_path=str(clean_path), attack_path=str(attack_path),
                  evaluation_path=str(evaluation_path), questions=len(rows), clean_literal_hits=clean_hits,
                  attack_literal_hits=attack_hits, original_new_exact=evaluation.get('new_exact'),
                  original_new_effects=evaluation.get('new_effects'), clean=clean, attack=attack)
    records.append(record)
    print(name, f'literal {clean_hits}->{attack_hits}/{len(rows)}', flush=True)

report = dict(date='2026-11-04', scope='Existing saved PNGs, CPU feature comparison only.',
              notice='Paired changes are diagnostic observations, not deployable thresholds or detection rates. Clean counterparts are unavailable for ordinary unknown inputs.',
              records=records)
(OUT / 'feature_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
columns = ['max_window_gradient', 'high_gradient_area_ratio', 'candidate_area_ratio',
           'kb_residual_energy', 'kv_residual_energy', 'red_de_five_high_low_ratio',
           'red_de_five_entropy_surrogate']
with (OUT / 'feature_comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=['case', 'image', 'literal_hits', 'questions'] + columns)
    writer.writeheader()
    for record in records:
        for side in ('clean', 'attack'):
            row = dict(case=record['name'], image=side, literal_hits=record[f'{side}_literal_hits'], questions=record['questions'])
            row.update({key: record[side]['values'][key] for key in columns})
            writer.writerow(row)
for record in records:
    a, b = record['clean']['values'], record['attack']['values']
    print(record['name'], '; '.join(f'{key} {a[key]:.5g}->{b[key]:.5g}' for key in columns), flush=True)
