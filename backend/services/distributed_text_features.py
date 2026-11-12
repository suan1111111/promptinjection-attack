"""Independent cross-box rule features using XYLayoutLM inference-time XY Cut.

Reading order: https://arxiv.org/html/2203.06947v1#S3.SS3
Cross-box windows and numerical aggregation are project extensions.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
from contextlib import contextmanager

from PIL import Image, ImageOps
from .snapguard_text_matching import detect_malicious_actions

CATEGORIES = ('position_click', 'password_input', 'link_invitation', 'control_instruction', 'credential_request')
SOURCE = 'https://arxiv.org/html/2203.06947v1#S3.SS3'
VERSION = 'xycut-inclusive-yfirst-fallback-yx-minimal-cross-box-v1'
FEATURE_VERSION = 'cross-box-structure-v2-unique-spans-pair-weighted-gaps'
RULE_HASH = hashlib.sha256(Path(__file__).with_name('snapguard_text_matching.py').read_bytes()).hexdigest()


def _validate(box, width, height):
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        raise ValueError('Invalid text box')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Text boxes must match the EXIF-oriented original image')


def parse_blocks(raw, format, width, height):
    """Preserve text/box association; old aggregate OCR cache cannot supply it."""
    blocks = []
    if format == 'tsv':
        rows = csv.DictReader(io.StringIO(raw), delimiter='\t')
        if not {'left', 'top', 'width', 'height', 'text', 'level'}.issubset(rows.fieldnames or []):
            raise ValueError('Expected word-level Tesseract TSV')
        for row in rows:
            if int(row['level']) != 5 or not row['text'].strip():
                continue
            x, y, w, h = [int(row[key]) for key in ('left', 'top', 'width', 'height')]
            box = [x, y, x + w, y + h]
            _validate(box, width, height)
            blocks.append({'box': box, 'text': row['text'].strip()})
    elif format == 'grounding':
        matches = list(re.finditer(r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>', raw, re.S))
        if not matches:
            raise ValueError('Grounded OCR has no boxes; plain text is insufficient')
        labels = {'text', 'title', 'header', 'footer', 'paragraph', 'caption', 'list', 'equation', 'table'}
        for i, match in enumerate(matches):
            if match[1].strip().lower() not in labels:
                continue
            coordinates = json.loads(match[2])
            if len(coordinates) == 4 and all(isinstance(v, (int, float)) for v in coordinates):
                coordinates = [coordinates]
            text = raw[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(raw)].strip()
            if not text:
                continue
            if len(coordinates) != 1:
                raise ValueError('Ambiguous OCR: one text segment maps to multiple boxes; provide word TSV')
            box = coordinates[0]
            _validate(box, 999, 999)
            box = [int(box[0] * width / 999), int(box[1] * height / 999),
                   int(box[2] * width / 999), int(box[3] * height / 999)]
            _validate(box, width, height)
            blocks.append({'box': box, 'text': text})
        if not blocks:
            raise ValueError('Grounded OCR has no nonempty text boxes')
    else:
        raise ValueError('Unknown OCR format')
    return blocks


def xy_cut_order(blocks):
    """Paper §3.3: zero-projection valleys, alternate Y/X, no augmentation.

    Closed integer intervals follow paper equations 1/4. Interval unions find
    the same zero-projection valleys without allocating image-sized profiles.
    Unsplit leaves use ascending Y then X; ties use original index.
    """
    boxes = [block['box'] for block in blocks]

    def clusters(ids, axis):
        ordered = sorted(ids, key=lambda i: (boxes[i][axis], boxes[i][axis + 2], i))
        groups, current, end = [], [], None
        for i in ordered:
            start, stop = boxes[i][axis], boxes[i][axis + 2]
            if end is not None and start > end + 1:
                groups.append(current)
                current = []
            end = stop if not current else max(end, stop)
            current.append(i)
        if current:
            groups.append(current)
        return groups

    # Explicit stack avoids recursion failure for adversarially deep layouts.
    pending, output = [(list(range(len(blocks))), 1, False)], []
    while pending:
        ids, axis, other_failed = pending.pop()
        if len(ids) <= 1:
            output.extend(ids)
            continue
        groups = clusters(ids, axis)
        if len(groups) > 1:
            pending.extend((group, 1 - axis, False) for group in reversed(groups))
        elif not other_failed:
            pending.append((ids, 1 - axis, True))
        else:
            output.extend(sorted(ids, key=lambda i: (boxes[i][1], boxes[i][0], i)))
    return output


def _hits(text):
    return {hit['action_type'] for hit in detect_malicious_actions(' '.join(text.split()))}


def cross_box_features(blocks, order, max_blocks=4, *, image_size):
    """Count minimal contiguous spans that need multiple OCR boxes to match.

    Exclude a category if any individual box already matches it. Deduplicate
    windows containing an already accepted shorter span of the same category.
    This is a rule-based candidate signal, not a semantic attack verdict.
    """
    if not isinstance(max_blocks, int) or not 2 <= max_blocks <= 32:
        raise ValueError('max_blocks must be between 2 and 32')
    if sorted(order) != list(range(len(blocks))):
        raise ValueError('Reading order is not a permutation of OCR boxes')
    width, height = image_size
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in image_size):
        raise ValueError('Image dimensions must be positive and finite')
    for block in blocks:
        _validate(block['box'], width, height)
    texts = [blocks[i]['text'] for i in order]
    baseline = [_hits(text) for text in texts]
    evidence = []
    for length in range(2, min(max_blocks, len(texts)) + 1):
        for start in range(len(texts) - length + 1):
            stop = start + length
            hits = _hits(' '.join(texts[start:stop])) - set().union(*baseline[start:stop])
            for category in sorted(hits):
                if any(item['category'] == category and start <= item['start'] and item['stop'] <= stop for item in evidence):
                    continue
                evidence.append({'category': category, 'start': start, 'stop': stop,
                                 'box_ids': order[start:stop]})
    values = structure_features(blocks, evidence, image_size)
    return values, evidence


def structure_features(blocks, evidence, image_size):
    """Five project-defined statistics; geometry counts identical spans once.

    Mean gap weights every adjacent pair in each distinct matched span equally.
    Match count retains category-hit semantics; geometry does not overweight a
    fragment solely because it matches multiple rule categories.
    """
    values = {'cross_box_match_count': len(evidence), 'mean_combination_box_count': None,
              'max_combination_box_count': None, 'mean_box_gap_normalized': None,
              'max_spatial_span_normalized': None}
    if not evidence:
        return values
    spans = {tuple(item['box_ids']) for item in evidence}
    diagonal = math.hypot(*image_size)
    counts, gaps, distances = [], [], []
    for ids in spans:
        boxes = [blocks[i]['box'] for i in ids]
        counts.append(len(boxes))
        for a, b in zip(boxes, boxes[1:]):
            dx = max(0, a[0] - b[2], b[0] - a[2])
            dy = max(0, a[1] - b[3], b[1] - a[3])
            gaps.append(math.hypot(dx, dy) / diagonal)
        centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]
        for i, a in enumerate(centers):
            distances.extend(math.hypot(a[0] - b[0], a[1] - b[1]) / diagonal for b in centers[i + 1:])
    values.update(mean_combination_box_count=sum(counts) / len(counts),
                  max_combination_box_count=max(counts),
                  mean_box_gap_normalized=sum(gaps) / len(gaps),
                  max_spatial_span_normalized=max(distances))
    return values


class DistributedTextStore:
    """Own database; raw OCR, layout, ordering and features commit separately."""
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / 'data' / 'image_features'
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'distributed_text.sqlite3'
        with self.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS stages (image_id TEXT, config_id TEXT, stage TEXT, payload TEXT, PRIMARY KEY(image_id,config_id,stage))')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=120)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, image_id, config_id, stage):
        with self.connection() as db:
            row = db.execute('SELECT payload FROM stages WHERE image_id=? AND config_id=? AND stage=?', (image_id, config_id, stage)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, image_id, config_id, stage, payload):
        with self.connection() as db:
            db.execute('INSERT OR REPLACE INTO stages VALUES (?,?,?,?)', (image_id, config_id, stage, json.dumps(payload, allow_nan=False)))

    def extract(self, image_path, raw=None, format='grounding', ocr=None, max_blocks=4):
        if not isinstance(max_blocks, int) or not 2 <= max_blocks <= 32:
            raise ValueError('max_blocks must be between 2 and 32')
        content = Path(image_path).read_bytes()
        image_id = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as source:
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Multi-frame images are not supported')
            width, height = ImageOps.exif_transpose(source).size
        if raw is None:
            cached = self.get(image_id, 'ocr-input-v1', 'raw')
            if cached:
                raw, format = cached['raw'], cached['format']
            elif ocr:
                raw, format = ocr(image_path), 'grounding'
            else:
                raise ValueError('No cached per-box OCR; supply grounded output/TSV or start backend')
        if not isinstance(raw, str):
            raise ValueError('OCR response must be text')
        # Even a later parse/order failure leaves the original OCR response saved.
        self.put(image_id, 'ocr-input-v1', 'raw', {'raw': raw, 'format': format})
        input_id = hashlib.sha256((format + '\0' + raw).encode()).hexdigest()
        layout_id = hashlib.sha256((VERSION + input_id).encode()).hexdigest()
        layout = self.get(image_id, layout_id, 'layout')
        if layout is None:
            layout = {'blocks': parse_blocks(raw, format, width, height)}
            self.put(image_id, layout_id, 'layout', layout)
        ordered = self.get(image_id, layout_id, 'order')
        if ordered is None:
            ordered = {'order': xy_cut_order(layout['blocks'])}
            self.put(image_id, layout_id, 'order', ordered)
        config_id = hashlib.sha256((layout_id + RULE_HASH + str(max_blocks) + FEATURE_VERSION).encode()).hexdigest()
        cached = self.get(image_id, config_id, 'features')
        if cached is None:
            values, evidence = cross_box_features(layout['blocks'], ordered['order'], max_blocks,
                                                 image_size=(width, height))
            cached = {'values': values, 'evidence': evidence, 'configuration': {
                'version': VERSION, 'feature_version': FEATURE_VERSION, 'max_blocks': max_blocks, 'source': SOURCE,
                'rule_hash': RULE_HASH, 'input_id': input_id}}
            self.put(image_id, config_id, 'features', cached)
        return cached['values']
