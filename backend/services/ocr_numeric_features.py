"""Numerical OCR quality, box geometry, contrast and cached response overlap."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps

from .frequency_features import FrequencyFeatureStore, _now
from .gradient_features import GradientFeatureStore, CONFIG_ID as GRADIENT_ID
from .entropy_features import EntropyFeatureStore, CONFIG_ID as ENTROPY_ID

VERSION = 'ocr-numeric-v1-edge5-ring3-pillow-gray-' + GRADIENT_ID + '-' + ENTROPY_ID
TEXT_LABELS = {'text', 'title', 'header', 'footer', 'paragraph', 'caption', 'list', 'equation', 'table'}


def parse_grounding(text, width, height):
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = list(re.finditer(pattern, text, re.S))
    if not matches:
        raise ValueError('OCR has no grounding boxes; provide grounded OCR output or word TSV')
    boxes, texts = [], []
    for i, match in enumerate(matches):
        label = match[1].strip().lower()
        if label not in TEXT_LABELS:
            continue
        coordinates = json.loads(match[2])  # Never execute model-generated expressions.
        if len(coordinates) == 4 and all(isinstance(v, (int, float)) for v in coordinates):
            coordinates = [coordinates]
        segment = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
        texts.append(segment)
        for box in coordinates:
            if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 999 for v in box):
                raise ValueError('Invalid OCR coordinates; expected [x1,y1,x2,y2] in 0..999')
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError('OCR box has non-positive area')
            boxes.append({'box': [int(x1 / 999 * width), int(y1 / 999 * height),
                                  int(x2 / 999 * width), int(y2 / 999 * height)],
                          'confidence': None})
    return {'text': '\n'.join(texts), 'boxes': boxes, 'box_granularity': 'layout block', 'source': 'DeepSeek grounding'}


def parse_tsv(text, width, height):
    rows = csv.DictReader(io.StringIO(text), delimiter='\t')
    required = {'left', 'top', 'width', 'height', 'conf', 'text', 'level'}
    if not required.issubset(rows.fieldnames or []):
        raise ValueError('Expected Tesseract word TSV columns: ' + ','.join(sorted(required)))
    boxes, words = [], []
    for row in rows:
        if int(row['level']) != 5 or not row['text'].strip():
            continue
        x, y, w, h = [int(row[key]) for key in ('left', 'top', 'width', 'height')]
        confidence = float(row['conf'])
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError('TSV boxes must use pixels of the EXIF-oriented original image')
        if not math.isfinite(confidence) or not -1 <= confidence <= 100:
            raise ValueError('Invalid TSV confidence')
        boxes.append({'box': [x, y, x + w, y + h], 'confidence': confidence / 100 if confidence >= 0 else None})
        words.append(row['text'])
    return {'text': ' '.join(words), 'boxes': boxes, 'box_granularity': 'word', 'source': 'Tesseract TSV'}


def box_mask(boxes, shape):
    mask = np.zeros(shape, dtype=bool)
    for record in boxes:
        x1, y1, x2, y2 = record['box']
        if not (0 <= x1 < x2 <= shape[1] and 0 <= y1 < y2 <= shape[0]):
            raise ValueError('OCR box became empty or falls outside the analysis image')
        mask[y1:y2, x1:x2] = True
    return mask


def quality(data):
    scores = [r['confidence'] for r in data['boxes'] if r['confidence'] is not None]
    return {'character_count': sum(not c.isspace() for c in data['text']),
            'text_box_count': len(data['boxes']), 'confidence_available_count': len(scores),
            'confidence_mean': float(np.mean(scores)) if scores else None,
            'confidence_min': float(min(scores)) if scores else None}


def geometry(data, gray):
    height, width = gray.shape
    mask = box_mask(data['boxes'], gray.shape)
    heights, edge_flags, contrasts = [], [], []
    for record in data['boxes']:
        x1, y1, x2, y2 = record['box']
        heights.append((y2 - y1) / height)
        edge_flags.append(x1 < width * .05 or y1 < height * .05 or x2 > width * .95 or y2 > height * .95)
        bx1, by1, bx2, by2 = max(0, x1 - 3), max(0, y1 - 3), min(width, x2 + 3), min(height, y2 + 3)
        ring = ~mask[by1:by2, bx1:bx2]
        surrounding = gray[by1:by2, bx1:bx2][ring]
        if surrounding.size:
            # Proxy: largest grayscale deviation from surrounding median.
            inside = gray[y1:y2, x1:x2]
            contrasts.append(float(np.max(np.abs(inside - np.median(surrounding)))))
    return {'text_area_ratio': float(mask.mean()),
            'relative_box_height_mean': float(np.mean(heights)) if heights else None,
            'relative_box_height_max': max(heights) if heights else None,
            'edge_text_box_ratio': float(np.mean(edge_flags)) if edge_flags else 0.0,
            'box_background_contrast_mean': float(np.mean(contrasts)) if contrasts else None}


def overlap(text, response, prefix):
    if text.shape != response.shape:
        raise ValueError('Cached response dimensions do not match OCR image')
    intersection = np.count_nonzero(text & response)
    union = np.count_nonzero(text | response)
    return {prefix + '_text_coverage': intersection / np.count_nonzero(text) if text.any() else None,
            prefix + '_response_in_text_ratio': intersection / np.count_nonzero(response) if response.any() else None,
            prefix + '_iou': intersection / union if union else None}


class OCRNumericStore(FrequencyFeatureStore):
    def _get(self, image_id, config_id, group):
        with self._connect() as db:
            row = db.execute('SELECT payload FROM features WHERE image_id=? AND config_id=? AND group_name=?', (image_id, config_id, group)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, image_id, config_id, group, value):
        with self._connect() as db:
            db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?)', (image_id, config_id, group, json.dumps(value, allow_nan=False)))

    def extract(self, image_path, raw=None, format='grounding', ocr=None):
        content = Path(image_path).read_bytes()
        image_id = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as source:
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Multi-frame images are not supported')
            rgba = ImageOps.exif_transpose(source).convert('RGBA')
            image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('L')
        gray = np.asarray(image, dtype=float) / 255
        record = self._get(image_id, 'ocr-layout-v1', 'ocr_layout') if raw is None else None
        if record:
            data = record['data']
        else:
            if raw is None:
                if not ocr:
                    raise ValueError('No cached OCR layout; supply structured OCR or start backend')
                raw, format = ocr(image_path), 'grounding'
            data = parse_tsv(raw, *image.size) if format == 'tsv' else parse_grounding(raw, *image.size)
            box_mask(data['boxes'], gray.shape)
            # Save structured OCR before any subsequent image statistic can fail.
            self._put(image_id, 'ocr-layout-v1', 'ocr_layout', {'data': data, 'completed_at': _now()})
        config_id = hashlib.sha256((VERSION + json.dumps(data, sort_keys=True)).encode()).hexdigest()[:16]
        output = {}
        computations = [('ocr_quality', lambda: quality(data)), ('text_geometry', lambda: geometry(data, gray)),
                        ('text_gradient_overlap', lambda: self._gradient(image_path, image_id, data, gray.shape)),
                        ('text_entropy_overlap', lambda: self._entropy(image_path, image_id, data, gray.shape))]
        for group, computation in computations:
            cached = self._get(image_id, config_id, group)
            if cached and cached.get('status') == 'completed':
                output.update(cached['values'])
                continue
            try:
                values = computation()
                self._put(image_id, config_id, group, {'status': 'completed', 'values': values, 'completed_at': _now()})
                output.update(values)
            except Exception as error:
                self._put(image_id, config_id, group, {'status': 'failed', 'error': str(error), 'completed_at': _now()})
                raise
        return output

    def _gradient(self, path, image_id, data, shape):
        GradientFeatureStore(self.root).extract(path)
        response = np.zeros(shape, dtype=bool)
        with np.load(self.root / image_id / GRADIENT_ID / 'windows.npz', allow_pickle=False) as saved:
            for i, j in np.argwhere(saved['selected_windows']):
                y, x = int(saved['window_y'][i]), int(saved['window_x'][j])
                response[y:y + int(saved['window_height']), x:x + int(saved['window_width'])] = True
        return overlap(box_mask(data['boxes'], shape), response, 'gradient')

    def _entropy(self, path, image_id, data, shape):
        EntropyFeatureStore(self.root).extract(path)
        response = np.load(self.root / image_id / ENTROPY_ID / 'candidate_mask.npy', allow_pickle=False)
        return overlap(box_mask(data['boxes'], shape), response, 'entropy')
