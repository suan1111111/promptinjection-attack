"""Request-scoped computation adapters. Persistent shared caching comes later."""
import io
import time
import numpy as np
from PIL import Image, ImageOps
from .detection_catalog import get_operation
from .ocr_numeric_features import parse_grounding, parse_tsv, box_mask, geometry_only, contrast_only, overlap
from .gradient_features import compute_gradient
from .text_rule_features import extract_text_rules


class ComputationSession:
    def __init__(self, content, raw=None, format='grounding', ocr_provider=None):
        if format not in ('grounding', 'tsv'):
            raise ValueError('format must be grounding or tsv')
        with Image.open(io.BytesIO(content)) as source:
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Multi-frame images are not supported')
            rgba = ImageOps.exif_transpose(source).convert('RGBA')
            self.image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
        self.raw, self.format, self.ocr_provider = raw, format, ocr_provider
        self.results, self.events = {}, []

    def compute(self, identifier):
        operation = get_operation(identifier)
        if operation is None:
            raise ValueError('Unknown operation: ' + identifier)
        if identifier in self.results:
            return self.results[identifier]
        for dependency in operation['dependencies']:
            self.compute(dependency)
        start = time.perf_counter()
        value = self._execute(identifier)
        self.results[identifier] = value
        self.events.append({'operation_id': identifier, 'status': 'completed',
                            'elapsed_ms': (time.perf_counter() - start) * 1000})
        return value

    def _execute(self, identifier):
        if identifier == 'ocr.structured':
            raw = self.raw
            output_format = self.format
            if raw is None:
                if self.ocr_provider is None:
                    raise ValueError('Structured OCR input required; supply ocr_input or configure an OCR provider')
                raw = self.ocr_provider(self.image)
                output_format = getattr(self.ocr_provider, 'output_format', self.format)
            parser = parse_tsv if output_format == 'tsv' else parse_grounding
            result = parser(raw, *self.image.size)
            box_mask(result['boxes'], (self.image.height, self.image.width))
            return {**result, 'image_size': list(self.image.size)}
        if identifier == 'image.gray':
            return {'gray': np.asarray(self.image.convert('L'), dtype=float) / 255}
        data = self.results.get('ocr.structured')
        if identifier == 'text.rules':
            return extract_text_rules(data['text'])
        if identifier == 'text.geometry':
            return geometry_only(data, (self.image.height, self.image.width))
        if identifier == 'text.contrast':
            return contrast_only(data, self.results['image.gray']['gray'])
        if identifier == 'image.gradient':
            summary, windows = compute_gradient(self.results['image.gray']['gray'])
            return {**summary, 'windows': windows}
        if identifier == 'text.gradient_overlap':
            windows = self.results['image.gradient']['windows']
            response = np.zeros((self.image.height, self.image.width), dtype=bool)
            for i, j in np.argwhere(windows['selected_windows']):
                y, x = int(windows['window_y'][i]), int(windows['window_x'][j])
                response[y:y + int(windows['window_height']), x:x + int(windows['window_width'])] = True
            return overlap(box_mask(data['boxes'], response.shape), response, 'gradient')
        raise ValueError('Unsupported operation: ' + identifier)


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value
