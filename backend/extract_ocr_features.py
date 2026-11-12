"""Print OCR quality, geometry and response overlap; no added UI."""
import argparse
from pathlib import Path
import sys
from extract_text_rules import request_ocr
from services.ocr_numeric_features import OCRNumericStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--ocr-input', help='Existing UTF-8 grounded output or word TSV')
    parser.add_argument('--format', choices=['grounding', 'tsv'], default='grounding')
    parser.add_argument('--store')
    parser.add_argument('--backend', default='http://127.0.0.1:5000')
    parser.add_argument('--timeout', type=float, default=1200)
    args = parser.parse_args()
    try:
        raw = Path(args.ocr_input).read_text(encoding='utf-8-sig') if args.ocr_input else None
        result = OCRNumericStore(args.store).extract(args.image, raw, args.format,
            ocr=lambda image: request_ocr(image, args.backend, args.timeout, function='markdown', resolution='gundam'))
        for key, value in result.items():
            print(f'{key}: ' + ('NA' if value is None else f'{value:.12g}'))
        return 0
    except Exception as error:
        print(f'Incomplete: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
