"""Print five cross-box structural features; independent of other extractors."""
import argparse
from pathlib import Path
import sys
from extract_text_rules import request_ocr
from services.distributed_text_features import DistributedTextStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--ocr-input', help='UTF-8 grounded OCR output or word TSV')
    parser.add_argument('--format', choices=['grounding', 'tsv'], default='grounding')
    parser.add_argument('--max-blocks', type=int, default=4, help='Maximum consecutive boxes per window, 2..32; default 4')
    parser.add_argument('--store')
    parser.add_argument('--backend', default='http://127.0.0.1:5000')
    parser.add_argument('--timeout', type=float, default=1200)
    args = parser.parse_args()
    try:
        raw = Path(args.ocr_input).read_text(encoding='utf-8-sig') if args.ocr_input else None
        values = DistributedTextStore(args.store).extract(args.image, raw, args.format,
            ocr=lambda path: request_ocr(path, args.backend, args.timeout, function='markdown', resolution='gundam'),
            max_blocks=args.max_blocks)
        for name, value in values.items():
            print(f'{name}: ' + ('NA' if value is None else f'{value:.12g}'))
        return 0
    except Exception as error:
        print(f'Incomplete: {error}; completed stages remain cached', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
