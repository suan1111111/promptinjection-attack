"""Print source-based scaling attack features; no OCR or model inference."""
import argparse
import sys
from services.scaling_features import ScalingFeatureStore, ALGORITHMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--target', nargs=2, type=int, default=(224, 224), metavar=('HEIGHT', 'WIDTH'), help='Downstream height and width; default: 224 224')
    parser.add_argument('--algorithm', choices=ALGORITHMS, default='nearest', help='Pillow interpolation; default: nearest')
    parser.add_argument('--radius', type=int, default=5, help='Peak sampling radius; default: 5')
    parser.add_argument('--store')
    args = parser.parse_args()
    try:
        values, failures = ScalingFeatureStore(args.store).extract(args.image, tuple(args.target), args.algorithm, args.radius)
        for name, value in values.items():
            print(f'{name}: ' + ('NA' if value is None else f'{value:.12g}'))
        for name, error in failures.items():
            print(f'{name} incomplete: {error}', file=sys.stderr)
        return 1 if failures else 0
    except Exception as error:
        print(f'Incomplete: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
