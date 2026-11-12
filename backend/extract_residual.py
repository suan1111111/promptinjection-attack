"""Print six SRM KB/KV residual summaries and persist incremental cache."""
import argparse
import sys

from services.residual_features import ResidualFeatureStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--store', help='Cache directory; default data/image_features')
    parser.add_argument('--groups', nargs='+', choices=['kb', 'kv'], default=['kb', 'kv'])
    args = parser.parse_args()
    try:
        values = ResidualFeatureStore(args.store).extract(args.image, args.groups)
        for name, value in values.items():
            print(f'{name}: {value:.12g}')
        return 0
    except Exception as error:
        print(f'Incomplete: {error}; completed groups remain cached', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
