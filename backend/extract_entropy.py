"""Print cached Jedi local entropy and candidate region statistics."""
import argparse
import sys
from services.entropy_features import EntropyFeatureStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--store')
    args = parser.parse_args()
    try:
        for name, value in EntropyFeatureStore(args.store).extract(args.image).items():
            print(f'{name}: {value:.12g}')
        return 0
    except Exception as error:
        print(f'Incomplete: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
