r"""Print six separate RGB DE-FIVE components, without a combined score."""
import argparse
import sys
from services.frequency_features import FrequencyFeatureStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Original image file path")
    parser.add_argument("--store", help="Optional cache directory")
    args = parser.parse_args()
    try:
        store = FrequencyFeatureStore(args.store)
        for name, value in store.extract_de_five_components(args.image).items():
            print(f"{name}: {value:.12g}")
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
