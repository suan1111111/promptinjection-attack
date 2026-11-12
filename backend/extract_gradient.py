"""Print three cached LGS-inspired gradient summaries; no model required."""
import argparse
import sys
from services.gradient_features import GradientFeatureStore


def main():
    parser = argparse.ArgumentParser(description="Extract LGS-inspired gradient summaries")
    parser.add_argument("image", help="Image file path")
    parser.add_argument("--store", help="Optional cache directory")
    args = parser.parse_args()
    try:
        values = GradientFeatureStore(args.store).extract(args.image)
        for name, value in values.items():
            print(f"{name}: {value:.12g}")
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
