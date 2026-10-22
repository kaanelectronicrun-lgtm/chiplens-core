"""Small JSON/CSV model-generation CLI; no desktop/PDF dependencies."""
import argparse
import math
import sys

from .pipeline import process


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimated capacitor/inductor SPICE and Touchstone models")
    parser.add_argument("input", help="component JSON; graph CSV paths are relative to it")
    parser.add_argument("-o", "--out", default="outputs")
    parser.add_argument("--topology", choices=("series", "shunt"), default="series")
    parser.add_argument("--z0", type=float, default=50.0)
    parser.add_argument("--fstart", type=float, default=1e4)
    parser.add_argument("--fstop", type=float, default=1e10)
    args = parser.parse_args(argv)
    if not all(math.isfinite(x) and x > 0 for x in (args.z0, args.fstart, args.fstop)):
        parser.error("z0 and sweep bounds must be finite and positive")
    if args.fstart >= args.fstop:
        parser.error("fstart must be smaller than fstop")
    try:
        for path in process(args.input, args.out, args.z0, args.fstart, args.fstop, args.topology):
            print(path)
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        print(f"Model generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
