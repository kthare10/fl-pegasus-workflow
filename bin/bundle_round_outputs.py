#!/usr/bin/env python3

"""Bundle round outputs into a single artifact for subworkflow export."""

import argparse
import os
import tarfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-in", required=True, help="Round model checkpoint to bundle.")
    parser.add_argument("--metric-in", required=True, help="Round aggregation metric to bundle.")
    parser.add_argument("--bundle-out", required=True, help="Output tar archive.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.bundle_out), exist_ok=True)
    with tarfile.open(args.bundle_out, "w") as archive:
        archive.add(args.model_in, arcname="model.pt")
        archive.add(args.metric_in, arcname="aggregation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
