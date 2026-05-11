#!/usr/bin/env python3

"""Unpack a bundled round artifact into parent-visible workflow files."""

import argparse
import os
import tarfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Round artifact bundle.")
    parser.add_argument("--model-out", required=True, help="Output model checkpoint path.")
    parser.add_argument("--metric-out", required=True, help="Output aggregation metric path.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metric_out), exist_ok=True)
    with tarfile.open(args.bundle, "r") as archive:
        model_member = archive.getmember("model.pt")
        metric_member = archive.getmember("aggregation.json")
        with archive.extractfile(model_member) as src, open(args.model_out, "wb") as dst:
            dst.write(src.read())
        with archive.extractfile(metric_member) as src, open(args.metric_out, "wb") as dst:
            dst.write(src.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
