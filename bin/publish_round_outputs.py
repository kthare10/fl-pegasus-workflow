#!/usr/bin/env python3

"""Copy round outputs to parent-visible Pegasus subworkflow outputs."""

import argparse
import os
import shutil


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-in", required=True, help="Internal round model checkpoint.")
    parser.add_argument("--model-out", required=True, help="Parent-visible round model checkpoint.")
    parser.add_argument("--metric-in", required=True, help="Internal round aggregation metric.")
    parser.add_argument("--metric-out", required=True, help="Parent-visible round aggregation metric.")
    return parser.parse_args()


def main():
    args = parse_args()
    copy_file(args.model_in, args.model_out)
    copy_file(args.metric_in, args.metric_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
