#!/usr/bin/env python3

"""Create the initial global model checkpoint for the FL workflow."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flwr_torch_utils import build_model_from_spec, ensure_parent, get_model_spec, load_config, save_model_payload, checkpoint_payload, set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("split_seed", 13)))
    model_spec = get_model_spec(config)
    model = build_model_from_spec(model_spec)

    payload = checkpoint_payload(
        0,
        model_spec,
        model.state_dict(),
        extra={
            "aggregation": model_spec["aggregation"],
            "fedprox_mu": model_spec["fedprox_mu"],
        },
    )

    ensure_parent(args.model)
    ensure_parent(args.model_config)
    save_model_payload(args.model, payload)
    with open(args.model_config, "w", encoding="utf-8") as handle:
        json.dump(model_spec, handle, indent=2, sort_keys=True)
    print(f"Initialized {model_spec['model_name']} model at {args.model}")


if __name__ == "__main__":
    main()
