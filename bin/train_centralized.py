#!/usr/bin/env python3

"""Train a centralized pooled-data baseline with the same image model stack."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from torch.utils.data import DataLoader  # noqa: E402

from flwr_torch_utils import (  # noqa: E402
    ImageRecordDataset,
    ResourceMonitor,
    build_model_from_spec,
    compute_pos_weight,
    ensure_parent,
    evaluate_records,
    get_model_spec,
    learning_rates,
    load_config,
    make_loss_fn,
    make_optimizer,
    make_scheduler,
    pooled_training_records,
    safe_mean,
    select_device,
    set_seed,
    train_one_epoch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-data", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("split_seed", 13)))
    device = select_device(config)
    model_spec = get_model_spec(config)
    model = build_model_from_spec(model_spec).to(device)

    train_records, val_records, test_records = pooled_training_records(args.client_data)
    train_dataset = ImageRecordDataset(train_records, config, train=True)
    batch_size = min(int(config.get("batch_size", 8)), max(len(train_dataset), 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config, int(config.get("local_epochs", 1)))
    pos_weight = compute_pos_weight(train_records, config)
    loss_fn = make_loss_fn(config, pos_weight=pos_weight, device=device)
    monitor = ResourceMonitor(float(config.get("monitor_interval_seconds", 5)), device=device).start()

    epoch_metrics = []
    for epoch_idx in range(int(config.get("local_epochs", 1))):
        metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config)
        metrics["epoch"] = epoch_idx + 1
        if scheduler is not None:
            scheduler.step()
        metrics["learning_rates"] = learning_rates(optimizer)
        if val_records:
            metrics["val"] = evaluate_records(model, val_records, config, device)
        epoch_metrics.append(metrics)

    resource_summary = monitor.stop()
    test_metrics = evaluate_records(model, test_records, config, device)
    output = {
        "dataset_name": config["dataset_name"],
        "method": "centralized_supervised_baseline",
        "model_name": model_spec["model_name"],
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": test_metrics["test_samples"],
        "accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["mean_loss"],
        "epoch_metrics": epoch_metrics,
        "class_weighted_loss": pos_weight.tolist() if pos_weight is not None else None,
        "resource_monitor": resource_summary,
        "mean_train_loss": safe_mean([item["mean_loss"] for item in epoch_metrics if item["mean_loss"] is not None]),
    }

    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Trained centralized baseline on {len(train_records)} samples")


if __name__ == "__main__":
    main()
