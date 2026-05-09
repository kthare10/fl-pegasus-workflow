#!/usr/bin/env python3

"""Train one federated client for one round using image models, PyTorch, and FedProx."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from flwr_torch_utils import (  # noqa: E402
    ImageRecordDataset,
    ResourceMonitor,
    build_model_from_spec,
    checkpoint_payload,
    compute_pos_weight,
    ensure_parent,
    evaluate_records,
    get_model_spec,
    learning_rates,
    load_config,
    load_model_payload,
    load_records,
    make_loss_fn,
    make_optimizer,
    make_scheduler,
    maybe_unfreeze_backbone,
    safe_mean,
    save_model_payload,
    select_device,
    set_seed,
    train_one_epoch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-id", required=True, type=int)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--client-data", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--count-output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("split_seed", 13)) + args.client_id + args.round)
    _, model_spec, global_state = load_model_payload(args.global_model)
    model_spec.update(get_model_spec(config))
    device = select_device(config)

    train_records = load_records([args.client_data], split="train")
    val_records = load_records([args.client_data], split="val")
    train_dataset = ImageRecordDataset(train_records, config, train=True)
    batch_size = min(int(config.get("batch_size", 8)), max(len(train_dataset), 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = build_model_from_spec(model_spec).to(device)
    model.load_state_dict(global_state)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config, int(config.get("local_epochs", 1)))
    pos_weight = compute_pos_weight(train_records, config)
    loss_fn = make_loss_fn(config, pos_weight=pos_weight, device=device)
    monitor = ResourceMonitor(float(config.get("monitor_interval_seconds", 5)), device=device).start()

    epoch_metrics = []
    unfroze_backbone = False
    for epoch_idx in range(int(config.get("local_epochs", 1))):
        if maybe_unfreeze_backbone(model, config, epoch_idx):
            optimizer = make_optimizer(model, config)
            scheduler = make_scheduler(optimizer, config, int(config.get("local_epochs", 1)))
            unfroze_backbone = True
        metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config, global_state=global_state)
        metrics["epoch"] = epoch_idx + 1
        if scheduler is not None:
            scheduler.step()
        metrics["learning_rates"] = learning_rates(optimizer)
        if val_records:
            metrics["val"] = evaluate_records(model, val_records, config, device)
        epoch_metrics.append(metrics)

    resource_summary = monitor.stop()

    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    update = checkpoint_payload(
        args.round,
        model_spec,
        state_dict,
        extra={
            "client_id": args.client_id,
            "sample_count": len(train_records),
            "aggregation": str(config.get("aggregation", "fedavg")).lower(),
            "fedprox_mu": float(config.get("fedprox_mu", config.get("mu", 0.0))),
        },
    )

    metrics = {
        "round": args.round,
        "client_id": args.client_id,
        "device": str(device),
        "framework": "flwr+pytorch",
        "algorithm": str(config.get("aggregation", "fedavg")).lower(),
        "model_name": model_spec["model_name"],
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "mean_train_loss": safe_mean([item["mean_loss"] for item in epoch_metrics if item["mean_loss"] is not None]),
        "mean_train_accuracy": safe_mean(
            [item["mean_accuracy"] for item in epoch_metrics if item["mean_accuracy"] is not None]
        ),
        "epoch_metrics": epoch_metrics,
        "class_weighted_loss": pos_weight.tolist() if pos_weight is not None else None,
        "resource_monitor": resource_summary,
        "unfroze_backbone": unfroze_backbone,
    }
    count = {"client_id": args.client_id, "sample_count": len(train_records)}

    for path in (args.output_model, args.metrics, args.count_output):
        ensure_parent(path)
    save_model_payload(args.output_model, update)
    with open(args.metrics, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with open(args.count_output, "w", encoding="utf-8") as handle:
        json.dump(count, handle, indent=2, sort_keys=True)

    print(
        f"Trained client {args.client_id} round {args.round} on {len(train_records)} samples using {metrics['algorithm']}"
    )


if __name__ == "__main__":
    main()
