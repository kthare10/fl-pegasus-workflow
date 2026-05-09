#!/usr/bin/env python3

"""Shared PyTorch and Flower-compatible helpers for the FL workflow."""

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import os
import random
import threading
import time

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for the FL training stack") from exc

try:
    from flwr.server.strategy.aggregate import aggregate as flwr_aggregate
except ImportError:  # pragma: no cover
    flwr_aggregate = None

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required for image training") from exc

try:
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for image model training") from exc

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return yaml.safe_load(text) if yaml else json.loads(text)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_output_dim(config):
    if "num_classes" in config:
        return int(config["num_classes"])
    task = str(config.get("task", "")).lower()
    dataset = str(config.get("dataset_name", "")).lower()
    labels = config.get("labels", [])
    if "multi_label" in task:
        return len(labels) if labels else 1
    if "binary" in task or dataset.startswith("tcia") or "lidc" in dataset:
        return 1
    return len(labels) if labels else 1


def uses_multilabel_loss(config_or_spec):
    task = str(config_or_spec.get("task", "")).lower()
    output_dim = int(config_or_spec.get("output_dim", 1))
    return "multi_label" in task or output_dim > 1


def get_model_spec(config):
    hidden_dims = config.get("hidden_dims", [128, 64])
    if isinstance(hidden_dims, str):
        hidden_dims = [int(part) for part in hidden_dims.split(",") if part.strip()]
    return {
        "model_name": config.get("model_name", "resnet18"),
        "image_size": int(config.get("image_size", 224)),
        "input_channels": int(config.get("input_channels", 3)),
        "output_dim": infer_output_dim(config),
        "hidden_dims": [int(value) for value in hidden_dims],
        "dropout": float(config.get("dropout", 0.1)),
        "learning_rate": float(config.get("learning_rate", 0.001)),
        "local_epochs": int(config.get("local_epochs", 1)),
        "batch_size": int(config.get("batch_size", 8)),
        "aggregation": str(config.get("aggregation", "fedavg")).lower(),
        "fedprox_mu": float(config.get("fedprox_mu", config.get("mu", 0.0))),
        "optimizer": str(config.get("optimizer", "adam")).lower(),
        "scheduler": str(config.get("scheduler", "none")).lower(),
        "scheduler_step_size": int(config.get("scheduler_step_size", 5)),
        "scheduler_gamma": float(config.get("scheduler_gamma", 0.5)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "momentum": float(config.get("momentum", 0.9)),
        "gradient_clip_norm": float(config.get("gradient_clip_norm", 0.0)),
        "pretrained": bool(config.get("pretrained", True)),
        "freeze_backbone": bool(config.get("freeze_backbone", False)),
        "unfreeze_backbone_epoch": int(config.get("unfreeze_backbone_epoch", -1)),
        "class_weighted_loss": bool(config.get("class_weighted_loss", True)),
        "task": str(config.get("task", "")),
    }


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def freeze_backbone_parameters(model):
    if hasattr(model, "fc"):
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")


def unfreeze_all_parameters(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def build_model_from_spec(model_spec):
    name = str(model_spec["model_name"]).lower()
    if name in {"resnet18", "resnet-18"}:
        weights = None
        if model_spec.get("pretrained", False):
            try:
                weights = ResNet18_Weights.DEFAULT
            except Exception:  # pragma: no cover
                weights = None
        try:
            model = resnet18(weights=weights)
        except Exception:  # pragma: no cover
            model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, int(model_spec["output_dim"]))
        if model_spec.get("freeze_backbone"):
            freeze_backbone_parameters(model)
        return model
    if name in {"torch-mlp", "mlp", "feature-mlp", "linear-smoke"}:
        hidden_dims = [] if name == "linear-smoke" else model_spec["hidden_dims"]
        dropout = 0.0 if name == "linear-smoke" else model_spec["dropout"]
        input_dim = int(model_spec.get("feature_dim", 16))
        return FeatureMLP(input_dim, model_spec["output_dim"], hidden_dims, dropout)
    raise ValueError(f"Unsupported model_name: {model_spec['model_name']}")


def checkpoint_payload(round_idx, model_spec, state_dict, extra=None):
    payload = {
        "round": round_idx,
        "framework": "flwr+pytorch",
        "model_spec": model_spec,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in state_dict.items()},
    }
    if extra:
        payload.update(extra)
    return payload


def save_model_payload(path, payload):
    ensure_parent(path)
    torch.save(payload, path)


def _load_legacy_json_payload(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_dict = OrderedDict(
        (name, torch.tensor(values, dtype=torch.float32))
        for name, values in payload["state_dict"].items()
    )
    model_spec = {
        "model_name": payload["model_name"],
        "feature_dim": int(payload.get("feature_dim", 16)),
        "output_dim": int(payload["output_dim"]),
        "hidden_dims": [int(value) for value in payload.get("hidden_dims", [])],
        "dropout": float(payload.get("dropout", 0.0)),
        "task": payload.get("task", ""),
    }
    return payload, model_spec, state_dict


def load_model_payload(path):
    if path.endswith(".json"):
        return _load_legacy_json_payload(path)
    payload = torch.load(path, map_location="cpu")
    model_spec = dict(payload["model_spec"])
    state_dict = OrderedDict((name, tensor.float().cpu()) for name, tensor in payload["state_dict"].items())
    return payload, model_spec, state_dict


def select_device(config):
    requested = int(config.get("request_gpus", 0))
    if requested > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def numpy_state_list(state_dict):
    keys = list(state_dict.keys())
    arrays = [state_dict[key].detach().cpu().numpy() for key in keys]
    return keys, arrays


def aggregate_state_dicts(weighted_state_dicts):
    keys, first_arrays = numpy_state_list(weighted_state_dicts[0][0])
    if flwr_aggregate is not None:
        aggregated = flwr_aggregate(
            [(numpy_state_list(state_dict)[1], sample_count) for state_dict, sample_count in weighted_state_dicts]
        )
    else:
        total = sum(weight for _, weight in weighted_state_dicts)
        if total <= 0:
            total = len(weighted_state_dicts)
            weighted_state_dicts = [(state, 1) for state, _ in weighted_state_dicts]
        aggregated = [np.zeros_like(array, dtype=np.float32) for array in first_arrays]
        for state_dict, sample_count in weighted_state_dicts:
            _, arrays = numpy_state_list(state_dict)
            factor = float(sample_count) / float(total)
            aggregated = [acc + factor * arr for acc, arr in zip(aggregated, arrays)]
    return OrderedDict((key, torch.tensor(array, dtype=torch.float32)) for key, array in zip(keys, aggregated))


def make_optimizer(model, config):
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", 0.0))
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, config, total_epochs):
    scheduler_name = str(config.get("scheduler", "none")).lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(config.get("scheduler_step_size", 5)), 1),
            gamma=float(config.get("scheduler_gamma", 0.5)),
        )
    return None


def normalize_record_image(record):
    image = np.asarray(record["image"], dtype=np.float32)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = np.moveaxis(image, 0, -1)
    image = np.clip(image, 0.0, 1.0)
    if image.ndim == 2:
        image = (image * 255.0).astype(np.uint8)
        return Image.fromarray(image, mode="L").convert("RGB")
    image = (image * 255.0).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def build_image_transform(config, train):
    image_size = int(config.get("image_size", 224))
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=float(config.get("rotation_degrees", 10))),
                transforms.ColorJitter(
                    brightness=float(config.get("jitter_brightness", 0.1)),
                    contrast=float(config.get("jitter_contrast", 0.1)),
                    saturation=float(config.get("jitter_saturation", 0.1)),
                    hue=float(config.get("jitter_hue", 0.02)),
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def load_records(paths, split=None):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if split is not None and record["split"] != split:
                    continue
                records.append(record)
    return records


class ImageRecordDataset(Dataset):
    def __init__(self, records, config, train=False):
        self.records = list(records)
        self.transform = build_image_transform(config, train=train)
        self.multilabel = uses_multilabel_loss(config)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = self.transform(normalize_record_image(record))
        label = torch.tensor(record["y"], dtype=torch.float32)
        if not self.multilabel:
            label = label.reshape(-1)
        return image, label


def compute_pos_weight(records, config):
    if not records or not bool(config.get("class_weighted_loss", True)):
        return None
    labels = np.asarray([record["y"] for record in records], dtype=np.float32)
    if labels.ndim == 1:
        labels = labels[:, None]
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, float(config.get("max_pos_weight", 20.0)))
    return torch.tensor(weights, dtype=torch.float32)


def make_loss_fn(config, pos_weight=None, device=None):
    weight = pos_weight.to(device) if pos_weight is not None and device is not None else pos_weight
    return nn.BCEWithLogitsLoss(pos_weight=weight)


def proximal_penalty(model, global_state, device):
    penalty = torch.zeros(1, device=device)
    for name, parameter in model.named_parameters():
        penalty = penalty + torch.sum((parameter - global_state[name].to(device)) ** 2)
    return penalty


def scalar_accuracy(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return float((preds == labels).all(dim=1).float().mean().item())


def evaluate_records(model, records, config, device):
    dataset = ImageRecordDataset(records, config, train=False)
    if len(dataset) == 0:
        return {"test_samples": 0, "accuracy": None}
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(int(config.get("batch_size", 8)), max(len(dataset), 1)),
        shuffle=False,
    )
    total = 0
    correct = 0.0
    loss_fn = make_loss_fn(config, device=device)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            losses.append(float(loss_fn(logits, batch_y).detach().cpu().item()))
            batch_acc = scalar_accuracy(logits, batch_y)
            correct += batch_acc * batch_y.shape[0]
            total += batch_y.shape[0]
    return {
        "test_samples": total,
        "accuracy": correct / total if total else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
    }


def maybe_unfreeze_backbone(model, config, epoch_idx):
    if not bool(config.get("freeze_backbone")):
        return False
    target_epoch = int(config.get("unfreeze_backbone_epoch", -1))
    if target_epoch >= 0 and epoch_idx >= target_epoch:
        unfreeze_all_parameters(model)
        return True
    return False


@dataclass
class ResourceSummary:
    samples: int
    peak_rss_mb: float
    mean_rss_mb: float
    mean_cpu_percent: float
    peak_gpu_memory_mb: float
    mean_gpu_utilization: float | None


class ResourceMonitor:
    def __init__(self, interval_seconds=5.0, device=None):
        self.interval_seconds = interval_seconds
        self.device = device
        self._stop = threading.Event()
        self._thread = None
        self.samples = []
        self.process = psutil.Process(os.getpid()) if psutil is not None else None
        self.nvml_handle = None
        self.gpu_index = 0
        if pynvml is not None and device is not None and getattr(device, "type", "") == "cuda":
            try:
                pynvml.nvmlInit()
                self.gpu_index = int(getattr(device, "index", 0) or 0)
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception:  # pragma: no cover
                self.nvml_handle = None

    def _sample(self):
        rss_mb = None
        cpu_percent = None
        if self.process is not None:
            rss_mb = self.process.memory_info().rss / (1024 * 1024)
            cpu_percent = self.process.cpu_percent(interval=None)
        gpu_memory_mb = None
        gpu_utilization = None
        if self.device is not None and getattr(self.device, "type", "") == "cuda" and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            if self.nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    gpu_utilization = float(util.gpu)
                except Exception:  # pragma: no cover
                    gpu_utilization = None
        self.samples.append(
            {
                "timestamp": time.time(),
                "rss_mb": rss_mb,
                "cpu_percent": cpu_percent,
                "gpu_memory_mb": gpu_memory_mb,
                "gpu_utilization": gpu_utilization,
            }
        )

    def _run(self):
        if self.process is not None:
            self.process.cpu_percent(interval=None)
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1.0)
        if self.nvml_handle is not None and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass
        return self.summary()

    def summary(self):
        if not self.samples:
            return None
        rss_values = [sample["rss_mb"] for sample in self.samples if sample["rss_mb"] is not None]
        cpu_values = [sample["cpu_percent"] for sample in self.samples if sample["cpu_percent"] is not None]
        gpu_mem_values = [sample["gpu_memory_mb"] for sample in self.samples if sample["gpu_memory_mb"] is not None]
        gpu_util_values = [
            sample["gpu_utilization"] for sample in self.samples if sample["gpu_utilization"] is not None
        ]
        return {
            "interval_seconds": self.interval_seconds,
            "samples": len(self.samples),
            "peak_rss_mb": max(rss_values) if rss_values else None,
            "mean_rss_mb": sum(rss_values) / len(rss_values) if rss_values else None,
            "mean_cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else None,
            "peak_gpu_memory_mb": max(gpu_mem_values) if gpu_mem_values else None,
            "mean_gpu_utilization": sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else None,
        }


def learning_rates(optimizer):
    return [float(group["lr"]) for group in optimizer.param_groups]


def safe_mean(values):
    return sum(values) / len(values) if values else None


def pooled_training_records(paths):
    return load_records(paths, split="train"), load_records(paths, split="val"), load_records(paths, split="test")


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    config,
    global_state=None,
):
    use_fedprox = str(config.get("aggregation", "fedavg")).lower() == "fedprox" or str(
        config.get("algorithm", "")
    ).lower() == "fedprox"
    fedprox_mu = float(config.get("fedprox_mu", config.get("mu", 0.0)))
    gradient_clip_norm = float(config.get("gradient_clip_norm", 0.0))
    losses = []
    accuracies = []
    model.train()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        if use_fedprox and fedprox_mu > 0 and global_state is not None:
            loss = loss + 0.5 * fedprox_mu * proximal_penalty(model, global_state, device)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        accuracies.append(scalar_accuracy(logits.detach(), batch_y.detach()))
    return {
        "mean_loss": safe_mean(losses),
        "mean_accuracy": safe_mean(accuracies),
        "steps": len(losses),
    }
