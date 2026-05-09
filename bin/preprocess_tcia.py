#!/usr/bin/env python3

"""Preprocess one TCIA federated client into JSONL features."""

import argparse
import csv
import json
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return yaml.safe_load(text) if yaml else json.loads(text)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_runtime_image_path(path, config):
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.getcwd(), path))
        data_root = config.get("data_root")
        if data_root:
            candidates.append(os.path.join(data_root, path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def normalize_image(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[array.shape[0] // 2]
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val > min_val:
        array = (array - min_val) / (max_val - min_val)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    return array


def resize_nearest(image, size):
    y_idx = np.linspace(0, image.shape[0] - 1, size).astype(int)
    x_idx = np.linspace(0, image.shape[1] - 1, size).astype(int)
    return image[np.ix_(y_idx, x_idx)]


def load_image_array(path, image_size):
    lower = path.lower()
    if os.path.isdir(path):
        slices = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".dcm")
        )
        if not slices:
            raise FileNotFoundError(f"No DICOM slices found in {path}")
        path = slices[len(slices) // 2]
        lower = path.lower()

    if lower.endswith(".dcm"):
        if pydicom is None:
            raise RuntimeError("pydicom is required to preprocess TCIA DICOM files")
        image = pydicom.dcmread(path).pixel_array
    elif lower.endswith(".nii") or lower.endswith(".nii.gz") or lower.endswith(".mha") or lower.endswith(".mhd"):
        if sitk is None:
            raise RuntimeError("SimpleITK is required to preprocess TCIA volume files")
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))
    elif lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
        if Image is None:
            raise RuntimeError("Pillow is required to preprocess TCIA image files")
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    else:
        if sitk is None:
       	    raise RuntimeError("SimpleITK is required to preprocess TCIA image files")
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

    image = normalize_image(image)
    image = resize_nearest(image, image_size)
    return image


def feature_vector_from_image(path, length, image_size):
    image = load_image_array(path, image_size)
    flat = image.reshape(-1)
    chunks = np.array_split(flat, length)
    features = []
    for chunk in chunks:
        mean = float(chunk.mean()) if chunk.size else 0.0
        features.append(mean * 2.0 - 1.0)
    return features


def feature_vector_from_archive_member(member_name, archive_path, length, image_size):
    suffix = "".join(Path(member_name).suffixes) or Path(member_name).suffix or ".dat"
    with tarfile.open(archive_path, "r:*") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(member_name)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_handle:
            temp_handle.write(member.read())
            temp_path = temp_handle.name
    try:
        return feature_vector_from_image(temp_path, length, image_size)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-manifest", required=True)
    parser.add_argument("--client-id", required=True, type=int)
    parser.add_argument("--archive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    positive_terms = set(config.get("positive_terms", ["nodule", "positive", "malignant"]))
    feature_dim = int(config.get("feature_dim", 24))
    image_size = int(config.get("image_size", 96))
    ensure_parent(args.output)

    count = 0
    with open(args.client_manifest, newline="", encoding="utf-8") as manifest, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        for row in csv.DictReader(manifest):
            if int(row["client_id"]) != args.client_id:
                continue
            raw = row["labels"].lower()
            y = 1 if any(term in raw for term in positive_terms) else 0
            image_path = row["image_path"] if args.archive else resolve_runtime_image_path(row["image_path"], config)
            record = {
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "split": row["split"],
                "x": feature_vector_from_archive_member(
                    image_path, args.archive, feature_dim, image_size
                )
                if args.archive
                else feature_vector_from_image(image_path, feature_dim, image_size),
                "y": [y],
            }
            out.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1

    print(f"Wrote {count} TCIA records for client {args.client_id} to {args.output}")


if __name__ == "__main__":
    main()
