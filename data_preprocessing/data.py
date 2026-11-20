"""Shared frame loading and small file utilities for preprocessing."""

import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_samples(samples):
    if not samples:
        raise ValueError("The sample list is empty.")
    seen = set()
    for sample in samples:
        name = sample["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid sample id: {name!r}. Use letters, digits, dots, '_' or '-'.")
        if name in seen:
            raise ValueError(f"Duplicate sample id: {name}")
        if type(sample["label"]) is not int or sample["label"] < 0:
            raise ValueError(f"{name}: labels must be zero-based, nonnegative integers.")
        seen.add(name)


def read_index(path):
    index = json.loads(Path(path).read_text())
    validate_samples(index["samples"])
    if index["frames"] < 1 or index["image_size"] < 1:
        raise ValueError("Frame count and image size must be positive.")
    return index


def read_frames(index_path, sample, index):
    source = sample["images"]
    path = Path(index_path).resolve().parent / source["path"]
    with h5py.File(path, "r") as handle:
        frames = handle[source["key"]][:]
    expected = (index["frames"], index["image_size"], index["image_size"], 3)
    if frames.dtype != np.uint8 or frames.shape != expected:
        raise ValueError(f"{sample['id']}: expected uint8 RGB frames of shape {expected}.")
    return frames


def image_tensor(frames, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    images = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255
    return (images - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[None, :, None, None]


def metadata(index_path, **settings):
    return {"frame_index_sha256": file_hash(index_path), **settings}


def check_source(info, index_path):
    if info["frame_index_sha256"] != file_hash(index_path):
        raise ValueError("These outputs were generated from a different frame index.")


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)
