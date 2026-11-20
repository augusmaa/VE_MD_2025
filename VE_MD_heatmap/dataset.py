"""Prepared video frames, cached ViT features and dense heatmap targets.

Tensor videos are already preprocessed. Image/HDF5 inputs are resized and
ImageNet-normalized. Heatmaps must follow the same frame order as the video and cached features.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """One dataset split described by a JSON list of paths and labels."""

    def __init__(self, 
                 manifest, 
                 num_classes, 
                 frames, 
                 image_size=224, 
                 vit_dim=1024,
                 use_vit=True, 
                 person_channels=18, 
                 face_channels=83,
                 use_person=True, 
                 use_face=False, 
                 require_targets=True):
        self.path = Path(manifest).resolve()
        self.records = json.loads(self.path.read_text())
        if not isinstance(self.records, list) or not self.records:
            raise ValueError(f"{self.path}: expected a non-empty JSON list.")
        self.num_classes, self.frames, self.image_size = num_classes, frames, image_size
        self.use_vit, self.vit_dim, self.require_targets = use_vit, vit_dim, require_targets
        self.channels = {name: channels for name, enabled, channels in (
            ("person", use_person, person_channels), ("face", use_face, face_channels)
        ) if enabled}
        for index, record in enumerate(self.records):
            label = record.get("label")
            if type(label) is not int or not 0 <= label < num_classes:
                raise ValueError(f"Record {index}: label must be an integer in [0,{num_classes - 1}].")
            required = ["images"] + (["feat_vit"] if use_vit else [])
            if require_targets:
                for name in self.channels:
                    required.append(f"limb_{name}")
            for key in required:
                if key not in record:
                    raise ValueError(f"Record {index}: missing {key}.")

    def __len__(self):
        return len(self.records)

    def _path(self, value):
        return self.path.parent / value

    def _tensor(self, value):
        return torch.load(self._path(value), map_location="cpu", weights_only=True)

    def _images(self, source):
        if isinstance(source, str) and Path(source).suffix.lower() == ".pt":
            images = self._tensor(source)
            if not isinstance(images, torch.Tensor) or not images.is_floating_point():
                raise ValueError("Video .pt files must contain a preprocessed floating-point tensor.")
            if images.ndim == 3:
                images = images.unsqueeze(0)
            return images.float()
        from PIL import Image

        if isinstance(source, dict):
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("HDF5 inputs require: pip install h5py==3.12.1") from exc
            # Open within __getitem__; never share an HDF5 handle between workers.
            with h5py.File(self._path(source["path"]), "r") as handle:
                arrays = handle[source["key"]][:]
            if arrays.dtype != np.uint8 or arrays.ndim != 4 or arrays.shape[-1] != 3:
                raise ValueError("HDF5 videos must contain uint8 RGB arrays with shape (T,H,W,3).")
            pil_frames = [Image.fromarray(frame) for frame in arrays]
        else:
            with Image.open(self._path(source)) as im:
                pil_frames = [im.convert("RGB")]
        frames = [torch.from_numpy(np.array(im.resize((self.image_size, self.image_size),
                  Image.Resampling.LANCZOS), copy=True)).permute(2, 0, 1).float() / 255
                  for im in pil_frames]
        images = torch.stack(frames)
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (images - mean) / std

    def __getitem__(self, index):
        record = self.records[index]
        images = self._images(record["images"])
        if images.shape != (self.frames, 3, self.image_size, self.image_size):
            raise ValueError(f"Record {index}: images shape {tuple(images.shape)} differs from "
                             f"({self.frames},3,{self.image_size},{self.image_size}).")
        if not torch.isfinite(images).all():
            raise ValueError(f"Record {index}: non-finite image values.")
        sample = {"images": images, "label_emotion": record["label"]}
        if self.use_vit:
            features = self._tensor(record["feat_vit"]).float()
            if features.ndim == 1 and self.frames == 1:
                features = features.unsqueeze(0)
            if features.shape != (self.frames, self.vit_dim) or not torch.isfinite(features).all():
                raise ValueError(f"Record {index}: feat_vit must be finite with shape ({self.frames},{self.vit_dim}).")
            sample["feat_vit"] = features
        if self.require_targets:
            for name, channels in self.channels.items():
                heatmap = self._tensor(record[f"limb_{name}"]).float()
                if heatmap.ndim == 3 and self.frames == 1:
                    heatmap = heatmap.unsqueeze(0)
                if heatmap.shape != (self.frames, channels, 56, 56):
                    raise ValueError(f"Record {index}: limb_{name} must have shape ({self.frames},{channels},56,56).")
                if not torch.isfinite(heatmap).all():
                    raise ValueError(f"Record {index}: heatmaps must be finite.")
                sample[f"limb_{name}"] = heatmap
        return sample
