"""
Prepared images, video frames, features and structural annotations.

Tensor videos are already preprocessed. Image/HDF5 inputs are resized and
ImageNet-normalized. Skeleton coordinates refer to the resized image plane.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """One dataset split described by a JSON list of paths and labels."""
    def __init__(self, 
                 manifest, # config file path, for data source
                 num_classes, 
                 frames, 
                 image_size=224, 
                 vit_dim=1024,
                 use_vit=True, 
                 person_limbs=18, 
                 face_limbs=20,
                 use_person=True, 
                 use_face=False, 
                 require_targets=True):
        self.path = Path(manifest).resolve()
        self.records = json.loads(self.path.read_text())
        if not isinstance(self.records, list) or not self.records:
            raise ValueError(f"{self.path}: expected a non-empty JSON list.")
        self.num_classes, self.frames, self.image_size = num_classes, frames, image_size
        self.use_vit, self.vit_dim, self.require_targets = use_vit, vit_dim, require_targets
        self.limbs = {name: limbs for name, enabled, limbs in (
            ("person", use_person, person_limbs), ("face", use_face, face_limbs)) if enabled}
        for index, record in enumerate(self.records):
            label = record.get("label")
            if type(label) is not int or not 0 <= label < num_classes:
                raise ValueError(f"Record {index}: label must be an integer in [0,{num_classes - 1}].")
            required = ["images"] + (["feat_vit"] if use_vit else [])
            if require_targets:
                for name in self.limbs:
                    required.extend((f"skeleton_{name}", f"adjacency_{name}"))
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

    def _structures(self, coordinates, adjacency, limbs):
        coords, adj = self._tensor(coordinates), self._tensor(adjacency)
        if isinstance(coords, torch.Tensor) and coords.ndim == 2 and self.frames == 1:
            coords, adj = coords.unsqueeze(0), adj.unsqueeze(0)
        if len(coords) != self.frames or len(adj) != self.frames:
            raise ValueError("Skeleton and adjacency frame counts must match the video.")
        coord_frames, adj_frames = [], []
        for c, a in zip(coords, adj):
            c, a = torch.as_tensor(c, dtype=torch.float32), torch.as_tensor(a, dtype=torch.float32)
            if c.ndim != 2 or c.shape[-1] != 4 * limbs:
                raise ValueError(f"Each skeleton frame must have shape (people,{4 * limbs}).")
            people = c.size(0)
            if a.shape not in ((people, limbs, limbs), (people, limbs * limbs)):
                raise ValueError("Adjacency dimensions do not match the skeleton dimensions.")
            a = a.reshape(people, limbs, limbs)
            if not torch.isfinite(c).all() or not torch.isfinite(a).all():
                raise ValueError("Skeleton/adjacency values must be finite.")
            if ((c != -1) & ((c < 0) | (c > self.image_size))).any():
                raise ValueError("Coordinates must be pixels in the resized image plane or -1 for missing.")
            if ((a < 0) | (a > 1)).any():
                raise ValueError("Adjacency targets must lie in [0,1].")
            coord_frames.append(c)
            adj_frames.append(a)
        people = max(c.size(0) for c in coord_frames)
        padded_coords = torch.full((self.frames, people, 4 * limbs), -1.0)
        padded_adj = torch.zeros(self.frames, people, limbs, limbs)
        for frame, (c, a) in enumerate(zip(coord_frames, adj_frames)):
            padded_coords[frame, :len(c)] = c
            padded_adj[frame, :len(a)] = a
        return padded_coords, padded_adj

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
            for name, limbs in self.limbs.items():
                sample[f"skeleton_{name}"], sample[f"adjacency_{name}"] = self._structures(
                    record[f"skeleton_{name}"], record[f"adjacency_{name}"], limbs
                )
        return sample


def collate_batch(samples):
    batch = {
        "images": torch.stack([sample["images"] for sample in samples]),
        "label_emotion": torch.tensor([sample["label_emotion"] for sample in samples], dtype=torch.long),
    }
    if "feat_vit" in samples[0]:
        batch["feat_vit"] = torch.stack([sample["feat_vit"] for sample in samples])
    for name in ("person", "face"):
        key = f"skeleton_{name}"
        if key not in samples[0]:
            continue
        frames, _, dim = samples[0][key].shape
        limbs = dim // 4
        people = max(sample[key].size(1) for sample in samples)
        coordinates = torch.full((len(samples), frames, people, dim), -1.0)
        adjacency = torch.zeros(len(samples), frames, people, limbs, limbs)
        for index, sample in enumerate(samples):
            count = sample[key].size(1)
            coordinates[index, :, :count] = sample[key]
            adjacency[index, :, :count] = sample[f"adjacency_{name}"]
        batch[key] = coordinates
        batch[f"adjacency_{name}"] = adjacency
    return batch
