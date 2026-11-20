"""Save one ordered set of RGB frames for all later preprocessing stages."""

import argparse
import csv
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image

from data import validate_samples, write_json


def video_frames(path, count, size):
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", str(path),
    ], check=True, capture_output=True, text=True)
    duration = float(probe.stdout.strip())
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration: {path}")
    # Same full-clip FPS sampling policy as the supplied extraction script.
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-vf", f"fps={count / duration},scale={size}:{size}",
        "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], check=True, capture_output=True)
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    if raw.size == 0 or raw.size % (size * size * 3):
        raise ValueError(f"No complete RGB frames decoded from {path}")
    frames = raw.reshape(-1, size, size, 3)
    decoded = len(frames)
    if decoded < count:
        frames = np.concatenate([frames, np.repeat(frames[-1:], count - decoded, axis=0)])
    return frames[:count], {"method": "ffmpeg_fps", "duration": duration,
                           "fps": count / duration, "repeated_last_frames": max(0, count - decoded)}


def sample_frames(path, key, count, size):
    if key:
        with h5py.File(path, "r") as handle:
            arrays = handle[key][:]
        if arrays.dtype != np.uint8 or arrays.ndim != 4 or arrays.shape[-1] != 3 or not len(arrays):
            raise ValueError("HDF5 input must contain nonempty uint8 RGB arrays (T,H,W,3).")
        indices = np.linspace(0, len(arrays) - 1, count).round().astype(int)
        frames = np.stack([np.asarray(Image.fromarray(arrays[i]).resize(
            (size, size), Image.Resampling.LANCZOS)) for i in indices])
        return frames, {"method": "uniform_indices", "indices": indices.tolist(), "source_key": key}
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        if count != 1:
            raise ValueError("Use --frames 1 for an image dataset.")
        with Image.open(path) as image:
            frames = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS))[None]
        return frames, {"method": "single_image"}
    return video_frames(path, count, size)


def extract(args):
    if args.frames < 1 or args.size < 1:
        raise ValueError("--frames and --size must be positive.")
    source = Path(args.samples).resolve()
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples = [{"id": row["id"], "label": int(row["label"])} for row in rows]
    validate_samples(samples)
    paths = [(source.parent / row["path"]).resolve() for row in rows]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(output / "frames.h5", "x") as handle:
        for sample, row, path in zip(samples, rows, paths):
            frames, sampling = sample_frames(path, row.get("key", ""), args.frames, args.size)
            handle.create_dataset(sample["id"], data=frames, compression="gzip")
            sample.update(images={"path": "frames.h5", "key": sample["id"]},
                          source=str(path), sampling=sampling)
            print(f"{sample['id']}: {len(frames)} frames", flush=True)
    write_json(output / "frames.json", {"frames": args.frames, "image_size": args.size, "samples": samples})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="CSV with id,path,label and optional HDF5 key.")
    parser.add_argument("--output", required=True, help="New directory for frames.h5 and frames.json.")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--size", type=int, default=224)
    extract(parser.parse_args())
