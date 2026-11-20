"""Create PersonQuery skeletons and Heatmap targets from the same keypoint annotations."""

import argparse
import json
from pathlib import Path

import torch

from data import check_source, metadata, read_index, write_json
from structures import BODY_LIMBS, FACE_LIMBS, FACE_HEATMAP_LIMBS
from structures import heatmap_targets, skeleton_targets, valid_keypoints


def read_annotations(path, index_path, index, keypoints):
    annotations = json.loads(Path(path).read_text())
    check_source(annotations["metadata"], index_path)
    if annotations["metadata"]["coordinate_order"] != "xy" or annotations["metadata"]["keypoints"] != keypoints:
        raise ValueError("The annotation keypoint count or coordinate convention is incorrect.")
    if set(annotations["samples"]) != {sample["id"] for sample in index["samples"]}:
        raise ValueError("Annotation ids differ from the frame index.")
    if any(len(frames) != index["frames"] for frames in annotations["samples"].values()):
        raise ValueError("Every frame must be represented, including empty detections.")
    return annotations["samples"]


def prepare(args):
    if not args.body and not args.faces:
        raise ValueError("Provide --body, --faces, or both.")
    if not 0 <= args.threshold <= 1 or min(args.body_sigma, args.face_sigma) <= 0:
        raise ValueError("Use a threshold in [0,1] and positive heatmap sigmas.")
    index = read_index(args.index)
    branches = {}
    for name, path, count, limbs, heatmap_limbs, sigma in (
        ("person", args.body, 17, BODY_LIMBS, BODY_LIMBS, args.body_sigma),
        ("face", args.faces, 68, FACE_LIMBS, FACE_HEATMAP_LIMBS, args.face_sigma),
    ):
        if path:
            branches[name] = (read_annotations(path, args.index, index, count), count, limbs, heatmap_limbs, sigma)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for sample in index["samples"]:
        name = sample["id"]
        for branch, (annotations, count, limbs, heatmap_limbs, sigma) in branches.items():
            coordinates, adjacency, heatmaps = [], [], []
            for frame in annotations[name]:
                points = valid_keypoints(frame, count, index["image_size"], args.threshold)
                coords, adj = skeleton_targets(points, limbs, args.coordinate_layout)
                coordinates.append(coords)
                adjacency.append(adj)
                heatmaps.append(heatmap_targets(points, heatmap_limbs, index["image_size"], sigma=sigma))
            # Lists retain variable person counts; the PersonQuery loader pads with missing values.
            torch.save(coordinates, output / f"{name}_{branch}_kp.pt")
            torch.save(adjacency, output / f"{name}_{branch}_adj.pt")
            torch.save(torch.stack(heatmaps), output / f"{name}_{branch}_limb.pt")
        print(name, flush=True)
    write_json(output / "metadata.json", metadata(args.index, branches=list(branches),
        coordinate_layout=args.coordinate_layout, coordinate_units="resized_image_pixels",
        threshold=args.threshold, heatmap_size=56, body_sigma=args.body_sigma, face_sigma=args.face_sigma,
        skeleton_limbs={k: v[2] for k, v in branches.items()}, heatmap_limbs={k: v[3] for k, v in branches.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--body", help="Body keypoints from extract_body.py.")
    parser.add_argument("--faces", help="Facial landmarks from extract_faces.py.")
    parser.add_argument("--output", required=True, help="New structural target directory.")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--body-sigma", type=float, default=1.0)
    parser.add_argument("--face-sigma", type=float, default=0.5)
    parser.add_argument("--coordinate-layout", choices=("channels_first", "limb_first"), default="channels_first")
    prepare(parser.parse_args())
