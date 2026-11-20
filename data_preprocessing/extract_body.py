"""Extract COCO-17 body keypoints with the original ViTPose-ONNX backend."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from data import file_hash, metadata, read_frames, read_index, write_json


def to_xy(groups):
    # A single-frame inference returns its detections followed by an empty split.
    if len(groups) == 0:
        return []
    points = np.asarray(groups[0], dtype=np.float32)
    if points.size == 0:
        return []
    if points.ndim != 3 or points.shape[1:] != (17, 3) or not np.isfinite(points).all():
        raise ValueError("Expected COCO-17 ViTPose output with shape (people,17,3).")
    # Upstream inference returns (y,x,score); all saved annotations use (x,y,score).
    return points[:, :, [1, 0, 2]].tolist()


def extract(args):
    import onnxruntime as ort

    if not 0 <= args.detection_threshold <= 1:
        raise ValueError("--detection-threshold must be in [0,1].")
    repository = Path(args.repo).resolve()
    if not (repository / "utils/inference.py").is_file():
        raise FileNotFoundError("--repo must point to a ViTPose-ONNX checkout.")
    for path in (args.detector, args.pose):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(repository))
    from configs.base_config import cfg
    from utils.inference import inference

    config = cfg.clone()
    config.no_bbox = config.no_skeleton = True
    config.result_scale = None
    config.yolo_batch_size = config.pose_batch_size = 1
    config.conf_thres = args.detection_threshold
    providers = ["CPUExecutionProvider"]
    if args.device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA body extraction requires onnxruntime-gpu and its CUDA dependencies.")
        providers.insert(0, "CUDAExecutionProvider")
    detector = ort.InferenceSession(str(Path(args.detector).resolve()), providers=providers)
    pose = ort.InferenceSession(str(Path(args.pose).resolve()), providers=providers)
    index = read_index(args.index)
    samples = {}
    for sample in index["samples"]:
        annotations = []
        for frame in read_frames(args.index, sample, index):
            _, predictions = inference(frame[None], detector, pose, config)
            annotations.append(to_xy(predictions[1]))
        samples[sample["id"]] = annotations
        print(sample["id"], flush=True)
    revision = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    info = metadata(args.index, coordinate_order="xy", keypoints=17, backend="ViTPose-ONNX",
                    revision=revision.stdout.strip() if revision.returncode == 0 else None,
                    detector_sha256=file_hash(args.detector), pose_sha256=file_hash(args.pose),
                    detection_threshold=args.detection_threshold)
    write_json(args.output, {"metadata": info, "samples": samples})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--repo", required=True, help="Path to the external ViTPose-ONNX repository.")
    parser.add_argument("--detector", required=True, help="YOLOv6 ONNX weights.")
    parser.add_argument("--pose", required=True, help="ViTPose COCO-17 ONNX weights.")
    parser.add_argument("--output", required=True, help="New body keypoint JSON file.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    extract(parser.parse_args())
