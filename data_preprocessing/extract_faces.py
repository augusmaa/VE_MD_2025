"""Extract 68 facial landmarks from every saved RGB frame."""

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np

from data import metadata, read_frames, read_index, write_json


def extract(args):
    import face_alignment

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                              device=args.device, flip_input=False)
    index = read_index(args.index)
    samples = {}
    for sample in index["samples"]:
        annotations = []
        for frame in read_frames(args.index, sample, index):
            faces = detector.get_landmarks(frame)
            points = []
            for face in faces if faces is not None else []:
                face = np.asarray(face, dtype=np.float32)
                if face.shape != (68, 2) or not np.isfinite(face).all():
                    raise ValueError("Expected finite (68,2) facial landmarks.")
                # This API returns locations only; 1 denotes a detected landmark, not calibrated confidence.
                points.append(np.column_stack((face, np.ones(68))).tolist())
            annotations.append(points)
        samples[sample["id"]] = annotations
        print(sample["id"], flush=True)
    info = metadata(args.index, coordinate_order="xy", keypoints=68, backend="face-alignment",
                    version=version("face-alignment"), flip_input=False, face_detector="sfd")
    write_json(args.output, {"metadata": info, "samples": samples})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True, help="New face keypoint JSON file.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    extract(parser.parse_args())
