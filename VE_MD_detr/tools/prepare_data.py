#!/usr/bin/env python3
"""Describe the original prepared datasets with explicit paths and zero-based labels.

This writes JSON manifests only; it does not modify data, resample frames,
extract ViT features, or generate skeleton annotations.
"""

import argparse
import csv
import json
from pathlib import Path


MER_LABELS = {"worried": 0, "happy": 1, "neutral": 2, "sad": 3, "angry": 4, "surprise": 5}
SAMSEMO_LABELS = {"happiness": 0, "neutral": 1, "sadness": 2, "disgust": 3, "anger": 3, "surprise": 4}


def path_in(directory, filename):
    return str((Path(directory) / filename).resolve())


def read_table(filename, whitespace=False, delimiter=","):
    with Path(filename).open(newline="") as handle:
        if whitespace:
            lines = [line.split() for line in handle if line.strip()]
            return [dict(zip(lines[0], row)) for row in lines[1:]]
        return list(csv.DictReader(handle, delimiter=delimiter))


def prepare(args):
    train = []
    for item in json.loads(Path(args.train_annotations).read_text()):
        name = item["file_name"]
        label = int(item["label_emotion"]) - (1 if args.dataset == "vgaf" else 0)
        record = {"label": label}
        record["images"] = (path_in(args.train_image_dir, name) if args.dataset == "gaf3"
                            else path_in(args.train_dir, name + ".pt"))
        feature_name = name.split(".")[0] if args.dataset in ("vgaf", "gaf3", "mer") else name
        if args.train_vit_dir:
            record["feat_vit"] = path_in(args.train_vit_dir, feature_name + ".vit_feat.pt")
        for branch in ("person", "face"):
            if (branch == "person" and args.structure not in ("body", "both")) or (
                branch == "face" and args.structure not in ("face", "both")
            ):
                continue
            record[f"skeleton_{branch}"] = path_in(args.train_dir, name + f"_{branch}_kp.pt")
            record[f"adjacency_{branch}"] = path_in(args.train_dir, name + f"_{branch}_adj.pt")
        train.append(record)
    rows = read_table(args.val_labels, whitespace=args.dataset == "vgaf",
                      delimiter="\t" if args.dataset == "samsemo" else ",")
    validation = []
    for row in rows:
        if args.dataset == "vgaf":
            name, label = row["Vid_name"], int(row["Label"]) - 1
            key, feature_name = name + ".mp4.img", name
        elif args.dataset == "gaf3":
            name, label = row["Img_name"], int(row["Labels"])
            key, feature_name = None, name.split(".")[0]
        elif args.dataset == "mer":
            name, label = row["name"], MER_LABELS[row["discrete"]]
            key, feature_name = name + ".img", name
        elif args.dataset == "dfew":
            name, label = str(row["video_name"]).zfill(5), int(row["label"]) - 1
            key, feature_name = name, name
        elif args.dataset == "samsemo":
            name, label = row["utterance_id"], SAMSEMO_LABELS[row["aggregated_emotions"]]
            key, feature_name = name, name
        else:
            name, label = row["chunk"], int(row["label"])
            key = name + ".img"
            # The supplied EngageNet validation code includes '.img' here.
            feature_name = key
        record = {"label": label, "images": path_in(args.val_data, name) if key is None else {
            "path": str(Path(args.val_data).resolve()), "key": key,
        }}
        if args.val_vit_dir:
            record["feat_vit"] = path_in(args.val_vit_dir, feature_name + ".vit_feat.pt")
        validation.append(record)
    if not train or not validation:
        raise ValueError("Both annotation tables must contain samples.")
    # Catch path errors before a long training run; HDF5 keys are validated on reading.
    for record in train + validation:
        for name, value in record.items():
            if name != "label":
                path = value["path"] if isinstance(value, dict) else value
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Missing {name}: {path}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in (("train.json", train), ("val.json", validation)):
        target = output / filename
        if target.exists():
            raise FileExistsError(f"{target} exists. Choose a new --output.")
    for filename, records in (("train.json", train), ("val.json", validation)):
        (output / filename).write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {len(train)} training and {len(validation)} validation records to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vgaf", "gaf3", "mer", "dfew", "samsemo", "engagenet"), required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--train-dir", required=True, help="Prepared video/skeleton .pt directory.")
    parser.add_argument("--train-image-dir", help="GAF-3.0 RGB image directory.")
    parser.add_argument("--train-vit-dir")
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--val-data", required=True, help="HDF5 file, or GAF-3.0 image directory.")
    parser.add_argument("--val-vit-dir")
    parser.add_argument("--structure", choices=("body", "face", "both", "none"))
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.structure is None:
        arguments.structure = "face" if arguments.dataset == "dfew" else "body"
    if arguments.dataset == "gaf3" and not arguments.train_image_dir:
        parser.error("GAF-3.0 requires --train-image-dir.")
    prepare(arguments)
