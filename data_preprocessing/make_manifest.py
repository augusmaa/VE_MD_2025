"""Write data lists that can be used directly by both VE-MD training repositories."""

import argparse
import json
import os
from pathlib import Path

from data import check_source, read_index, write_json


def make_manifests(args):
    index_path = Path(args.index).resolve()
    index = read_index(index_path)
    output = Path(args.output).resolve()
    for name in ("personquery.json", "heatmap.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    features = Path(args.features).resolve() if args.features else None
    targets = Path(args.targets).resolve() if args.targets else None
    branches = []
    for directory in (features, targets):
        if directory:
            info = json.loads((directory / "metadata.json").read_text())
            check_source(info, index_path)
            if directory == targets:
                branches = info["branches"]
                print(f"PersonQuery coordinate_layout: {info['coordinate_layout']}")

    def relative(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        return os.path.relpath(path, output)

    personquery, heatmap = [], []
    for sample in index["samples"]:
        record = {"id": sample["id"], "label": sample["label"], "images": {
            "path": relative(index_path.parent / sample["images"]["path"]), "key": sample["images"]["key"]}}
        if features:
            record["feat_vit"] = relative(features / f"{sample['id']}.vit_feat.pt")
        query_record, heatmap_record = dict(record), dict(record)
        for branch in branches:
            query_record[f"skeleton_{branch}"] = relative(targets / f"{sample['id']}_{branch}_kp.pt")
            query_record[f"adjacency_{branch}"] = relative(targets / f"{sample['id']}_{branch}_adj.pt")
            heatmap_record[f"limb_{branch}"] = relative(targets / f"{sample['id']}_{branch}_limb.pt")
        personquery.append(query_record)
        heatmap.append(heatmap_record)
    write_json(output / "personquery.json", personquery)
    write_json(output / "heatmap.json", heatmap)
    print(f"Wrote {len(personquery)} samples to each training data list.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--features", help="Feature directory; omit only when training with use_vit: false.")
    parser.add_argument("--targets", help="Target directory; omit for validation or classification-only training.")
    parser.add_argument("--output", required=True, help="Directory for personquery.json and heatmap.json.")
    make_manifests(parser.parse_args())
