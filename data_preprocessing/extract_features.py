"""Extract one frozen ViT feature vector per saved frame."""

import argparse
from pathlib import Path

import torch

from data import (IMAGENET_MEAN, IMAGENET_STD, file_hash, image_tensor, metadata,
                  read_frames, read_index, select_device, write_json)


def load_backbone(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict", "model"):
        if key in state:
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    # The supplied VideoVitNet checkpoints contain vit.*, att_pooling.* and linear_out.*.
    if any(key.startswith("vit.") for key in state):
        state = {key[4:]: value for key, value in state.items() if key.startswith("vit.")}
    state = {key: value for key, value in state.items() if not key.startswith("head.")}
    model.load_state_dict(state, strict=True)


@torch.inference_mode()
def extract(args):
    import timm

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    index = read_index(args.index)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=0)
    if args.checkpoint:
        load_backbone(model, args.checkpoint)
    if tuple(model.pretrained_cfg["input_size"][-2:]) != (index["image_size"],) * 2:
        raise ValueError("The saved frame size differs from this ViT's configured input size.")
    device = select_device(args.device)
    model = model.eval().requires_grad_(False).to(device)
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    if args.normalization == "model":
        mean, std = model.pretrained_cfg["mean"], model.pretrained_cfg["std"]
    output.mkdir(parents=True)
    for sample in index["samples"]:
        images = image_tensor(read_frames(args.index, sample, index), mean, std)
        features = torch.cat([model(batch.to(device)).cpu() for batch in images.split(args.batch_size)])
        if features.shape != (index["frames"], model.num_features) or not torch.isfinite(features).all():
            raise ValueError(f"{sample['id']}: expected finite per-frame feature vectors.")
        torch.save(features.float().contiguous(), output / f"{sample['id']}.vit_feat.pt")
        print(f"{sample['id']}: {tuple(features.shape)}", flush=True)
    write_json(output / "metadata.json", metadata(args.index, model=args.model,
        timm_version=timm.__version__, feature_dim=model.num_features,
        checkpoint=str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        checkpoint_sha256=file_hash(args.checkpoint) if args.checkpoint else None,
        pretrained=args.pretrained, normalization=args.normalization, mean=list(mean), std=list(std)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="frames.json from extract_frames.py.")
    parser.add_argument("--output", required=True, help="New feature directory.")
    parser.add_argument("--model", default="vit_large_patch14_clip_224.laion2b")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", help="Explicit local ViT or VideoVitNet checkpoint.")
    weights.add_argument("--pretrained", action="store_true", help="Use published timm weights.")
    parser.add_argument("--normalization", choices=("imagenet", "model"), default="imagenet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    extract(parser.parse_args())
