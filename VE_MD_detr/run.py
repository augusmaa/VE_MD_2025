"""Train and validate VE-MD-PersonQuery on one dataset."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
import yaml

from dataset import EmotionDataset, collate_batch
from loss import VEMDLoss
from model import VE_MultiDecoder
from train import train_one_epoch
from validation import validate


def setup_device(name):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if world_size == 1 and torch.backends.mps.is_available() else "cpu"
        )
    if name == "mps" and world_size > 1:
        raise ValueError("Use CPU or CUDA for DDP; MPS runs in one process.")
    if name == "cuda":
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if name == "cuda" else torch.device(name)
    if world_size > 1:
        dist.init_process_group("nccl" if name == "cuda" else "gloo")
    return device, rank, world_size


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    data, architecture, training = config["data"], config["model"], config["training"]
    device, rank, world_size = setup_device(args.device)
    output = Path(args.output)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("This output contains a checkpoint; use --resume or another --output.")
    seed_everything(training["seed"])

    # 1. Model, optimizer and loss.
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    model_options = dict(architecture)
    if checkpoint is not None:
        previous = {k: v for k, v in checkpoint["config"]["model"].items() if k != "pretrained"}
        current = {k: v for k, v in architecture.items() if k != "pretrained"}
        if previous != current:
            raise ValueError("Resume requires the same model configuration.")
        model_options["pretrained"] = False
    model = VE_MultiDecoder(**model_options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    criterion = VEMDLoss(image_size=data["image_size"], **config["loss"])
    start_epoch, best_accuracy = 1, -1.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_accuracy = checkpoint["epoch"] + 1, checkpoint["best_accuracy"]
        training["learning_rate"] = optimizer.param_groups[0]["lr"]
        training["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
    if world_size > 1 and not args.evaluate:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    # 2. One training split and one validation split.
    dataset_options = dict(
        num_classes=architecture["num_classes"], 
        frames=data["frames"], 
        image_size=data["image_size"],
        use_vit=architecture["use_vit"], 
        vit_dim=architecture["vit_dim"],
        use_person=architecture["use_person_pose"], 
        use_face=architecture["use_face_pose"],
        person_limbs=architecture["num_limbs_person"], 
        face_limbs=architecture["num_limbs_face"],
    )
    
    loader_options = dict(batch_size=training["batch_size"], 
                          num_workers=training["workers"],
                          collate_fn=collate_batch, 
                          pin_memory=device.type == "cuda")
    if training["workers"] > 0:
        loader_options["multiprocessing_context"] = "spawn"
    val_data = EmotionDataset(config_path.parent / data["validation"], require_targets=False, **dataset_options)
    # Subset avoids duplicating validation examples when the split is uneven.
    if world_size > 1:
        val_data = Subset(val_data, range(rank, len(val_data), world_size))
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        metrics = validate(model, val_loader, device)
        if rank == 0:
            (output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics, indent=2))
        return
    if start_epoch > training["epochs"]:
        raise ValueError("Increase training.epochs beyond the completed checkpoint epoch.")
    train_data = EmotionDataset(config_path.parent / data["train"], require_targets=True, **dataset_options)
    sampler = DistributedSampler(train_data, seed=training["seed"]) if world_size > 1 else None
    generator = torch.Generator()
    train_loader = DataLoader(train_data, shuffle=sampler is None, sampler=sampler,
                              generator=generator, **loader_options)
    if rank == 0:
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Device: {device} | processes: {world_size} | training samples: {len(train_data)}", flush=True)

    # 3. Train, validate and save at the end of each epoch.
    for epoch in range(start_epoch, training["epochs"] + 1):
        seed_everything(training["seed"] + 100_003 * epoch + rank)
        generator.manual_seed(training["seed"] + epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, training["grad_clip"])
        val_metrics = validate(model, val_loader, device)
        improved = val_metrics["accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, val_metrics["accuracy"])
        if rank == 0:
            record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            network = model.module if hasattr(model, "module") else model
            saved = {"epoch": epoch, "model": network.state_dict(), "optimizer": optimizer.state_dict(),
                     "best_accuracy": best_accuracy, "config": config}
            torch.save(saved, output / "last.tmp")
            (output / "last.tmp").replace(output / "last.pt")
            if improved:
                torch.save(saved, output / "best.pt")
            print(f"Epoch {epoch}/{training['epochs']} | train loss {train_metrics['loss']:.4f} | "
                  f"validation accuracy {val_metrics['accuracy']:.2%} | macro-F1 {val_metrics['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/personquery")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", help="Path to a saved checkpoint.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate --resume on the validation split.")
    args = parser.parse_args()
    if args.evaluate and not args.resume:
        parser.error("--evaluate requires --resume.")
    try:
        main(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
