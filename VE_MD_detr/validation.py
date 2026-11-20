"""Validation loss, accuracy and macro-F1 over the complete validation split."""

import torch
import torch.distributed as dist
from torch.nn import functional as F


def classification_metrics(confusion):
    """Confusion rows are true labels; macro-F1 includes every configured class."""
    matrix = confusion.to(device="cpu", dtype=torch.float64)
    count = int(matrix.sum())
    true_positive = matrix.diag()
    denominator = matrix.sum(0) + matrix.sum(1)
    per_class_f1 = 2 * true_positive / denominator.clamp_min(1)
    return {
        "count": count,
        "accuracy": true_positive.sum().item() / count if count else 0.0,
        "macro_f1": per_class_f1.mean().item(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": confusion.cpu().tolist(),
    }


@torch.no_grad()
def validate(model, loader, device):
    model = model.module if hasattr(model, "module") else model
    model.eval()
    if dist.is_initialized():
        # Use identical BatchNorm buffers before evaluating disjoint rank subsets.
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)

    classes = model.num_classes
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    total_loss = torch.zeros(1, dtype=torch.float64)
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["label_emotion"].to(device)
        features = batch["feat_vit"].to(device) if "feat_vit" in batch else None
        logits = model(images, features)["preds_emotion"]
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss.")
        indices = labels.cpu() * classes + logits.argmax(1).cpu()
        confusion += torch.bincount(indices, minlength=classes**2).reshape(classes, classes)
        total_loss += loss.item() * labels.size(0)

    # Reduce counts, not per-batch or per-process F1 scores.
    if dist.is_initialized():
        confusion, total_loss = confusion.to(device), total_loss.to(device)
        dist.all_reduce(confusion)
        dist.all_reduce(total_loss)
    result = classification_metrics(confusion)
    if result["count"] == 0:
        raise ValueError("The validation split is empty.")
    result["loss"] = total_loss.item() / result["count"]
    return result
