"""One training epoch on one dataset."""

import torch
import torch.distributed as dist


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    names = ("loss", "classification", "structural", "mmd")
    totals = torch.zeros(6, dtype=torch.float64)

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["label_emotion"]
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch["images"], batch.get("feat_vit"))
        loss, terms = criterion(outputs, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()

        batch_size = labels.size(0)
        correct = (outputs["preds_emotion"].argmax(1) == labels).sum().item()
        totals += torch.tensor([terms[name].item() * batch_size for name in names]
                               + [correct, batch_size], dtype=torch.float64)

    if dist.is_initialized():
        totals = totals.to(device)
        dist.all_reduce(totals)
    totals = totals.cpu()
    count = int(totals[-1])
    if count == 0:
        raise ValueError("The training loader is empty.")
    result = {name: totals[index].item() / count for index, name in enumerate(names)}
    result.update(accuracy=totals[4].item() / count, count=count)
    return result
