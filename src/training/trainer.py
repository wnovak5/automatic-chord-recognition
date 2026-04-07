from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

IGNORE_INDEX = -100


def _num_model_classes(model: nn.Module) -> int:
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        return int(model.classifier.out_features)
    if hasattr(model, "net") and len(model.net) > 0 and isinstance(model.net[-1], nn.Linear):
        return int(model.net[-1].out_features)
    raise AttributeError("Could not infer the model's output dimension")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(y)

    return {"loss": total_loss / total, "accuracy": correct / total}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += len(y)

    return {"loss": total_loss / total, "accuracy": correct / total}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    patience: int = 10,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Compute class weights from training set labels
    all_labels = torch.cat([y for _, y in train_loader])
    num_classes = _num_model_classes(model)
    counts = torch.bincount(all_labels, minlength=num_classes).float()
    weights = 1.0 / counts.clamp(min=1.0)
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    for epoch in range(num_epochs):
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr["loss"])
        val_losses.append(va["loss"])
        train_accs.append(tr["accuracy"])
        val_accs.append(va["accuracy"])

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs}  "
                f"train_loss={tr['loss']:.4f}  train_acc={tr['accuracy']:.3f}  "
                f"val_loss={va['loss']:.4f}  val_acc={va['accuracy']:.3f}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch+1})")
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
    }


def train_sequence_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ignore_index: int = IGNORE_INDEX,
    progress_interval: int | None = None,
    progress_prefix: str = "train",
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    started_at = time.time()

    for batch_index, (x, y, lengths) in enumerate(loader, start=1):
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths=lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        valid_mask = y != ignore_index
        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * int(valid_mask.sum().item())
        correct += int(((preds == y) & valid_mask).sum().item())
        total += int(valid_mask.sum().item())

        if progress_interval is not None and (
            batch_index == 1 or batch_index % progress_interval == 0 or batch_index == num_batches
        ):
            elapsed = time.time() - started_at
            avg_seconds = elapsed / batch_index
            remaining = (num_batches - batch_index) * avg_seconds
            running_loss = total_loss / max(total, 1)
            running_acc = correct / max(total, 1)
            print(
                f"  [{progress_prefix}] batch {batch_index}/{num_batches} "
                f"loss={running_loss:.4f} acc={running_acc:.3f} "
                f"elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m"
            )

    return {"loss": total_loss / total, "accuracy": correct / total}


def evaluate_sequence(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ignore_index: int = IGNORE_INDEX,
    progress_interval: int | None = None,
    progress_prefix: str = "val",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    started_at = time.time()

    with torch.no_grad():
        for batch_index, (x, y, lengths) in enumerate(loader, start=1):
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits = model(x, lengths=lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            valid_mask = y != ignore_index
            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * int(valid_mask.sum().item())
            correct += int(((preds == y) & valid_mask).sum().item())
            total += int(valid_mask.sum().item())

            if progress_interval is not None and (
                batch_index == 1 or batch_index % progress_interval == 0 or batch_index == num_batches
            ):
                elapsed = time.time() - started_at
                avg_seconds = elapsed / batch_index
                remaining = (num_batches - batch_index) * avg_seconds
                running_loss = total_loss / max(total, 1)
                running_acc = correct / max(total, 1)
                print(
                    f"  [{progress_prefix}] batch {batch_index}/{num_batches} "
                    f"loss={running_loss:.4f} acc={running_acc:.3f} "
                    f"elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m"
                )

    return {"loss": total_loss / total, "accuracy": correct / total}


def train_sequence(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    patience: int = 10,
    ignore_index: int = IGNORE_INDEX,
    progress_interval: int | None = None,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_labels = torch.cat([y[y != ignore_index] for _, y, _ in train_loader])
    num_classes = _num_model_classes(model)
    counts = torch.bincount(all_labels, minlength=num_classes).float()
    weights = 1.0 / counts.clamp(min=1.0)
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=ignore_index)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        tr = train_sequence_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            ignore_index=ignore_index,
            progress_interval=progress_interval,
            progress_prefix="train",
        )
        va = evaluate_sequence(
            model,
            val_loader,
            criterion,
            device,
            ignore_index=ignore_index,
            progress_interval=progress_interval,
            progress_prefix="val",
        )

        train_losses.append(tr["loss"])
        val_losses.append(va["loss"])
        train_accs.append(tr["accuracy"])
        val_accs.append(va["accuracy"])

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:3d}/{num_epochs}  "
            f"train_loss={tr['loss']:.4f}  train_acc={tr['accuracy']:.3f}  "
            f"val_loss={va['loss']:.4f}  val_acc={va['accuracy']:.3f}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch+1})")
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
    }


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    torch.save(state, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")
