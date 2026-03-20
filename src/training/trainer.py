from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
    num_classes = model.net[-1].out_features
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


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    torch.save(state, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")
