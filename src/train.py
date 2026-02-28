"""Training loop for PulmoSight-CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
from torch.optim import Adam

from data import build_transforms, create_dataloaders, inspect_batch, plot_class_distribution
from evaluate import evaluate_model
from models import SimpleLungCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-class lung histopathology CNN")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        probs = model(images)
        loss = criterion(torch.log(probs + 1e-8), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = probs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            probs = model(images)
            loss = criterion(torch.log(probs + 1e-8), labels)

            total_loss += loss.item() * images.size(0)
            preds = probs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_training_curves(history: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    artifacts_dir = Path(config["output"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    transforms = build_transforms(tuple(config["data"]["image_size"]))
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=config["data"]["dataset_dir"],
        transforms_by_split=transforms,
        batch_size=config["training"]["batch_size"],
        val_split=config["data"]["val_split"],
        num_workers=config["training"]["num_workers"],
        seed=config["training"]["seed"],
    )

    plot_class_distribution(
        config["data"]["dataset_dir"],
        save_path=str(artifacts_dir / "class_distribution.png"),
    )

    preview_images, preview_labels = next(iter(train_loader))
    inspect_batch(
        preview_images,
        preview_labels,
        class_names,
        save_path=str(artifacts_dir / "sample_batch.png"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLungCNN(num_classes=len(class_names)).to(device)

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), artifacts_dir / config["output"]["weights_name"])

    save_training_curves(history, artifacts_dir)

    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        class_names=class_names,
        device=device,
        output_dir=str(artifacts_dir),
    )

    metrics_path = artifacts_dir / "metrics.yaml"
    with metrics_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"history": history, "evaluation": metrics}, f)

    print(f"Training complete. Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
