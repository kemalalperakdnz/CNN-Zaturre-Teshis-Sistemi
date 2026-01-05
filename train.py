import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision import datasets, models, transforms


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    arch: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    backbone_lr_mult: float
    weight_decay: float
    num_workers: int
    patience: int
    seed: int
    amp: bool
    freeze_epochs: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_tfms, eval_tfms


def build_model(arch: str, num_classes: int, device: torch.device) -> nn.Module:
    arch = arch.lower()

    if arch == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)

    if arch in {"efficientnet_b0", "efficientnet-b0"}:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model.to(device)

    raise ValueError(f"Unsupported arch: {arch}. Use resnet18 or efficientnet_b0")


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    arch = arch.lower()
    for p in model.parameters():
        p.requires_grad = trainable

    if arch == "resnet18":
        for p in model.fc.parameters():
            p.requires_grad = True
        if trainable:
            for p in model.fc.parameters():
                p.requires_grad = True
        return

    if arch in {"efficientnet_b0", "efficientnet-b0"}:
        for p in model.classifier.parameters():
            p.requires_grad = True
        if trainable:
            for p in model.classifier.parameters():
                p.requires_grad = True
        return


def build_optimizer(model: nn.Module, arch: str, lr: float, backbone_lr_mult: float, weight_decay: float) -> optim.Optimizer:
    arch = arch.lower()
    if arch == "resnet18":
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    elif arch in {"efficientnet_b0", "efficientnet-b0"}:
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.") and p.requires_grad]
    else:
        head_params = [p for p in model.parameters() if p.requires_grad]
        backbone_params = []

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    return optim.AdamW(param_groups, weight_decay=weight_decay)


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, targets)
        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = float(accuracy_score(y_true, y_pred))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    return avg_loss, acc, macro_f1


def build_weighted_sampler(train_dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="chest_xray")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arch=args.arch,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        patience=args.patience,
        seed=args.seed,
        amp=not args.no_amp,
        freeze_epochs=args.freeze_epochs,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(cfg.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_tfms, eval_tfms = build_transforms(cfg.image_size)

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tfms)

    num_classes = len(train_ds.classes)
    if num_classes != 2:
        raise ValueError(f"Expected 2 classes (NORMAL, PNEUMONIA). Got {num_classes}: {train_ds.classes}")

    sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg.arch, num_classes=num_classes, device=device)

    loss_fn = nn.CrossEntropyLoss()

    set_backbone_trainable(model, cfg.arch, trainable=(cfg.freeze_epochs <= 0))
    optimizer = build_optimizer(model, cfg.arch, lr=cfg.lr, backbone_lr_mult=cfg.backbone_lr_mult, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, verbose=False)

    try:
        scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "arch": cfg.arch,
        "image_size": cfg.image_size,
    }

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs + 1:
            set_backbone_trainable(model, cfg.arch, trainable=True)
            optimizer = build_optimizer(
                model,
                cfg.arch,
                lr=cfg.lr,
                backbone_lr_mult=cfg.backbone_lr_mult,
                weight_decay=cfg.weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, verbose=False)

        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda" and scaler.is_enabled():
                try:
                    autocast_ctx = torch.amp.autocast("cuda")
                except Exception:
                    autocast_ctx = torch.cuda.amp.autocast()
            else:
                autocast_ctx = torch.autocast(device_type=device.type, enabled=False)

            with autocast_ctx:
                logits = model(images)
                loss = loss_fn(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            pbar.set_postfix({"train_loss": float(np.mean(train_losses))})

        val_loss, val_acc, val_f1 = run_eval(model, val_loader, device)
        scheduler.step(val_f1)

        ckpt = {
            "model_state": model.state_dict(),
            "meta": meta,
            "config": asdict(cfg),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }

        torch.save(ckpt, output_dir / "last.pt")

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(ckpt, output_dir / "best.pt")
        else:
            epochs_no_improve += 1

        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "best_val_f1": float(best_val_f1),
            "best_epoch": int(best_epoch),
            "lrs": [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups],
        }
        with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        print(json.dumps(metrics))

        if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
            print(f"Early stopping: no improvement for {cfg.patience} epoch(s).")
            break

    print(f"Done. Best val macro-F1: {best_val_f1:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
