import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, models, transforms


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()

    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch in {"efficientnet_b0", "efficientnet-b0"}:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported arch: {arch}")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    return y_true, y_pred, y_prob


def save_confusion_matrix(cm: np.ndarray, labels, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="chest_xray")
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})

    classes = meta.get("classes")
    class_to_idx = meta.get("class_to_idx")
    arch = meta.get("arch", "resnet18")
    image_size = int(meta.get("image_size", 224))

    if not classes or not class_to_idx:
        raise ValueError("Checkpoint meta is missing classes/class_to_idx. Train again to generate a valid checkpoint.")

    tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dir = Path(args.data_dir) / "test"
    test_ds = datasets.ImageFolder(str(test_dir), transform=tfms)

    if test_ds.class_to_idx != class_to_idx:
        raise ValueError(
            f"Class mapping mismatch. Checkpoint: {class_to_idx}, test_ds: {test_ds.class_to_idx}. "
            "Ensure folder names and ordering are consistent."
        )

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(arch=arch, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    y_true, y_pred, y_prob = predict(model, loader, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_prob.npy", y_prob)

    if len(classes) == 2:
        pneumonia_idx = int(class_to_idx.get("PNEUMONIA", 1))
        auc = float(roc_auc_score((y_true == pneumonia_idx).astype(int), y_prob[:, pneumonia_idx]))
        print("ROC-AUC (PNEUMONIA):", round(auc, 4))

        if args.threshold is not None:
            thr = float(args.threshold)
            y_pred = (y_prob[:, pneumonia_idx] >= thr).astype(int)
            print("Using threshold for PNEUMONIA:", thr)

    print("Classes:", classes)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")

    np.save(out_dir / "confusion_matrix.npy", cm)


if __name__ == "__main__":
    main()
