import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


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
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pt")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(Path(args.checkpoint), map_location="cpu")
    meta = ckpt.get("meta", {})

    classes = meta.get("classes")
    arch = meta.get("arch", "resnet18")
    image_size = int(meta.get("image_size", 224))

    if not classes:
        raise ValueError("Checkpoint meta is missing classes. Train again to generate a valid checkpoint.")

    tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(args.image).convert("RGB")
    x = tfms(img).unsqueeze(0)

    model = build_model(arch=arch, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred_label = classes[pred_idx]
    pred_prob = float(probs[pred_idx].item())

    print({"pred": pred_label, "prob": pred_prob, "probs": {classes[i]: float(probs[i]) for i in range(len(classes))}})


if __name__ == "__main__":
    main()
