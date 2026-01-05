import os
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from PIL import Image
from torchvision import models, transforms
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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


def load_checkpoint(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    meta = ckpt.get("meta", {})

    classes = meta.get("classes")
    arch = meta.get("arch", "resnet18")
    image_size = int(meta.get("image_size", 224))

    if not classes:
        raise ValueError("Checkpoint meta is missing classes. Train again to generate a valid checkpoint.")

    model = build_model(arch=arch, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)

    tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return model, tfms, classes


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev")

    root = Path(__file__).resolve().parent
    checkpoint_path = Path(os.environ.get("CHECKPOINT", str(root / "outputs_v2" / "best.pt")))
    uploads_dir = Path(os.environ.get("UPLOADS_DIR", str(root / "uploads")))
    uploads_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tfms, classes = load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    @app.get("/")
    def index():
        return render_template("index.html", result=None, image_url=None, classes=classes)

    @app.post("/predict")
    def predict():
        if "file" not in request.files:
            flash("Dosya bulunamadı.")
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "":
            flash("Dosya seçilmedi.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Sadece png/jpg/jpeg/webp dosyaları yükleyebilirsiniz.")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        save_path = uploads_dir / filename
        file.save(save_path)

        img = Image.open(save_path).convert("RGB")
        x = tfms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        pred_idx = int(torch.argmax(probs).item())
        pred_label = classes[pred_idx]
        pred_prob = float(probs[pred_idx].item())

        probs_dict = {classes[i]: float(probs[i].item()) for i in range(len(classes))}

        image_url = url_for("uploaded_file", filename=filename)
        result = {
            "pred": pred_label,
            "prob": pred_prob,
            "probs": probs_dict,
            "checkpoint": str(checkpoint_path),
            "device": str(device),
        }

        return render_template("index.html", result=result, image_url=image_url, classes=classes)

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(uploads_dir, filename)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
