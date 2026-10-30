"""TV4 — đánh giá mô hình đã được chọn trên tập test."""

from typing import Any

import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(
    model,
    test_loader,
    config: dict[str, Any],
) -> dict[str, float]:
    """Đánh giá mô hình trên test và trả về các metrics thực đo."""

    device_name = config.get("device") or "auto"

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu CUDA nhưng máy không có CUDA khả dụng.")

    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            logits = model(images)

            if logits.ndim != 2 or logits.shape != (labels.size(0), 10):
                raise ValueError(
                    "CIFAR-10 yêu cầu logits có shape [B, 10], "
                    f"nhận được {tuple(logits.shape)}."
                )

            if not torch.isfinite(logits).all().item():
                raise ValueError("Logits chứa NaN hoặc Inf.")

            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise ValueError("Evaluation loss chứa NaN hoặc Inf.")

            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    if total_samples == 0:
        raise ValueError("Tập test không có dữ liệu.")

    return {
        "test_loss": float(total_loss / total_samples),
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "precision": float(precision_score(
            all_labels, all_predictions,
            average="weighted", zero_division=0,
        )),
        "recall": float(recall_score(
            all_labels, all_predictions,
            average="weighted", zero_division=0,
        )),
        "f1_score": float(f1_score(
            all_labels, all_predictions,
            average="weighted", zero_division=0,
        )),
    }
