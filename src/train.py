
"""TV3 - Training and validation pipeline for CIFAR-10."""

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.tensorboard import SummaryWriter


def _plain_config(value: Any) -> Any:
    """Convert config values to basic serializable Python types."""
    if isinstance(value, dict):
        return {
            str(key): _plain_config(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]

    if isinstance(value, (Path, torch.device)):
        return str(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        f"Unsupported config value: {type(value).__name__}"
    )


def _positive_int(config: dict, key: str) -> int:
    """Read and validate a positive integer."""
    value = config.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"config['{key}'] must be a positive integer; "
            f"received {value!r}"
        )

    return value


def _make_optimizer(
    model: nn.Module,
    config: dict,
    learning_rate: float,
    weight_decay: float,
):
    """Create an optimizer using trainable parameters only."""
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise ValueError(
            "No trainable parameters found. "
            "Check the model freeze/fine-tuning configuration."
        )

    name = str(config.get("optimizer") or "adam").lower()

    if name == "adam":
        return Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "adamw":
        return AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "sgd":
        momentum = float(config.get("momentum", 0.9))

        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1).")

        return SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer {name!r}. "
        "Use adam, adamw, or sgd."
    )


def _keep_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Prevent frozen BatchNorm layers from updating running statistics."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))

            if all(not parameter.requires_grad for parameter in parameters):
                module.eval()


def _run_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer=None,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train(training)

    if training:
        _keep_frozen_batchnorm_eval(model)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    phase = "Train" if training else "Validation"
    print(f"{phase}: bắt đầu {len(loader)} batch", flush=True)

    for batch_index, (images, labels) in enumerate(loader, start=1):

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)

            if (
                logits.ndim != 2
                or logits.shape[0] != labels.size(0)
                or logits.shape[1] != 10
            ):
                raise ValueError(
                    "Expected model logits with shape [B, 10], "
                    f"received {tuple(logits.shape)}."
                )

            loss = loss_fn(logits, labels)

            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite loss encountered.")

            if training:
                loss.backward()
                optimizer.step()

        count = labels.size(0)

        # Weight losses by sample count, including the final smaller batch.
        total_loss += float(loss.detach().item()) * count
        total_correct += int(
            (logits.detach().argmax(dim=1) == labels).sum().item()
        )
        total_samples += count

        if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"{phase}: batch {batch_index}/{len(loader)} | "
                f"loss={total_loss / total_samples:.4f} | "
                f"accuracy={total_correct / total_samples:.4f}",
                flush=True,
            )

    if total_samples == 0:
        raise ValueError("The DataLoader is empty.")

    return {
        "loss": float(total_loss / total_samples),
        "accuracy": float(total_correct / total_samples),
    }


def train_model(
    model,
    dataloaders: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Train and validate a CIFAR-10 model.

    Required DataLoader keys: train and val.

    Returns:
        history
        best_checkpoint_path
        best_epoch
        best_val_accuracy
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary.")

    for split in ("train", "val"):
        if split not in dataloaders or dataloaders[split] is None:
            raise KeyError(
                f"dataloaders must contain a '{split}' DataLoader."
            )

    epochs = _positive_int(config, "epochs")
    batch_size = _positive_int(config, "batch_size")

    if config.get("learning_rate") is None:
        raise ValueError("config['learning_rate'] cannot be null.")

    learning_rate = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay") or 0.0)

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")

    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(
            "weight_decay must be finite and non-negative."
        )

    device_name = str(
        config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if device_name == "auto":
        device_name = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    experiment_name = str(
        config.get("experiment_name") or "training_run"
    )

    run_dir = Path(
        config.get("run_dir")
        or Path("outputs") / "runs" / experiment_name
    )
    tensorboard_dir = Path(
        config.get("tensorboard_dir")
        or run_dir / "tensorboard"
    )
    checkpoint_dir = Path(
        config.get("checkpoint_dir")
        or run_dir / "checkpoints"
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "best_model.pth"

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = _make_optimizer(
        model=model,
        config=config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint_config = _plain_config(dict(config))
    checkpoint_config["device"] = str(device)
    checkpoint_config["batch_size"] = batch_size

    history = []
    best_epoch = None
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")
    start_time = time.perf_counter()

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    try:
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(
                model=model,
                loader=dataloaders["train"],
                loss_fn=loss_fn,
                device=device,
                optimizer=optimizer,
            )

            val_metrics = _run_epoch(
                model=model,
                loader=dataloaders["val"],
                loss_fn=loss_fn,
                device=device,
            )

            row = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }

            if not all(
                math.isfinite(row[key])
                for key in (
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy",
                )
            ):
                raise ValueError(
                    f"Non-finite metric at epoch {epoch}."
                )

            history.append(row)

            writer.add_scalar("Loss/train", row["train_loss"], epoch)
            writer.add_scalar("Loss/val", row["val_loss"], epoch)
            writer.add_scalar(
                "Accuracy/train", row["train_accuracy"], epoch
            )
            writer.add_scalar(
                "Accuracy/val", row["val_accuracy"], epoch
            )
            writer.flush()

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"train_loss={row['train_loss']:.4f} "
                f"train_acc={row['train_accuracy']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.4f}"
            )

            # Select by highest validation accuracy, then lowest val loss.
            is_better = (
                best_epoch is None
                or row["val_accuracy"] > best_val_accuracy
                or (
                    row["val_accuracy"] == best_val_accuracy
                    and row["val_loss"] < best_val_loss
                )
            )

            if is_better:
                best_epoch = epoch
                best_val_accuracy = row["val_accuracy"]
                best_val_loss = row["val_loss"]

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": int(epoch),
                        "val_accuracy": float(best_val_accuracy),
                        "val_loss": float(best_val_loss),
                        "config": checkpoint_config,
                    },
                    checkpoint_path,
                )

    finally:
        writer.close()

    if best_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError(
            "Training finished without saving a checkpoint."
        )

    elapsed = time.perf_counter() - start_time

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training time: {elapsed:.2f} seconds")

    return {
        "history": history,
        "best_checkpoint_path": str(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
    }
