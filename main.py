"""TV6 — tích hợp pipeline Lab 2."""

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import build_dataloaders
from src.evaluate import evaluate_model
from src.models import build_model
from src.train import train_model
from src.utils import (
    get_device,
    load_config,
    save_json,
    set_seed,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lab 2 — mô hình pre-trained trên CIFAR-10."
    )
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-config",
        action="store_true",
        help="Chỉ kiểm tra cấu hình, không huấn luyện.",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Kiểm tra pipeline với DenseNet121 và vài mẫu trên CPU.",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)

    try:
        config = load_config(config_path)
        validate_config(config)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        parser.exit(2, f"Config error: {error}\n")

    if args.check_config:
        print(f"Config: {config_path}")
        print(f"Experiment: {config['experiment_name']}")
        print("Basic config validation: OK")
        return 0

    # Chỉ thay bản cấu hình trong bộ nhớ, không sửa YAML gốc.
    config = dict(config)
    config["device"] = get_device()

    if args.smoke_test:
        config.update({
            "experiment_name": "integration_smoke_densenet",
            "model_name": "densenet121",
            "pretrained": False,
            "strategy": "classifier_only",
            "finetune_layers": [],
            "num_classes": 10,
            "batch_size": 2,
            "epochs": 1,
            "image_size": 32,
            "num_workers": 0,
            "device": "cpu",
        })

    validate_config(config)
    set_seed(config["seed"])

    if config["device"] == "cpu":
        torch.set_num_threads(1)

    name = config["experiment_name"]

    if args.smoke_test:
        # Mỗi smoke test có thư mục riêng trong outputs/tmp.
        from src.utils import create_run_dir

        run_dir = create_run_dir(
            name, PROJECT_ROOT / "outputs" / "tmp"
        )
        checkpoint_dir = run_dir / "checkpoints"
    else:
        # Khớp đường dẫn mà TV5 hiện đang đọc.
        run_dir = PROJECT_ROOT / "outputs" / "runs" / name
        checkpoint_dir = (
            PROJECT_ROOT / "outputs" / "checkpoints" / name
        )

        # Không ghi đè kết quả hoặc checkpoint đã tồn tại.
        if (
            (run_dir / "metrics.json").exists()
            or (checkpoint_dir / "best_model.pth").exists()
        ):
            parser.exit(
                2,
                f"Output already exists for {name}. "
                "Archive the previous run before rerunning.\n",
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config.update({
        "data_dir": str(resolve_path(config.get("data_dir", "./data"))),
        "run_dir": str(run_dir),
        "log_dir": str(run_dir),
        "tensorboard_dir": str(run_dir / "tensorboard"),
        "checkpoint_dir": str(checkpoint_dir),
    })

    save_json(run_dir / "config.json", config)

    print(f"Experiment: {name}")
    print(f"Device: {config['device']}")
    print(f"Smoke test: {args.smoke_test}")
    print(f"Run directory: {run_dir}")

    loaders = build_dataloaders(config)

    if config.get("num_classes", 10) != loaders["num_classes"]:
        raise ValueError("num_classes không khớp với dataset.")

    if args.smoke_test:
        for split, count in (("train", 4), ("val", 2), ("test", 2)):
            dataset = loaders[split].dataset
            loaders[split] = DataLoader(
                Subset(dataset, range(min(count, len(dataset)))),
                batch_size=2,
                shuffle=(split == "train"),
                num_workers=0,
            )

    model = build_model(config)

    started = time.perf_counter()
    result = train_model(model, loaders, config)
    training_time = time.perf_counter() - started

    history = result["history"]
    best_row = next(
        row for row in history
        if row["epoch"] == result["best_epoch"]
    )

    with (run_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "experiment_name": name,
        "smoke_test": args.smoke_test,
        "best_epoch": result["best_epoch"],
        "best_val_acc": best_row["val_accuracy"],
        "best_val_loss": best_row["val_loss"],
        "training_time_sec": training_time,
        "best_checkpoint_path": result["best_checkpoint_path"],
    }

    if args.smoke_test:
        checkpoint = torch.load(
            result["best_checkpoint_path"],
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = evaluate_model(
            model, loaders["test"], {"device": "cpu"}
        )

        if not all(
            torch.isfinite(torch.tensor(value)).item()
            for value in test_metrics.values()
        ):
            raise ValueError("Smoke test trả về metric không hữu hạn.")

        metrics["smoke_test_metrics"] = test_metrics
        print("Smoke test metrics:", test_metrics)

    save_json(run_dir / "metrics.json", metrics)

    print(f"History: {run_dir / 'history.csv'}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    print(
        "ENTRY POINT SMOKE TEST: PASSED"
        if args.smoke_test
        else "TRAINING AND VALIDATION: COMPLETED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())