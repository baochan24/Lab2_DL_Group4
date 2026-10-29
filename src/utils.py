"""TV6 — tiện ích dùng chung cho pipeline Lab 2."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
import math
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Đặt seed để hỗ trợ tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Seed không bảo đảm kết quả giống tuyệt đối giữa mọi thiết bị.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> str:
    """Trả về CUDA nếu khả dụng, ngược lại dùng CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc YAML; kiểm tra cấu trúc cơ bản.

    Kiểm tra các trường bắt buộc cho huấn luyện sẽ được bổ sung
    sau khi nhóm chốt schema cấu hình.
    """
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Cấu hình phải là một dictionary YAML: {config_path}"
        )

    return config


def create_run_dir(
    experiment_name: str,
    output_root: str | Path = "outputs/runs",
) -> Path:
    """Tạo thư mục riêng cho mỗi lần chạy, tránh ghi đè."""
    if (
        not experiment_name
        or experiment_name in {".", ".."}
        or any(char in experiment_name for char in '<>:"/\\|?*')
        or experiment_name.endswith((" ", "."))
    ):
        raise ValueError("Tên thí nghiệm không hợp lệ.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{experiment_name}_{timestamp}_{uuid4().hex[:8]}"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    for folder in ("tensorboard", "checkpoints", "figures"):
        (run_dir / folder).mkdir()

    return run_dir


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """Lưu dữ liệu JSON với tiếng Việt UTF-8.

    Dữ liệu cần gồm các kiểu JSON chuẩn; chuyển tensor thành số
    bằng .item() và chuyển Path thành str trước khi gọi.
    """
    # Chuyển đổi trước khi mở file để tránh ghi dở nếu dữ liệu lỗi.
    content = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")

def validate_config(config: dict[str, Any]) -> None:
    """Kiểm tra các trường cơ bản trước khi chạy huấn luyện.

    Không sửa config và không tự điền hyperparameter.
    Các trường riêng của từng module được kiểm tra khi tích hợp.
    """
    required = (
        "experiment_name",
        "model_name",
        "pretrained",
        "batch_size",
        "learning_rate",
        "epochs",
        "strategy",
        "seed",
    )

    missing = [
        key for key in required
        if key not in config or config[key] is None
    ]
    if missing:
        raise ValueError(
            "Cấu hình thiếu hoặc còn null: " + ", ".join(missing)
        )

    name = config["experiment_name"]
    if (
        not isinstance(name, str)
        or not name.strip()
        or name in {".", ".."}
        or any(char in name for char in '<>:"/\\|?*')
        or name.endswith((" ", "."))
    ):
        raise ValueError("experiment_name không hợp lệ.")

    if config["model_name"] not in ("vgg16", "densenet121"):
        raise ValueError("model_name phải là vgg16 hoặc densenet121.")

    if not isinstance(config["pretrained"], bool):
        raise ValueError("pretrained phải là true hoặc false.")

    for key in ("batch_size", "epochs"):
        value = config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} phải là số nguyên dương.")

    lr = config["learning_rate"]
    if (
        type(lr) not in (int, float)
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("learning_rate phải là số hữu hạn lớn hơn 0.")

    seed = config["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed phải là số nguyên từ 0 đến 2**32 - 1.")

    if config["strategy"] not in (
        "classifier_only",
        "finetune_last_layers",
    ):
        raise ValueError("strategy không được hỗ trợ.")

    if config["strategy"] == "finetune_last_layers":
        layers = config.get("finetune_layers")
        if (
            not isinstance(layers, list)
            or not layers
            or any(
                not isinstance(layer, str) or not layer.strip()
                for layer in layers
            )
        ):
            raise ValueError(
                "Fine-tuning cần finetune_layers là danh sách tên lớp."
            )