"""TV6 — tiện ích dùng chung cho pipeline Lab 2."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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