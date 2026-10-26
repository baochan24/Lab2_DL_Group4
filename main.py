"""TV6 — điểm chạy tổng của Lab 2."""

import argparse
from pathlib import Path

import yaml

from src.utils import load_config, validate_config


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    """Đọc tham số dòng lệnh và kiểm tra cấu hình."""
    parser = argparse.ArgumentParser(
        description="Lab 2 — mô hình pre-trained trên CIFAR-10."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Đường dẫn YAML; đường dẫn tương đối tính từ gốc dự án.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Chỉ kiểm tra cấu hình cơ bản, không huấn luyện.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

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

    parser.exit(
        2,
        "Training pipeline is not integrated yet. "
        "Use --check-config to validate configuration.\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())