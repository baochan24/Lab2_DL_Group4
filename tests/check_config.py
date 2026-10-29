"""Kiểm tra config mà không tải dữ liệu hoặc huấn luyện."""

from copy import deepcopy

from src.utils import load_config, validate_config


def main() -> None:
    # Giá trị chỉ phục vụ kiểm tra validator, không phải cấu hình
    # thực nghiệm chính thức của nhóm.
    valid = {
        "experiment_name": "validator_check",
        "model_name": "vgg16",
        "pretrained": True,
        "batch_size": 2,
        "learning_rate": 0.001,
        "epochs": 1,
        "strategy": "classifier_only",
        "seed": 42,
    }
    validate_config(valid)

    for key, value in (
        ("batch_size", None),
        ("batch_size", True),
        ("epochs", 0),
        ("learning_rate", float("nan")),
        ("model_name", "unknown"),
    ):
        invalid = deepcopy(valid)
        invalid[key] = value
        try:
            validate_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Validator accepted invalid {key}")

    # E1 hiện có thể chưa hoàn thiện: chỉ đọc và báo trạng thái.
    config = load_config("configs/E1_vgg16.yaml")
    try:
        validate_config(config)
    except ValueError as error:
        print("E1 not ready:", error)
    else:
        print("E1 basic validation: OK")

    print("Config validator checks: OK")


if __name__ == "__main__":
    main()