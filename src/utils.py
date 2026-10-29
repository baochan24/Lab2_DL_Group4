"""TV6 — tiện ích dùng chung cho pipeline.

TODO:
- Thiết lập random seed.
- Kiểm tra thiết bị CPU/GPU.
- Đọc cấu hình YAML.
- Tạo thư mục output/log.
- Lưu metadata thực nghiệm theo định dạng thống nhất.
"""
from typing import Any


def set_seed(seed: int) -> None:
    """TODO: thiết lập seed cho thư viện sử dụng."""
    raise NotImplementedError("TV6: triển khai sau khi chốt môi trường")


def get_device() -> str:
    """TODO: trả về thiết bị khả dụng; cần xác minh CUDA thực tế."""
    raise NotImplementedError("TV6: triển khai sau khi cài PyTorch")
