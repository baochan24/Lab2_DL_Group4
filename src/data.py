"""TV1 — tải và tiền xử lý CIFAR-10.

TODO:
1. Dùng torchvision.datasets.CIFAR10.
2. Tạo train/validation split có seed tái lập.
3. Dùng transforms phù hợp với weights pre-trained đã chọn.
4. Dùng DataLoader.
5. Kiểm tra shape, dtype, nhãn và batch.
6. Giữ test split riêng để đánh giá cuối.

Giao diện dự kiến cần được nhóm thống nhất trong docs/TEAM_CONTRACT.md.
"""
from typing import Any


def build_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """Trả về train/validation/test loaders và metadata lớp sau khi triển khai."""
    raise NotImplementedError("TV1: triển khai theo docs/TEAM_CONTRACT.md")
