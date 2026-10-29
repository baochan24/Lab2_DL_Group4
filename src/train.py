"""TV3 — training và validation.

TODO:
1. Nhận model, loaders, loss, optimizer và config.
2. Chạy train loop (forward, loss, backward, optimizer step).
3. Đánh giá validation sau mỗi epoch.
4. Ghi train/validation loss và accuracy.
5. Phối hợp TV4 để ghi TensorBoard.
6. Lưu checkpoint theo validation metric đã thống nhất.
7. Trả về history và đường dẫn/metadata checkpoint.

Chưa có vòng lặp huấn luyện trong bản scaffold này.
"""
from typing import Any


def train_model(model, dataloaders: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Trả về history và metadata checkpoint sau khi triển khai."""
    raise NotImplementedError("TV3: triển khai theo docs/TEAM_CONTRACT.md")
