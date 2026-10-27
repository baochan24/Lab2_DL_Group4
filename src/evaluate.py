"""TV4 — đánh giá mô hình đã được chọn.

TODO:
1. Nạp checkpoint đã chọn bằng validation.
2. Đánh giá trên test set sau khi quá trình chọn model đã kết thúc.
3. Tính test loss, accuracy và các metric bổ sung đã thống nhất.
4. Trả kết quả có thể ghi vào outputs/experiment_summary.csv.
5. Không dùng test metrics để chọn hyperparameter/model.
"""
from typing import Any


def evaluate_model(model, test_loader, config: dict[str, Any]) -> dict[str, float]:
    """Trả về metrics đo thực tế sau khi triển khai."""
    raise NotImplementedError("TV4: triển khai theo docs/TEAM_CONTRACT.md")
