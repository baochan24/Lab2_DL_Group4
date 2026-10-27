"""TV2 — nạp mô hình pre-trained và thay classifier cho CIFAR-10.

TODO:
1. Hỗ trợ các kiến trúc đã chốt (đề xuất ban đầu: VGG16 và DenseNet121).
2. Nạp weights pre-trained từ Torchvision.
3. Thay classifier cuối để trả về 10 logits.
4. Cho phép freeze backbone hoặc fine-tune các layer đã chọn.
5. Báo cáo tổng số tham số và số tham số trainable.

Không thêm Softmax trước CrossEntropyLoss.
"""
from typing import Any


def build_model(config: dict[str, Any]):
    """Trả về model có đầu ra [batch_size, 10] sau khi triển khai."""
    raise NotImplementedError("TV2: triển khai theo docs/TEAM_CONTRACT.md")
