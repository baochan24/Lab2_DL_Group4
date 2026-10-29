# Dữ liệu CIFAR-10

TV1 phụ trách cách tải dữ liệu qua `torchvision.datasets.CIFAR10`.

- Không commit dữ liệu tải về Git.
- Ghi rõ split train/validation, seed, transforms và normalization.
- CIFAR-10 có 10 lớp; tập test chuẩn phải giữ riêng cho đánh giá cuối.
- Đường dẫn dữ liệu mặc định dự kiến: `data/raw/` (đã được `.gitignore` bỏ qua).
