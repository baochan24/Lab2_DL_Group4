# TV1 — Dữ liệu và tiền xử lý

## 1. Dataset

Nhóm sử dụng CIFAR-10 gồm 60.000 ảnh RGB kích thước gốc
32 × 32, thuộc 10 lớp: airplane, automobile, bird, cat,
deer, dog, frog, horse, ship và truck.

Dataset gốc có 50.000 ảnh train và 10.000 ảnh test.
Mỗi lớp có 5.000 ảnh train gốc và 1.000 ảnh test.

Ảnh minh họa được lưu tại reports/figures/cifar10_samples.png.

## 2. Chia dữ liệu

| Tập | Số ảnh | Vai trò |
|---|---:|---|
| Train | 45.000 | Cập nhật parameter |
| Validation | 5.000 | So sánh cấu hình và chọn checkpoint |
| Test | 10.000 | Đánh giá checkpoint sau khi lựa chọn |

Train/validation được chia từ tập train gốc bằng random_split,
val_ratio = 0.1 và torch.Generator().manual_seed(42).

Đây là chia ngẫu nhiên, không phân tầng theo lớp, nên không
khẳng định train/validation có số ảnh mỗi lớp bằng nhau.

Cùng dữ liệu, thứ tự mẫu và môi trường tương thích, seed cố định
hỗ trợ tái lập phép chia. Không bảo đảm mọi kết quả tính toán
giống tuyệt đối giữa các phiên bản và thiết bị.

## 3. Tiền xử lý thực tế

Các thí nghiệm chính dùng image_size = 224.

Transform train theo đúng thứ tự trong src/data.py:
1. Resize về 224 × 224.
2. RandomHorizontalFlip(p=0.5).
3. RandomCrop(224, padding=4).
4. ToTensor().
5. Normalize theo mean/std ImageNet.

Transform validation và test:
1. Resize về 224 × 224.
2. ToTensor().
3. Normalize theo cùng mean/std.

Mean RGB: (0.485, 0.456, 0.406).
Std RGB: (0.229, 0.224, 0.225).

ToTensor chuyển ảnh sang float32 trong khoảng [0, 1]
trước Normalize. Sau Normalize, giá trị không còn bị giới hạn
trong [0, 1].

Validation dùng bản sao dataset với transform đánh giá riêng,
không dùng augmentation ngẫu nhiên của train.

## 4. DataLoader

API tích hợp: build_dataloaders(config).

Hàm trả về các khóa:
train, val, test, class_names và num_classes.

- Train: shuffle=True.
- Validation/test: shuffle=False.
- Batch size: 32; riêng E4 dùng 64.
- num_workers=0 trong các lượt chạy CPU.
- pin_memory được đặt theo torch.cuda.is_available();
  trên môi trường CPU đã chạy, giá trị là False.

Ảnh đầu vào các lượt chính có shape [B, 3, 224, 224].
Nhãn có shape [B], kiểu torch.int64; ảnh kiểu torch.float32.

## 5. Kiểm tra đã thực hiện

Pipeline đã vượt qua smoke test tích hợp với image_size=32
và batch nhỏ. Sau đó các lượt thực nghiệm chính đã hoàn thành
với ảnh 224 × 224.

Smoke test chỉ kiểm tra khả năng kết nối và thực thi;
metrics của smoke test không đưa vào bảng kết quả chính.