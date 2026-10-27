# Tài liệu dữ liệu CIFAR-10

Tài liệu này mô tả cấu trúc dữ liệu, cách chia tập, quy trình tiền xử lý và cách sử dụng module `src/data.py` trong dự án **Lab 2: Transfer Learning & Fine-tuning trên CIFAR-10**.

## 1. Cấu trúc thư mục

Khi `torchvision.datasets.CIFAR10` tải và giải nén dữ liệu, thư mục `data/` thường có cấu trúc như sau:

```text
data/
├── README.md
├── cifar-10-batches-py/
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   └── test_batch
└── cifar-10-python.tar.gz
```

Các tệp trong `cifar-10-batches-py/` chứa dữ liệu nhị phân được mã hóa bằng pickle. Không mở chúng như tệp văn bản thông thường; `torchvision` sẽ tự tải, giải nén và đọc dữ liệu.

## 2. Thông tin bộ dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Tổng số ảnh | 60.000 |
| Ảnh huấn luyện ban đầu | 50.000 |
| Ảnh kiểm thử | 10.000 |
| Kích thước ảnh | 32 × 32 pixel |
| Số kênh | 3 (RGB) |
| Số lớp | 10 |
| Số ảnh mỗi lớp | 6.000 |

Các lớp của CIFAR-10:

| Nhãn | Tên lớp | Tiếng Việt |
|---:|---|---|
| 0 | `airplane` | Máy bay |
| 1 | `automobile` | Ô tô |
| 2 | `bird` | Chim |
| 3 | `cat` | Mèo |
| 4 | `deer` | Hươu |
| 5 | `dog` | Chó |
| 6 | `frog` | Ếch |
| 7 | `horse` | Ngựa |
| 8 | `ship` | Tàu thủy |
| 9 | `truck` | Xe tải |

## 3. Chia tập dữ liệu

Trong `src/data.py`, 50.000 ảnh huấn luyện ban đầu được chia thành tập train và validation bằng `torch.utils.data.random_split`. Tập test được giữ riêng để đánh giá cuối cùng.

| Tập dữ liệu | Số lượng mặc định | Mục đích |
|---|---:|---|
| Train | 45.000 ảnh | Huấn luyện mô hình và cập nhật trọng số |
| Validation | 5.000 ảnh | Theo dõi quá trình huấn luyện, điều chỉnh siêu tham số và hỗ trợ early stopping |
| Test | 10.000 ảnh | Đánh giá hiệu năng tổng quát hóa cuối cùng |

Việc chia train/validation sử dụng `seed=42` theo mặc định, giúp kết quả chia tập có thể tái lập giữa các lần chạy.

## 4. Tiền xử lý ảnh

Hàm `get_transforms()` trong `src/data.py` thiết lập các phép biến đổi sau:

### Train

1. `Resize(img_size)` (mặc định là `32 × 32`).
2. `RandomHorizontalFlip(p=0.5)` để lật ngang ngẫu nhiên.
3. `RandomCrop(32, padding=4)` để cắt ngẫu nhiên ảnh sau khi thêm viền.
4. `ToTensor()` để chuyển ảnh thành tensor có giá trị pixel trong khoảng `[0, 1]`.
5. `Normalize()` theo trung bình và độ lệch chuẩn của ImageNet.

### Validation và test

1. `Resize(img_size)`.
2. `ToTensor()`.
3. `Normalize()` với cùng thông số ImageNet; không cấu hình augmentation ngẫu nhiên.

Thông số chuẩn hóa theo thứ tự kênh RGB:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Công thức chuẩn hóa cho mỗi kênh:

```text
x_normalized = (x - mean) / std
```

Train và validation có transform riêng: train áp dụng augmentation, còn validation không áp dụng augmentation ngẫu nhiên. Hai tập dùng chung dữ liệu ảnh đã tải để tránh tạo thêm một bản sao lớn trong bộ nhớ.

## 5. Sử dụng DataLoader

Các tham số mặc định của `get_dataloaders()` là:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `data_dir` | `'./data'` | Thư mục lưu dữ liệu |
| `batch_size` | `64` | Số ảnh trong mỗi batch |
| `val_split` | `0.1` | Tỷ lệ validation lấy từ tập train ban đầu |
| `seed` | `42` | Seed dùng để chia train/validation |
| `num_workers` | `2` | Số tiến trình phụ tải dữ liệu |
| `img_size` | `(32, 32)` | Kích thước ảnh đầu vào |

Ví dụ:

```python
from src.data import get_dataloaders

train_loader, val_loader, test_loader, class_names = get_dataloaders(
    data_dir="./data",
    batch_size=64,
    val_split=0.1,
    seed=42,
    num_workers=2,
    img_size=(32, 32),
)

for images, labels in train_loader:
    # images: [batch_size, 3, 32, 32], kiểu torch.float32
    # labels: [batch_size], kiểu torch.int64
    pass
```

Hàm trả về ba `DataLoader` (`train_loader`, `val_loader`, `test_loader`) và danh sách tên lớp `class_names`.

## 6. Kiểm tra nhanh

Chạy module từ thư mục gốc của dự án:

```bash
python src/data.py
```

Lần chạy đầu tiên, `torchvision` có thể cần tải CIFAR-10. Khi chạy thành công, smoke test sẽ hiển thị:

- Kích thước batch ảnh, mặc định `[32, 3, 32, 32]` trong smoke test.
- Kiểu dữ liệu ảnh `torch.float32`.
- Kiểu dữ liệu nhãn `torch.int64` (`torch.long`).
- Khoảng giá trị nhãn, từ `0` đến `9`.
- Số lượng và tên các lớp.
- Thông báo smoke test hoàn tất.
