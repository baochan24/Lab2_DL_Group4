# 1. Pipeline Dữ Liệu & Khám Phá Dữ Liệu (Data Pipeline & Exploration)

## 1.1 Khám Phá Đặc Tính Bộ Dữ Liệu CIFAR-10 (Exploratory Data Analysis - EDA)
Phân tích chi tiết đặc tính bộ dữ liệu CIFAR-10 tại notebook `notebooks/01_data_exploration.ipynb`:

* **Quy mô và kích thước:** Bộ dữ liệu gồm $60{,}000$ ảnh màu RGB kích thước $32 \times 32$ pixels, độ phân giải không gian $32 \times 32 \times 3$.
* **Phân bố tập nhãn:** Bộ dữ liệu đạt độ cân bằng tuyệt đối (Balanced Dataset) trên $10$ lớp nhãn đối tượng riêng biệt với $6{,}000$ mẫu/lớp, giúp loại bỏ hiện tượng lệch nhãn (class imbalance) khi tính toán hàm mất mát.
* **Danh mục phân loại:** Bao gồm $6$ lớp động vật tự nhiên (`bird`, `cat`, `deer`, `dog`, `frog`, `horse`) và $4$ lớp phương tiện giao thông (`airplane`, `automobile`, `ship`, `truck`).
* **Trực quan hóa mẫu:** Hình ảnh minh họa đại diện cho $10$ lớp dữ liệu đã được xuất và lưu trữ cố định tại `reports/figures/cifar10_samples.png`.

---

## 1.2 Chiến Lược Phân Chia Dữ Liệu & Đảm Bảo Tính Tái Lập (Reproducibility)
Toàn bộ $60{,}000$ mẫu dữ liệu được phân chia theo cấu trúc chuẩn cho bài toán Deep Learning:

| Tập dữ liệu | Số lượng mẫu | Mục đích sử dụng |
| :--- | :--- | :--- |
| **Train Set** | $45{,}000$ ảnh | Huấn luyện mô hình và lan truyền ngược cập nhật trọng số. |
| **Validation Set** | $5{,}000$ ảnh | Theo dõi hiện tượng Overfitting, căn chỉnh Hyperparameters và Early Stopping. |
| **Test Set** | $10{,}000$ ảnh | Đánh giá độc lập hiệu năng tổng quát hóa cuối cùng của mô hình. |

> **Cấu hình Tái lập (Reproducibility):** Việc phân chia tập Train và Validation từ $50{,}000$ ảnh train gốc được thực hiện bằng hàm `torch.utils.data.random_split` cùng với bộ sinh số ngẫu nhiên cố định:
>
> $$\text{Generator Seed} = 42$$
>
> Thiết lập này đảm bảo $100\%$ tính đồng nhất về mặt dữ liệu trên tất cả môi trường thử nghiệm của các thành viên trong nhóm.

---

## 1.3 Kỹ Thuật Tiền Xử Lý & Tăng Cường Dữ Liệu (Data Augmentation)
Nhằm chuẩn hóa đầu vào cho các mô hình Transfer Learning (Pre-trained trên tập ImageNet) và giảm thiểu hiện tượng Overfitting, pipeline xử lý tại `src/data.py` được thiết kế theo các công đoạn:

### 1. Kỹ thuật Tăng cường Dữ liệu tập Train (Train Augmentation)
* **`RandomHorizontalFlip(p=0.5)`:** Lật ngang ảnh ngẫu nhiên với xác suất $50\%$ giúp mô hình nhận diện đối tượng bất kể hướng nhìn.
* **`RandomCrop(32, padding=4)`:** Mở rộng viền thêm $4$ pixels xung quanh ảnh rồi cắt ngẫu nhiên lại khung $32 \times 32$, tạo độ bền vững đối với sự dịch chuyển vị trí vật thể.

### 2. Chuẩn hóa Z-Score (Normalization)
Chuyển đổi điểm ảnh từ dạng biểu diễn số nguyên $[0, 255]$ sang PyTorch Tensor dạng số thực float32 trong khoảng $[0.0, 1.0]$, sau đó thực hiện chuẩn hóa Z-Score theo tham số ImageNet chuẩn:

$$x_{\text{normalized}} = \frac{x - \mu}{\sigma}$$

Trong đó, các giá trị Kỳ vọng ($\boldsymbol{\mu}$) và Độ lệch chuẩn ($\boldsymbol{\sigma}$) cho 3 kênh màu RGB lần lượt là:
* $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$
* $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$

### 3. Tập Validation và Test
Chỉ thực hiện chuyển đổi Tensor và Chuẩn hóa Z-Score với tham số $(\boldsymbol{\mu}, \boldsymbol{\sigma})$ tương tự, tuyệt đối không áp dụng Data Augmentation để bảo toàn tính toàn vẹn khi đánh giá mô hình.

---

## 1.4 Thiết Kế Kiến Trúc Module Kỹ Thuật (`src/data.py`)
Toàn bộ quy trình nạp, tiền xử lý và phân chia dữ liệu được đóng gói thành hàm trung tâm `get_dataloaders()`. Hàm khởi tạo các đối tượng `DataLoader` với cấu hình tối ưu luồng tính toán:

* **`batch_size` (Mặc định: 64):** Kích thước lô dữ liệu linh hoạt, tương thích với dung lượng VRAM của GPU.
* **`pin_memory=True`:** Khóa các trang bộ nhớ RAM giúp tăng tốc độ truyền tải Tensor dữ liệu trực tiếp sang VRAM của GPU.
* **`num_workers=2`:** Khai thác tính năng đa luồng của CPU để nạp dữ liệu song song, triệt tiêu thời gian chờ của GPU giữa các bước huấn luyện.

---

## 1.5 Kết Quả Kiểm Thử Hệ Thống (Smoke Test Verification)
Module `src/data.py` đã hoàn thành bài kiểm thử độc lập (Smoke Test) thông qua giao diện dòng lệnh với kết quả kiểm định đạt tiêu chuẩn kỹ thuật:

* **Dạng Tensor đầu ra (Output Shape):** $[32, 3, 32, 32]$ theo đúng cấu trúc $[N, C, H, W]$ ($N$: Batch size, $C$: Kênh màu, $H$: Chiều cao, $W$: Chiều rộng).
* **Kiểu dữ liệu (Data Type):** Biểu diễn dưới dạng `torch.float32` đối với ảnh và `torch.int64` đối với nhãn.
* **Miền giá trị nhãn:** Nằm chính xác trong đoạn số nguyên $[0, 9]$.
* **Trạng thái kiểm thử:** Đạt các tiêu chí kiểm thử tính đúng đắn và độ tin cậy của dữ liệu.