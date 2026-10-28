# 05. THỰC NGHIỆM VÀ ĐÁNH GIÁ MÔ HÌNH (EXPERIMENTS & HYPERPARAMETERS)

## 5.1. Câu hỏi thí nghiệm (Research Questions)
Thực nghiệm của nhóm được thiết kế để trả lời 3 câu hỏi nghiên cứu cốt lõi về kỹ thuật Transfer Learning & Fine-tuning trên bộ dữ liệu CIFAR-10:
* **RQ1 (Kiến trúc mô hình):** Khi đóng băng toàn bộ backbone (`classifier_only`), kiến trúc nào giữa **VGG16** (mạng chuỗi truyền thống) và **DenseNet121** (mạng kết nối dày đặc) đạt hiệu năng phân loại và tốc độ hội tụ tốt hơn?
* **RQ2 (Tối ưu siêu tham số):** Việc thay đổi **Learning Rate** (từ `0.001` xuống `0.0001`) và **Batch Size** (từ `32` lên `64`) ảnh hưởng như thế nào đến độ mịn của đường cong loss và mức độ tổng quát hóa của mô hình?
* **RQ3 (Chiến lược huấn luyện):** Mở đóng băng một số lớp cuối (`finetune_last_layers`) kết hợp với Learning Rate nhỏ cải thiện bao nhiêu % độ chính xác Validation/Test so với việc chỉ đóng băng hoàn toàn backbone?

---

## 5.2. Các yếu tố được giữ cố định (Controlled Variables)
Để đảm bảo nguyên tắc kiểm soát biến số đơn (Single Variable Control) và tính so sánh công bằng giữa các Run, các yếu tố sau được giữ cố định trên tất cả 5 thí nghiệm (E1–E5):
* **Dataset & Split:** Bộ dữ liệu CIFAR-10 được chia Train/Val/Test cố định với cùng hạt giống ngẫu nhiên (`seed: 42`, `val_ratio: 0.1`).
* **Preprocessing & Image Size:** Ảnh được resize về chuẩn $224 \times 224$ và chuẩn hóa (normalize) theo mean/std của ImageNet.
* **Optimizer & Loss Function:** Đều sử dụng **Adam Optimizer** (`weight_decay: 0.0001`) và hàm mất mát **CrossEntropyLoss**.
* **Số lượng Epochs:** Cố định **10 epochs** cho tất cả các run thử nghiệm.

---

## 5.3. Cấu hình từng run (Run Configurations)

| Run ID | Model Name | Strategy | Learning Rate | Batch Size | Epochs | Finetune Layers |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1_vgg16** | `vgg16` | `classifier_only` | 0.001 | 32 | 10 | Non-trainable |
| **E2_densenet121** | `densenet121` | `classifier_only` | 0.001 | 32 | 10 | Non-trainable |
| **E3_learning_rate** | `densenet121` | `classifier_only` | 0.0001 | 32 | 10 | Non-trainable |
| **E4_batch_size** | `densenet121` | `classifier_only` | 0.001 | 64 | 10 | Non-trainable |
| **E5_finetuning** | `densenet121` | `finetune_last_layers` | 0.0001 | 32 | 10 | `denseblock4`, `norm5` |

---

## 5.4. Metrics đo thật (Empirical Metrics)

Bảng dưới đây tổng hợp các chỉ số thực đo từ file `outputs/experiment_summary.csv` sau khi quá trình huấn luyện của nhóm hoàn tất. (Theo nguyên tắc của đề bài, các ô số liệu chỉ được điền khi đã có kết quả chạy thực tế).

| Run ID | Best Val Loss | Best Val Accuracy | Test Accuracy | Training Time (s) | Best Model Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **E1_vgg16** | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | Candidate |
| **E2_densenet121** | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | Candidate |
| **E3_learning_rate** | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | Candidate |
| **E4_batch_size** | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | Candidate |
| **E5_finetuning** | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | *Chờ thực đo* | Candidate |

---

## 5.5. Phân tích có bằng chứng (Evidence-based Analysis)

*(Phần này sẽ được phân tích chi tiết dựa trên số liệu thực tế xuất ra từ `outputs/experiment_summary.csv` và biểu đồ TensorBoard):*

1. **Bằng chứng So sánh Kiến trúc (E1 vs E2):**
   * Đánh giá hiệu quả tái sử dụng đặc trưng của các đường nối dày đặc (Dense Connections) trên DenseNet121 so với VGG16 khi chỉ train lớp phân loại.
2. **Bằng chứng Tác động Siêu tham số (E3 & E4):**
   * **Learning Rate (E2 vs E3):** So sánh mức độ dao động Loss và độ chính xác Validation để chứng minh xem `0.0001` có giúp mô hình tránh bớt hiện tượng overshooting xung quanh cực tiểu hay không.
   * **Batch Size (E2 vs E4):** Đánh giá trade-off giữa tốc độ train (giây/epoch) và độ chính xác khi tăng Batch Size từ 32 lên 64.
3. **Bằng chứng Chiến lược Fine-tuning (E2 vs E5):**
   * Chứng minh việc mở các lớp `denseblock4` cho phép mô hình thích ứng tốt hơn với phân bố không gian màu và đặc trưng của tập ảnh CIFAR-10.

---

## 5.6. Hạn chế / TODO (Limitations & Future Work)

### Hạn chế hiện tại (Limitations)
* **Số lượng Epochs:** Do giới hạn về thời gian huấn luyện trên máy cá nhân, các thử nghiệm chỉ mới chạy ở mức **10 epochs**.
* **Phạm vi mô hình:** Chưa mở rộng thử nghiệm thêm các họ mạng hiện đại khác như ResNet hoặc EfficientNet.

### TODO / Đề xuất nâng cấp
- [ ] Tự động hóa trích xuất biểu đồ so sánh các đường cong Loss/Accuracy từ log TensorBoard vào báo cáo chung.
- [ ] Chạy thực nghiệm bổ sung với chiến lược Learning Rate Scheduler (giảm LR theo mốc epoch) để đạt hiệu năng tối ưu nhất.