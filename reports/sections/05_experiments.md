# 05. Thực nghiệm và hyperparameters

## 5.1. Mục đích

Nhóm thực hiện các so sánh:
- E1–E2: VGG16 và DenseNet121 với chiến lược classifier_only.
- E2–E3: learning rate 0.001 và 0.0001.
- E2–E4: batch size 32 và 64.
- E3–E5_cpu_1epoch: classifier_only và fine-tuning phần cuối.

## 5.2. Thiết lập chung

- Dataset: CIFAR-10.
- Chia dữ liệu: 45.000 ảnh train, 5.000 ảnh validation,
  10.000 ảnh test; seed 42.
- Ảnh đầu vào: 224 × 224; chuẩn hóa theo mean/std ImageNet.
- Optimizer: Adam; weight_decay = 0.0001.
- Loss: CrossEntropyLoss.
- Mỗi lượt thực nghiệm hoàn thành 1 epoch trên CPU.
- Chọn checkpoint theo validation accuracy cao nhất;
  nếu bằng nhau, ưu tiên validation loss thấp hơn.
- Tập test được dùng để đánh giá checkpoint đã chọn,
  không dùng để điều chỉnh hyperparameter hoặc chọn mô hình.

## 5.3. Cấu hình thực tế

| Run | Model | Strategy | Learning rate | Batch | Epoch |
|---|---|---|---:|---:|---:|
| E1_vgg16 | VGG16 | classifier_only | 0.001 | 32 | 1 |
| E2_densenet121 | DenseNet121 | classifier_only | 0.001 | 32 | 1 |
| E3_learning_rate | DenseNet121 | classifier_only | 0.0001 | 32 | 1 |
| E4_batch_size | DenseNet121 | classifier_only | 0.001 | 64 | 1 |
| E5_cpu_1epoch | DenseNet121 | finetune_last_layers | 0.0001 | 32 | 1 |

Với classifier_only, backbone bị đóng băng và toàn bộ classifier
được huấn luyện. VGG16 có 119.586.826 parameter được huấn luyện;
DenseNet121 có 10.250 parameter được huấn luyện.

E5 mở thêm features.denseblock4 và features.norm5, tổng cộng
2.170.378 parameter được huấn luyện.

E5_finetuning là cấu hình 10 epoch chưa có kết quả tương ứng;
không tính là một lượt thực nghiệm đã hoàn thành.

## 5.4. Kết quả thực đo

Nguồn: outputs/experiment_summary.csv.
Thời gian gồm training và validation theo phép đo của pipeline.

| Run | Val loss | Val accuracy | Test accuracy | Thời gian (giây) |
|---|---:|---:|---:|---:|
| E1_vgg16 | 0.524489 | 82.88% | Chưa đánh giá | 21929.01 |
| E2_densenet121 | 0.346216 | 87.84% | Chưa đánh giá | 5407.23 |
| E3_learning_rate | 0.577797 | 83.44% | Chưa đánh giá | 5329.62 |
| E4_batch_size | 0.367412 | 87.16% | Chưa đánh giá | 5499.97 |
| E5_cpu_1epoch | 0.232135 | 92.10% | 91.81% | 6135.23 |

## 5.5. Nhận xét trong phạm vi thực nghiệm

### Kiến trúc: E1 và E2

DenseNet121 đạt validation accuracy cao hơn VGG16
4.96 điểm phần trăm và có thời gian thực đo thấp hơn.

Đây là so sánh hai cấu hình classifier_only đã triển khai.
Hai kiến trúc có classifier và số parameter được huấn luyện
khác nhau; kết quả không tách riêng tác động của từng khác biệt.

### Learning rate: E2 và E3

Với classifier_only và một epoch, learning rate 0.001
đạt validation accuracy cao hơn 0.0001 là 4.40 điểm phần trăm.

Chưa đủ dữ liệu để kết luận learning rate nào tốt hơn
khi huấn luyện nhiều epoch.

### Batch size: E2 và E4

Batch 32 đạt validation accuracy cao hơn batch 64
0.68 điểm phần trăm.

Batch 64 mất thêm khoảng 92.74 giây trong hai lượt đo này,
nên không cải thiện thời gian thực đo.

Cùng một epoch, batch 32 có 1.407 bước cập nhật,
batch 64 có 704 bước cập nhật. Khác biệt này cần được
xét khi diễn giải kết quả.

### Chiến lược huấn luyện: E3 và E5_cpu_1epoch

Cùng learning rate 0.0001 và batch 32, fine-tuning phần cuối
đạt validation accuracy cao hơn classifier_only
8.66 điểm phần trăm.

E5_cpu_1epoch được chọn vì có validation accuracy cao nhất
trong năm lượt hoàn thành. Test accuracy sau lựa chọn là 91.81%.

## 5.6. Hạn chế

- Mỗi cấu hình chỉ chạy một epoch và một seed.
- Chưa đánh giá độ biến thiên giữa nhiều lần chạy.
- Chưa đủ dữ liệu để kết luận mô hình đã hội tụ.
- TensorBoard chỉ có một điểm mỗi chỉ số tại epoch 1;
  không suy luận đường cong học hay overshooting từ các điểm này.
- Thời gian đo trên máy cá nhân có thể chịu ảnh hưởng
  của tải hệ thống và điều kiện vận hành.
- Chưa thực hiện phân tích lỗi theo lớp hoặc confusion matrix.