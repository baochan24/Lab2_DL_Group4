# Báo cáo Lab 2 — Pre-trained Neural Network Architectures

## 1. Mục tiêu và phạm vi

Thực hành nạp mô hình pretrained từ torchvision.models,
thay classifier cho bài toán cụ thể, áp dụng transfer learning
và fine-tuning.

Nhóm sử dụng CIFAR-10 để:
- Thử nghiệm VGG16 và DenseNet121.
- Khảo sát learning rate và batch size.
- So sánh classifier_only với fine-tuning phần cuối.
- Theo dõi training bằng TensorBoard.
- Đánh giá checkpoint được chọn trên tập test.

Các thí nghiệm chạy một epoch, seed 42 trên CPU.
Đây là phạm vi thực nghiệm của nhóm, không phải số epoch
bắt buộc do đề quy định.

## 2. Dataset và tiền xử lý

CIFAR-10 gồm 60.000 ảnh RGB thuộc 10 lớp.

Nhóm sử dụng 45.000 ảnh train, 5.000 ảnh validation
và 10.000 ảnh test. Train/validation được chia ngẫu nhiên
từ tập train gốc với seed 42.

Ảnh được resize về 224 × 224 và normalize theo ImageNet.
Train có augmentation; validation và test không có
augmentation ngẫu nhiên.

Chi tiết: [Dữ liệu và tiền xử lý](sections/01_data.md).

## 3. Kiến trúc và transfer learning

Nhóm nạp VGG16 và DenseNet121 pretrained ImageNet,
thay lớp phân loại cuối để trả về logits cho 10 lớp CIFAR-10.

classifier_only mở toàn bộ classifier của mỗi kiến trúc.
Với DenseNet121 fine-tuning, nhóm mở thêm
features.denseblock4 và features.norm5.

| Cấu hình | Tổng parameter | Parameter được huấn luyện |
|---|---:|---:|
| VGG16 classifier_only | 134.301.514 | 119.586.826 |
| DenseNet121 classifier_only | 6.964.106 | 10.250 |
| DenseNet121 fine-tuning phần cuối | 6.964.106 | 2.170.378 |

Chi tiết: [Kiến trúc mô hình](sections/02_model.md).

## 4. Huấn luyện

Nhóm sử dụng CrossEntropyLoss, Adam và weight_decay=0.0001.

Mỗi lượt chạy một epoch. Learning rate khảo sát là
0.001 và 0.0001; batch size khảo sát là 32 và 64.

Checkpoint được chọn theo validation accuracy cao nhất;
nếu bằng nhau, ưu tiên validation loss thấp hơn.
Tập test không dùng để chọn cấu hình.

Chi tiết: [Huấn luyện](sections/03_training.md).

## 5. Kết quả thực nghiệm

Nguồn: outputs/experiment_summary.csv.
Thời gian gồm training và validation.

| Run | Model | Strategy | LR | Batch | Val loss | Val accuracy | Thời gian (s) |
|---|---|---|---:|---:|---:|---:|---:|
| E1_vgg16 | VGG16 | classifier_only | 0.001 | 32 | 0.524489 | 82.88% | 21929.01 |
| E2_densenet121 | DenseNet121 | classifier_only | 0.001 | 32 | 0.346216 | 87.84% | 5407.23 |
| E3_learning_rate | DenseNet121 | classifier_only | 0.0001 | 32 | 0.577797 | 83.44% | 5329.62 |
| E4_batch_size | DenseNet121 | classifier_only | 0.001 | 64 | 0.367412 | 87.16% | 5499.97 |
| E5_cpu_1epoch | DenseNet121 | finetune_last_layers | 0.0001 | 32 | 0.232135 | 92.10% | 6135.23 |

Tất cả các lượt trong bảng đều hoàn thành một epoch.

Trong thiết lập đã chạy:
- E2 cao hơn E1 4.96 điểm phần trăm validation accuracy.
- E2 cao hơn E3 4.40 điểm phần trăm.
- E4 thấp hơn E2 0.68 điểm phần trăm;
  batch 64 không giảm thời gian thực đo.
- E5 cao hơn E3 8.66 điểm phần trăm.

So sánh E1–E2 có khác biệt về classifier và số parameter
được huấn luyện. So sánh batch size ở cùng số epoch
cũng có số bước cập nhật optimizer khác nhau.

E5_finetuning là cấu hình 10 epoch chưa có kết quả;
không tính vào năm lượt hoàn thành.

Chi tiết: [Thực nghiệm và phân tích](sections/05_experiments.md).

## 6. Đánh giá test và TensorBoard

E5_cpu_1epoch được chọn theo validation accuracy 92.10%.
Checkpoint được đánh giá trên 10.000 ảnh test.

| Metric | Giá trị |
|---|---:|
| Test loss | 0.245938 |
| Accuracy | 0.918100 |
| Precision — weighted | 0.918185 |
| Recall — weighted | 0.918100 |
| F1-score — weighted | 0.917776 |

Test accuracy đạt 91.81%.
Metric test được lưu từ terminal đã làm tròn
6 chữ số thập phân; nguồn được ghi trong metrics.json.

TensorBoard đã hiển thị đủ năm lượt với các tag:
Loss/train, Loss/val, Accuracy/train và Accuracy/val.

Mỗi chỉ số có một điểm tại epoch 1; không sử dụng
các điểm này để kết luận xu hướng hội tụ.

Notebook notebooks/02_results_analysis.ipynb đã đọc
kết quả đã lưu và tạo biểu đồ so sánh.

Chi tiết: [Đánh giá và TensorBoard](sections/04_evaluation.md).

## 7. Tích hợp và tái lập

Pipeline được tích hợp qua main.py, cấu hình bằng YAML.
Các module dữ liệu, model, training và evaluation đã được
kiểm tra kết nối bằng smoke test.

Mỗi lượt lưu cấu hình, history, metrics, checkpoint
và TensorBoard logs. Notebook phân tích đọc các artifact
đã lưu, không yêu cầu training lại.

Chi tiết: [Tích hợp và tái lập](sections/06_integration.md).

## 8. Kết luận và hạn chế

Nhóm đã thực hành hai kiến trúc pretrained, điều chỉnh
learning rate và batch size, triển khai hai chiến lược
transfer learning/fine-tuning và theo dõi metrics bằng TensorBoard.

E5_cpu_1epoch có validation accuracy cao nhất trong năm
lượt đã hoàn thành và đạt test accuracy 91.81%.

Kết luận chỉ áp dụng cho các cấu hình đã chạy:
- Mỗi cấu hình có một epoch và một seed.
- Chưa đánh giá biến thiên qua nhiều lần chạy.
- Chưa xác nhận mô hình đã hội tụ.
- Chưa thực hiện confusion matrix hoặc phân tích lỗi theo lớp.
- Thời gian đo trên CPU cá nhân không phải benchmark
  chuẩn hóa cho mọi phần cứng.