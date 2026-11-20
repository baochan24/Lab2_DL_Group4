# TV2 — Kiến trúc mô hình

## 1. Kiến trúc và weights

Nhóm dùng VGG16 và DenseNet121 từ torchvision.models.

Weights được nạp khi huấn luyện:
- VGG16_Weights.DEFAULT.
- DenseNet121_Weights.DEFAULT.

Các weights này được pretrained trên ImageNet.
API tích hợp: build_model(config).

VGG16 sử dụng các khối tích chập và pooling để trích xuất
đặc trưng, sau đó đưa qua classifier gồm các lớp fully connected.

DenseNet121 sử dụng các dense block, trong đó các lớp nhận
đặc trưng được nối từ những lớp trước; tiếp theo là pooling
và lớp phân loại tuyến tính.

## 2. Thay đầu ra thành 10 lớp

- VGG16: thay model.classifier[-1] bằng nn.Linear
  với 10 đầu ra.
- DenseNet121: thay model.classifier bằng nn.Linear
  với 10 đầu ra.

Model trả về raw logits shape [B, 10].
Không thêm Softmax trước CrossEntropyLoss.

## 3. Chiến lược huấn luyện

### classifier_only

Đóng băng toàn bộ parameter, sau đó mở toàn bộ
model.classifier cho huấn luyện.

Với VGG16, classifier gồm nhiều lớp fully connected.
Với DenseNet121, classifier là một lớp Linear.
Vì vậy số parameter được huấn luyện khác nhau đáng kể.

### finetune_last_layers

Ngoài classifier, E5_cpu_1epoch mở thêm:
- features.denseblock4
- features.norm5

Tên prefix được kiểm tra với tên parameter thực tế;
prefix không tồn tại sẽ gây lỗi thay vì bị bỏ qua.

## 4. Số parameter từ output thực tế

| Cấu hình | Tổng parameter | Parameter được huấn luyện |
|---|---:|---:|
| VGG16 classifier_only | 134.301.514 | 119.586.826 |
| DenseNet121 classifier_only | 6.964.106 | 10.250 |
| DenseNet121 fine-tuning phần cuối | 6.964.106 | 2.170.378 |

Số parameter được huấn luyện là một khác biệt cần nêu
khi so sánh VGG16 và DenseNet121.

## 5. BatchNorm và kiểm tra

Trong training, các BatchNorm có parameter bị đóng băng
được chuyển sang eval để không cập nhật running statistics.
Các BatchNorm được mở cho fine-tuning vẫn hoạt động ở train mode.

Đã kiểm tra forward/backward và optimizer step:
- DenseNet121 fine-tuning: batch 32, ảnh 224 × 224.
- DenseNet121 classifier_only: batch 64, ảnh 224 × 224.
- VGG16 classifier_only: batch 32, ảnh 224 × 224.

Năm lượt thực nghiệm chính đều hoàn thành một epoch.

## 6. Hạn chế

Thực nghiệm chỉ khảo sát hai kiến trúc với các cấu hình đã nêu.
Không kết luận một kiến trúc luôn tốt hơn trong mọi bài toán.