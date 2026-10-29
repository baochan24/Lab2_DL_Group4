# TV2 — Kiến trúc mô hình

## 1. Mục tiêu

Xây dựng model factory dùng chung cho hai kiến trúc VGG16 và DenseNet121 từ Torchvision. Mô hình được điều chỉnh lớp phân loại cuối để phù hợp với CIFAR-10 gồm 10 lớp. Hàm `build_model(config)` trả về `torch.nn.Module`, giữ nguyên đầu ra logits và không áp dụng Softmax trước `CrossEntropyLoss`.

## 2. Pre-trained weights

- **VGG16:** `torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)` khi `pretrained: true`.
- **DenseNet121:** `torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)` khi `pretrained: true`.
- Khi `pretrained: false`, mô hình khởi tạo trọng số ngẫu nhiên; chế độ này phù hợp cho smoke test không cần tải weights.

Weights mặc định được huấn luyện trên ImageNet. Vì vậy, TV1 cần đảm bảo ảnh đầu vào có 3 kênh, kích thước và normalization phù hợp với preprocessing mà nhóm thống nhất.

## 3. Thay classifier

- **VGG16:** thay `model.classifier[-1]` bằng `nn.Linear(in_features, 10)`.
- **DenseNet121:** thay `model.classifier` bằng `nn.Linear(in_features, 10)`.

Đầu ra mong đợi cho batch kích thước `B` là `[B, 10]`. Các giá trị là logits thô, phù hợp với `torch.nn.CrossEntropyLoss`.

## 4. Chiến lược trainable parameters

### `classifier_only`

Đặt `requires_grad=False` cho toàn bộ tham số trước, sau đó bật gradient cho classifier mới. Đây là chiến lược transfer learning chỉ huấn luyện lớp phân loại cuối.

### `finetune_last_layers`

Classifier luôn được mở. Các layer backbone bổ sung chỉ được mở nếu được liệt kê tường minh trong `config['finetune_layers']`, theo prefix tên module/tham số trong `model.named_parameters()`. Nếu danh sách thiếu hoặc chứa prefix không hợp lệ, hàm báo lỗi thay vì âm thầm chạy sai cấu hình.

Ví dụ cấu hình VGG16:

```yaml
model_name: vgg16
pretrained: true
strategy: finetune_last_layers
finetune_layers:
  - features.28
```

Ví dụ cấu hình DenseNet121:

```yaml
model_name: densenet121
pretrained: true
strategy: finetune_last_layers
finetune_layers:
  - features.denseblock4
  - features.norm5
```

## 5. Thống kê tham số

`get_model_summary(model)` trả về:

- `total_parameters`: tổng số tham số của mô hình.
- `trainable_parameters`: số tham số có `requires_grad=True`.

Khi gọi `build_model`, hai số này cũng được in ra console và lưu trong `model.tv2_metadata`, cùng tên kiến trúc, trạng thái pretrained, chiến lược và các layer fine-tune. Không ghi số tham số cố định vào báo cáo vì cần lấy số thực tế từ đúng phiên bản Torchvision đang chạy.

## 6. Kiểm thử bàn giao

Chạy từ thư mục gốc repository:

```bash
python -m unittest tests.test_models
```

Các smoke test kiểm tra:

1. VGG16 và DenseNet121 trả đầu ra shape `[2, 10]`.
2. Với `classifier_only`, classifier được mở gradient và backbone bị freeze.
3. Fine-tuning yêu cầu danh sách layer tường minh và mở đúng layer được chọn.
4. Tên kiến trúc không hỗ trợ bị từ chối.
5. Tổng tham số và trainable parameters được tính ra.
6. BatchNorm bị freeze có thể được giữ ở chế độ eval.
7. Prefix fine-tuning không hợp lệ bị từ chối.

Test dùng `pretrained: false` để tránh tải weights trong lúc kiểm thử. Để xác nhận tải weights pretrained, chạy `build_model` với `pretrained: true` khi có kết nối mạng hoặc weights đã được cache.
