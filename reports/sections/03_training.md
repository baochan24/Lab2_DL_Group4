# TV3 — Huấn luyện

## 1. Thiết lập

- Loss: nn.CrossEntropyLoss.
- Optimizer: Adam, chỉ nhận parameter có requires_grad=True.
- weight_decay: 0.0001.
- Learning rate khảo sát: 0.001 và 0.0001.
- Batch size khảo sát: 32 và 64.
- Mỗi lượt chạy chính: 1 epoch.
- Seed: 42.
- Thiết bị: CPU; torch.set_num_threads(1).
- Không triển khai early stopping hoặc learning rate scheduler
  trong các lượt thực nghiệm này.

## 2. Training

API tích hợp: train_model(model, dataloaders, config).

Với mỗi batch:
1. Chuyển ảnh và nhãn sang thiết bị.
2. Xóa gradient bằng zero_grad(set_to_none=True).
3. Forward để lấy logits [B, 10].
4. Tính CrossEntropyLoss.
5. Backward.
6. optimizer.step() cập nhật parameter được mở.

Model được đặt ở train mode; BatchNorm đã đóng băng
được giữ ở eval mode.

## 3. Validation

Sau training của epoch, model được đặt ở eval mode.
Gradient bị tắt và không cập nhật parameter.

Loss trung bình được tính theo số mẫu:
tổng batch_loss × số mẫu batch, chia tổng số mẫu.

Accuracy là tổng số dự đoán đúng chia tổng số mẫu.
Cách cộng này xử lý cả batch cuối có kích thước nhỏ hơn.

Train metrics được tích lũy trong lúc parameter đang thay đổi;
validation metrics được đo với model sau khi kết thúc epoch.
Ngoài ra train có augmentation, validation không có.
Do đó hai nhóm metrics không được đo trong điều kiện giống nhau.

## 4. Chọn và lưu checkpoint

Ưu tiên validation accuracy cao hơn.
Nếu accuracy bằng nhau, ưu tiên validation loss thấp hơn.

Checkpoint best_model.pth chứa:
- model_state_dict
- epoch
- val_accuracy
- val_loss
- config

Trong các lượt một epoch, checkpoint được chọn là epoch 1.
Checkpoint chỉ lưu sau khi hoàn tất training và validation;
chưa hỗ trợ tiếp tục từ batch đang dở hoặc khôi phục optimizer.

## 5. Output và TensorBoard

Mỗi lượt có:
- outputs/runs/<experiment_name>/config.json
- outputs/runs/<experiment_name>/history.csv
- outputs/runs/<experiment_name>/metrics.json
- outputs/runs/<experiment_name>/tensorboard/
- outputs/checkpoints/<experiment_name>/best_model.pth

Bốn tag TensorBoard:
- Loss/train
- Loss/val
- Accuracy/train
- Accuracy/val

Console in tiến độ batch và metrics cuối epoch.
Năm lượt E1, E2, E3, E4 và E5_cpu_1epoch đã hoàn thành.

## 6. Hạn chế

Chỉ chạy một epoch và một seed cho mỗi cấu hình.
Chưa đủ dữ liệu để kết luận hội tụ hoặc độ ổn định qua nhiều lần chạy.
Thời gian đo gồm training và validation trên CPU,
không phải phép đo tốc độ chuẩn hóa giữa các phần cứng.