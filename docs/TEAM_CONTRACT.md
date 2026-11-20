# Quy ước giao diện module Lab 2 — v1

Trạng thái: đề xuất tích hợp, chờ TV1–TV5 đối chiếu.
Giữ nguyên chữ ký hàm của bộ khung.
Nếu code đang triển khai khác quy ước, báo TV6 trước khi đổi.

## 1. Quy ước chung

- Config là dictionary Python, đọc từ YAML.
- Accuracy dùng thang 0–1; loss là giá trị trung bình trên mẫu.
- Epoch trong history và checkpoint bắt đầu từ 1.
- Model trả logits; không thêm Softmax trước CrossEntropyLoss.
- Chọn cấu hình và checkpoint bằng validation.
- Test chỉ dùng đánh giá cuối, không dùng để tune.
- Các đường dẫn trong config được tính từ thư mục gốc repository.
- Các giá trị chưa chốt không được tự điền thành kết quả thực nghiệm.

## 2. TV1 — Dữ liệu

Chữ ký:
`build_dataloaders(config) -> dict`

Các khóa trả về:
- train: DataLoader huấn luyện.
- val: DataLoader validation.
- test: DataLoader test.
- class_names: danh sách tên lớp theo thứ tự nhãn.
- num_classes: số lớp, bằng 10.

Mỗi batch là cặp (images, labels):
- images: tensor float32, shape [B, 3, H, W].
- labels: tensor int64, shape [B], giá trị 0–9.

Train và validation tách từ tập train CIFAR-10 bằng seed chung.
Validation/test không dùng augmentation ngẫu nhiên.
Split phải giữ nguyên giữa các thí nghiệm so sánh.
TV1 phối hợp TV2 thống nhất resize và normalization.

## 3. TV2 — Mô hình

Chữ ký:
`build_model(config) -> torch.nn.Module`

- Hỗ trợ vgg16 và densenet121 với trọng số pre-trained.
- Đầu ra logits shape [B, 10].
- classifier_only: đóng băng backbone, học phần classifier
  đã được TV2 mô tả rõ.
- finetune_last_layers: mở các lớp cuối được chỉ định rõ
  trong config; không tự suy đoán lớp cần mở.
- Ghi kiến trúc, tên weights, tổng số và số tham số trainable.
- TV2 và TV3 thống nhất chế độ BatchNorm/Dropout khi freeze;
  requires_grad=False không tự ngăn cập nhật BatchNorm buffers.

## 4. TV3 — Huấn luyện

Chữ ký:
`train_model(model, dataloaders, config) -> dict`

TV3 chịu trách nhiệm:
- Đưa model và batch lên config["device"].
- Tạo CrossEntropyLoss và optimizer từ config.
- Optimizer chỉ nhận các tham số được phép huấn luyện.
- Train và validation mỗi epoch.
- Ghi TensorBoard và lưu checkpoint.
- Đóng SummaryWriter khi kết thúc.

Dictionary trả về:
- history: list các dictionary theo epoch.
- best_checkpoint_path: chuỗi đường dẫn checkpoint.
- best_epoch: epoch tốt nhất.
- best_val_accuracy: validation accuracy tốt nhất.

Mỗi dòng history gồm:
epoch, train_loss, train_accuracy, val_loss, val_accuracy.

Chọn checkpoint:
- Validation accuracy cao hơn được ưu tiên.
- Nếu accuracy bằng nhau, chọn validation loss thấp hơn.

Checkpoint gồm ít nhất:
- model_state_dict
- epoch
- val_accuracy
- val_loss
- config

Config lưu trong checkpoint chỉ chứa kiểu dữ liệu cơ bản,
không chứa model, DataLoader, torch.device hoặc Path.

TensorBoard tags:
- Loss/train
- Loss/val
- Accuracy/train
- Accuracy/val

## 5. TV4 — Đánh giá

Chữ ký:
`evaluate_model(model, test_loader, config) -> dict`

- Nhận model đã được TV6 nạp checkpoint được chọn.
- Dùng model.eval() và tắt tính gradient.
- Đưa model và batch lên config["device"].
- Tính loss trung bình theo số mẫu.

Dictionary trả về:
- test_loss: float.
- test_accuracy: float, thang 0–1.
- num_samples: int.

TV4 phụ trách đọc biểu đồ, phân tích và minh họa kết quả.
TV3 trực tiếp ghi log trong training loop.

## 6. TV5 — Cấu hình thí nghiệm

Giữ các khóa hiện có:
experiment_name, model_name, pretrained, batch_size,
learning_rate, epochs, strategy, seed.

Bổ sung và thống nhất với chủ module:
- data_dir: nơi lưu CIFAR-10.
- val_ratio: tỷ lệ validation trong tập train.
- num_workers: số worker DataLoader.
- image_size: kích thước ảnh đầu vào.
- optimizer: tên optimizer được TV3 hỗ trợ.
- weight_decay: hệ số regularization.
- finetune_layers: danh sách lớp mở khi fine-tuning.

Trên Windows, bắt đầu kiểm tra với num_workers = 0.
Các giá trị chạy chính thức được chốt sau lượt chạy thử.
Không chạy huấn luyện với trường bắt buộc còn null.

E3/E4 phải ghi rõ baseline dùng để so sánh:
- E3 chỉ thay learning rate.
- E4 chỉ thay batch size.
- Các yếu tố còn lại giữ nguyên so với baseline.

## 7. TV6 — Điều phối

TV6 bổ sung vào config khi chạy:
- device: chuỗi cpu hoặc cuda.
- run_dir: thư mục riêng của run.
- tensorboard_dir: thư mục log TensorBoard.
- checkpoint_dir: thư mục checkpoint.

Các đường dẫn trên là chuỗi, được chuyển từ Path bằng str().

Luồng main.py:
1. Đọc và kiểm tra config.
2. Đặt seed, chọn thiết bị, tạo thư mục run.
3. Lưu cấu hình thực tế của run.
4. Gọi build_dataloaders.
5. Gọi build_model.
6. Gọi train_model.
7. Lưu history.
8. Khi thực hiện đánh giá cuối, nạp model_state_dict
   từ best_checkpoint_path vào model.
9. Gọi evaluate_model và lưu metrics.

Không tự động chạy test cho mọi thử nghiệm đang dùng để tune.
TV6 không sao chép training loop hoặc logic dữ liệu.

## 8. Điều kiện tích hợp

- Các thành viên đã đối chiếu chữ ký hàm và khóa dữ liệu.
- Config chạy có đủ giá trị và các kiểu dữ liệu hợp lệ.
- Một batch đi qua data → model → loss → backward thành công.
- Checkpoint lưu và nạp lại được.
- TensorBoard đọc được log.
- Output ghi riêng từng run, không ghi đè thí nghiệm khác.