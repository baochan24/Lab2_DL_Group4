
# Training Pipeline — TV3

## 1. Mục tiêu

TV3 xây dựng vòng lặp huấn luyện và validation cho mô hình phân loại ảnh CIFAR-10. Module nhận mô hình từ TV2, DataLoader từ TV1 và cấu hình thực nghiệm từ TV5.

## 2. Quy trình huấn luyện

Mỗi epoch gồm hai giai đoạn: training và validation.

Trong training, mô hình chuyển sang chế độ train, nhận batch ảnh và nhãn, tính logits, tính CrossEntropyLoss, lan truyền ngược gradient và cập nhật tham số bằng optimizer.

Trong validation, mô hình chuyển sang chế độ eval và tắt gradient. Loss và accuracy được tính theo tổng số mẫu để xử lý đúng batch cuối có kích thước nhỏ hơn.

## 3. Cấu hình và optimizer

Số epoch, batch size, learning rate, optimizer và weight decay được đọc từ config. Optimizer chỉ nhận những tham số có `requires_grad=True`, phù hợp với chiến lược freeze hoặc fine-tune do TV2 thiết lập.

## 4. TensorBoard

TV3 ghi các metric theo epoch với các tag:

- `Loss/train`
- `Loss/val`
- `Accuracy/train`
- `Accuracy/val`

SummaryWriter được đóng khi quá trình huấn luyện kết thúc, kể cả khi xảy ra lỗi.

## 5. Lựa chọn checkpoint

Checkpoint tốt nhất được chọn dựa trên validation accuracy. Nếu hai epoch có accuracy bằng nhau, epoch có validation loss thấp hơn được ưu tiên.

Checkpoint chứa model state, epoch, validation accuracy, validation loss và config thực nghiệm. Tập test không được dùng để lựa chọn checkpoint.

## 6. Đầu ra

Hàm `train_model(model, dataloaders, config)` trả về:

- `history`: danh sách metric theo epoch.
- `best_checkpoint_path`: đường dẫn checkpoint tốt nhất.
- `best_epoch`: epoch được chọn.
- `best_val_accuracy`: validation accuracy tốt nhất.

Lịch sử huấn luyện được lưu thành CSV; checkpoint và TensorBoard log được lưu trong thư mục của từng run do TV6 cung cấp hoặc cấu hình.

## 7. Kiểm thử

Cần kiểm tra một batch đi qua đầy đủ các bước: data → model → loss → backward → optimizer update. Sau đó xác nhận validation chạy được, checkpoint lưu và nạp lại được, TensorBoard đọc được log.

**Trạng thái:** Chưa ghi số liệu thực nghiệm ở thời điểm viết báo cáo. Cập nhật kết quả sau khi tích hợp với TV1, TV2, TV4, TV5 và TV6 và thực hiện chạy thử thực tế.
