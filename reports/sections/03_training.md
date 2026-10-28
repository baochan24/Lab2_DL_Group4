# Training Pipeline — TV3

## 1. Mục tiêu

TV3 xây dựng quy trình huấn luyện và validation cho bài toán phân loại ảnh CIFAR-10. Module nhận mô hình từ TV2, DataLoader từ TV1 và cấu hình thực nghiệm từ TV5.

## 2. Quy trình huấn luyện

Mỗi epoch gồm hai giai đoạn. Trong training, mô hình chuyển sang chế độ train, nhận batch ảnh và nhãn, tính logits, sử dụng CrossEntropyLoss, lan truyền ngược gradient và cập nhật tham số bằng optimizer.

Trong validation, mô hình chuyển sang chế độ eval và không tính gradient. Loss và accuracy được tính theo số lượng mẫu để xử lý đúng cả batch cuối có kích thước nhỏ hơn.

## 3. Optimizer và cấu hình

Số epoch, batch size, learning rate, optimizer, weight decay và thiết bị được lấy từ config. Optimizer chỉ cập nhật các tham số có requires_grad=True, phù hợp với chiến lược freeze hoặc fine-tune do TV2 thiết lập.

Module hỗ trợ Adam, AdamW và SGD. Mô hình cần trả về logits có kích thước [B, 10] cho 10 lớp CIFAR-10.

## 4. TensorBoard

TV3 ghi bốn metric theo epoch:

- Loss/train
- Loss/val
- Accuracy/train
- Accuracy/val

Log được lưu trong thư mục TensorBoard của từng lần chạy. SummaryWriter được đóng khi quá trình kết thúc.

## 5. Lựa chọn checkpoint

Checkpoint tốt nhất được chọn dựa trên validation accuracy. Nếu accuracy bằng nhau, validation loss thấp hơn được ưu tiên. Tập test không được sử dụng để lựa chọn checkpoint.

Checkpoint lưu model_state_dict, epoch, val_accuracy, val_loss và config thực nghiệm.

## 6. Đầu ra

Hàm train_model(model, dataloaders, config) trả về:

- history: lịch sử metric theo epoch.
- best_checkpoint_path: đường dẫn checkpoint tốt nhất.
- best_epoch: epoch được chọn.
- best_val_accuracy: validation accuracy tốt nhất.

## 7. Kiểm thử và kết quả

Cần tích hợp với DataLoader của TV1, mô hình của TV2 và pipeline điều phối của TV6 trước khi xác nhận toàn bộ quy trình hoạt động.

Chưa ghi số liệu accuracy hoặc loss chính thức vì chưa xác nhận kết quả chạy thực tế. Sau khi smoke test và thí nghiệm hoàn tất, cập nhật báo cáo bằng số liệu đo được.

## 8. Kết luận

TV3 phụ trách vòng lặp training, validation, ghi metric và lưu checkpoint theo kết quả validation. Việc xác nhận hoàn chỉnh cần dựa trên kết quả kiểm thử thực tế sau khi tích hợp các module của nhóm.
