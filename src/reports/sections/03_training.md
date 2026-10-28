
# Training Pipeline — TV3

## 1. Mục tiêu

TV3 xây dựng quy trình huấn luyện và validation cho bài toán phân loại ảnh CIFAR-10. Module nhận mô hình từ TV2, DataLoader từ TV1 và cấu hình thực nghiệm từ TV5. TV3 phụ trách vòng lặp training, validation, ghi metric, lưu checkpoint và TensorBoard.

## 2. Quy trình huấn luyện

Mỗi epoch gồm hai giai đoạn. Trong training, mô hình chuyển sang chế độ `train`, nhận từng batch ảnh và nhãn, tính logits, sử dụng `CrossEntropyLoss`, lan truyền ngược gradient và cập nhật tham số bằng optimizer.

Trong validation, mô hình chuyển sang chế độ `eval` và không tính gradient. Loss được tính trung bình theo số mẫu; accuracy được tính bằng tỷ lệ dự đoán đúng trên tổng số mẫu. Cách tính theo số mẫu xử lý chính xác cả batch cuối có kích thước nhỏ hơn.

## 3. Optimizer và cấu hình

Số epoch, batch size, learning rate, optimizer, weight decay và thiết bị được lấy từ `config`. Optimizer chỉ nhận tham số có `requires_grad=True`, phù hợp với chiến lược freeze hoặc fine-tune do TV2 thiết lập.

Module hỗ trợ Adam, AdamW và SGD. Mô hình cần trả về logits có kích thước `[B, 10]` cho 10 lớp CIFAR-10.

## 4. Theo dõi bằng TensorBoard

TV3 ghi bốn metric theo epoch:

- `Loss/train`
- `Loss/val`
- `Accuracy/train`
- `Accuracy/val`

Các log được lưu tại thư mục TensorBoard của từng run. `SummaryWriter` được đóng khi quy trình kết thúc.

## 5. Chọn checkpoint

Checkpoint được lựa chọn dựa trên validation, không dựa trên tập test. Validation accuracy cao hơn được ưu tiên; nếu accuracy bằng nhau, chọn validation loss thấp hơn.

Checkpoint lưu `model_state_dict`, `epoch`, `val_accuracy`, `val_loss` và `config`. Cấu hình lưu trong checkpoint chỉ chứa các kiểu dữ liệu cơ bản để thuận tiện lưu và đọc lại.

## 6. Đầu ra

Hàm `train_model(model, dataloaders, config)` trả về:

- `history`: danh sách metric theo epoch.
- `best_checkpoint_path`: đường dẫn checkpoint tốt nhất.
- `best_epoch`: epoch được chọn.
- `best_val_accuracy`: validation accuracy tốt nhất.

Mỗi dòng history gồm `epoch`, `train_loss`, `train_accuracy`, `val_loss` và `val_accuracy`.

## 7. Kiểm thử và kết quả

Tại thời điểm viết báo cáo, chưa xác nhận pipeline đã chạy end-to-end với toàn bộ module của nhóm. Cần tích hợp DataLoader từ TV1, mô hình từ TV2 và luồng điều phối từ TV6, sau đó kiểm tra một batch, xác nhận checkpoint có thể lưu và nạp lại, và kiểm tra log TensorBoard.

| Nội dung | Trạng thái |
|---|---|
| Kiểm tra cú pháp | Cần xác nhận trên môi trường dự án |
| Training và validation | Cần smoke test tích hợp |
| Lưu checkpoint | Cần kiểm thử |
| TensorBoard | Cần kiểm tra log thực tế |
| Accuracy và loss chính thức | Chưa có số liệu xác nhận |

Không sử dụng số liệu dự kiến làm kết quả thực nghiệm. Cập nhật bảng sau khi chạy kiểm thử thực tế.

## 8. Kết luận

TV3 cung cấp vòng lặp huấn luyện, validation, ghi metric và lưu checkpoint theo validation accuracy. Việc xác nhận hoàn chỉnh cần thực hiện sau khi tích hợp với các module dữ liệu, mô hình và điều phối của nhóm.
