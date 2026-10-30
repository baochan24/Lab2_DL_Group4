# TV4 — Đánh giá và TensorBoard

## Quy trình lựa chọn và đánh giá

Nhóm so sánh các cấu hình bằng validation.
E5_cpu_1epoch được chọn với validation accuracy 92.10%
và validation loss 0.232135.

Checkpoint:
outputs/checkpoints/E5_cpu_1epoch/best_model.pth

Sau khi lựa chọn, checkpoint được nạp để đánh giá trên
10.000 ảnh test CIFAR-10. Evaluation dùng model.eval()
và torch.no_grad(), không cập nhật trọng số.

## Kết quả test

| Metric | Giá trị |
|---|---:|
| Test loss | 0.245938 |
| Accuracy | 0.918100 |
| Precision — weighted | 0.918185 |
| Recall — weighted | 0.918100 |
| F1-score — weighted | 0.917776 |

Test accuracy đạt 91.81%, thấp hơn validation accuracy
0.29 điểm phần trăm.

Các metric test được ghi lại từ output terminal đã làm tròn
đến 6 chữ số thập phân, sau đó lưu trong final_test của:
outputs/runs/E5_cpu_1epoch/metrics.json

Trường test_acc được dùng để cập nhật bảng tổng hợp.
Các lượt còn lại không có kết quả test chính thức.

## TensorBoard

TensorBoard đã hiển thị đủ năm lượt thực nghiệm với các tag:
- Loss/train
- Loss/val
- Accuracy/train
- Accuracy/val

Khi xem minh chứng, đặt smoothing = 0 và bỏ tùy chọn
Ignore outliers in chart scaling để thấy đầy đủ các điểm.

Mỗi lượt chỉ có một điểm tại step 1, tương ứng một epoch.
Biểu đồ dùng đối chiếu metrics giữa các lượt;
không đủ để phân tích xu hướng hội tụ qua nhiều epoch.

## Hạn chế

Chưa thực hiện confusion matrix hoặc phân tích lỗi theo lớp.
Precision, recall và F1 được báo cáo theo weighted average.