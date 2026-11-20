# Lab 2 — Group 4 | Notebook workflow

## Thứ tự

| Notebook                                  | Nội dung                                         | Bước đề bài |
| ----------------------------------------- | ------------------------------------------------ | ----------- |
| `01_environment_pretrained.ipynb`         | Môi trường, pretrained, kiến trúc                | 1–3         |
| `02_model_adaptation.ipynb`               | Classifier, freeze, fine-tuning                  | 4           |
| `03_data_exploration_preprocessing.ipynb` | CIFAR-10, EDA, transforms, DataLoader            | 5           |
| `04_training_experiments.ipynb`           | Train E1–E5, lưu log và checkpoint               | 6           |
| `05_model_monitoring.ipynb`               | TensorBoard, learning curves, gap, LR, thời gian | 6           |
| `06_evaluation_results.ipynb`             | So sánh, chọn checkpoint, test, kết luận         | 7           |

## Cài đặt và chạy

- `MODE = "review"`: Run All từng notebook, xem số liệu thực nghiệm đã lưu (không tải dataset hoặc train lại).
- `MODE = "full"`: huấn luyện mới; chỉnh `MODE = "full"` trong từng notebook muốn chạy thực tế; đặt `RUN_ID` khác cho từng lần train và dùng cùng `MODE` và `RUN_ID` ở notebook 04–06. Cần Internet để tải CIFAR-10 và pretrained weights. `EPOCHS_OVERRIDE = None` trong notebook 04 giữ cấu hình gốc; có thể đặt `5` để quan sát nhiều epoch, kết quả không còn là cùng cấu hình gốc.
- Notebook 04 ghi kết quả vào `outputs/notebook_runs/<RUN_ID>/training/`; history cập nhật sau từng epoch. Mở TensorBoard từ terminal riêng bằng lệnh được in ở notebook 04 hoặc 05 (truy cập http://localhost:6006). Notebook 05 đọc history, có thể chạy lại khi log cập nhật. Notebook 06 cần cả 5 run hoàn thành để chọn model theo validation, rồi test.

Kết quả gốc chỉ có 1 epoch mỗi thí nghiệm. Không có checkpoint, dataset hay event TensorBoard gốc trong ZIP; không suy diễn số liệu batch, GPU hay thời gian epoch lịch sử. E5 10 epoch chưa có kết quả. EDA được giữ, phần phân tích giảm chiều đã loại bỏ khỏi workflow đang sử dụng.
