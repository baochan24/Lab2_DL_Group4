## Môi trường tích hợp của TV6

- Hệ điều hành: Windows.
- Python: 3.13.7.
- Môi trường riêng: `.venv`.
- PyTorch: 2.14.1+cpu.
- Torchvision: 0.29.1+cpu.
- Thiết bị chạy hiện tại: CPU.
- Windows nhận card đồ họa Intel UHD Graphics.
- CUDA không khả dụng trong môi trường đã kiểm tra.
- PyYAML và TensorBoard import thành công.
- `pip check`: không phát hiện xung đột thư viện.

Thông tin trên chỉ mô tả máy TV6. Thiết bị chạy thực nghiệm
của các thành viên khác cần được ghi nhận riêng.

## Tiện ích dùng chung

Đã triển khai trong `src/utils.py`:

- `set_seed(seed)`: đặt seed cho Python, NumPy và PyTorch.
- `get_device()`: chọn CUDA nếu khả dụng, ngược lại dùng CPU.
- `load_config(path)`: đọc cấu hình YAML dạng dictionary.
- `create_run_dir(...)`: tạo thư mục riêng cho mỗi lần chạy.
- `save_json(...)`: lưu dữ liệu JSON bằng UTF-8.

`load_config` hiện chỉ kiểm tra cấu trúc YAML cơ bản,
chưa kiểm tra đầy đủ schema hoặc các giá trị `null`.

## Kết quả kiểm tra đã xác nhận

- Kiểm tra cú pháp `src/utils.py`: thành công.
- Import các tiện ích: thành công.
- Thiết bị được chọn: `cpu`.
- Đọc cấu hình E1: trả về `experiment_name = E1_vgg16`.
- Code tiện ích đã được push lên `feature/integration`,
  commit `fd14c20`.

## Công việc tích hợp còn lại

- [ x] Ghi nhận kết quả kiểm tra seed, tạo thư mục và lưu JSON.
- [ ] Đối chiếu API thực tế của TV1–TV5.
- [ ] Thống nhất khóa cấu hình và dữ liệu trả về.
- [ ] Triển khai điểm chạy tổng `main.py`.
- [ ] Kiểm tra pipeline bằng một batch nhỏ.
- [ ] Kiểm tra lưu và nạp checkpoint.
- [ ] Kiểm tra TensorBoard đọc được log.
- [ ] Chạy pipeline hoàn chỉnh và ghi kết quả thực tế.

Chưa có kết quả huấn luyện hoặc đánh giá end-to-end
được xác nhận trong phần tích hợp.