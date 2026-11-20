# Checklist trước khi bắt đầu code

## Phạm vi
- [ ] Đã đọc đề và `00_Lab2_Planning.md`.
- [ ] Đã đọc `ASSIGNMENT_LAB2.md` và xác nhận TV1–TV6.
- [ ] Đã đọc `01_Project_Structure.md`.
- [ ] Đã thống nhất giao diện module trong `TEAM_CONTRACT.md`.

## Môi trường
- [ ] Đã kiểm tra `nvidia-smi`.
- [ ] Đã kiểm tra Python và Git.
- [ ] Đã cài PyTorch/Torchvision tương thích và kiểm tra CUDA.
- [ ] Đã xác định giới hạn VRAM để chọn batch size.

## Code/data
- [ ] TV1 kiểm tra được batch CIFAR-10.
- [ ] TV2 model trả logits 10 lớp.
- [ ] TV3 có smoke test một batch trước khi chạy nhiều epoch.
- [ ] TV4 kiểm tra TensorBoard ghi được metrics.
- [ ] TV5 cấu hình run có thể tái lập.
- [ ] TV6 kiểm tra end-to-end khi module sẵn sàng.

## Output cuối
- [ ] Bảng kết quả chỉ có metrics thực đo.
- [ ] Có train/validation curves và nhận xét.
- [ ] Chọn cấu hình bằng validation.
- [ ] Đánh giá test cuối sau khi đã chọn model.
- [ ] Có checkpoint theo chính sách lưu trữ của nhóm.
- [ ] README đủ lệnh chạy lại.
- [ ] Báo cáo tổng hợp đầy đủ.
