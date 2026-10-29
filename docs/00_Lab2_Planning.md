# 00 — Hoạch định Lab 2

## Bài toán
Dùng mô hình neural pre-trained để phân loại ảnh CIFAR-10, điều chỉnh classifier cho 10 lớp, thử các kiến trúc và hyperparameter, theo dõi training bằng TensorBoard và đánh giá mô hình.

## Yêu cầu từ đề
- Thử các pre-trained architectures khác nhau (ví dụ VGG, DenseNet).
- Điều chỉnh hyperparameters như learning rate và batch size.
- Dùng TensorBoard theo dõi training.
- Khảo sát kiến trúc, layers và parameters.
- Thay classifier cuối phù hợp với dataset.
- Thử transfer learning (freeze các layer trước) và/hoặc fine-tuning (mở các layer được chọn).
- Dùng transforms và DataLoader.
- Chọn loss/optimizer, training loop và đánh giá test.

## Quyết định nhóm đã chốt
- Dataset: CIFAR-10.
- Môi trường: máy cá nhân có GPU NVIDIA.
- Cách tổ chức: Python scripts + Jupyter notebooks.
- Mô hình ứng viên ban đầu: VGG16 và DenseNet121; cần kiểm tra khả năng phần cứng trước khi chốt cấu hình.

## Tài liệu liên quan
- `ASSIGNMENT_LAB2.md`: phân công TV1–TV6.
- `01_Project_Structure.md`: cấu trúc, ownership và output.
- `TEAM_CONTRACT.md`: giao diện module.
- `GIT_WORKFLOW.md`: branch/PR.
- `REVIEW_CHECKLIST.md`: checklist trước khi bắt đầu code.

## Nguyên tắc
Đề không ghi rõ số epoch, số lần chạy hoặc accuracy tối thiểu. Không tự xem các giá trị đề xuất là yêu cầu bắt buộc. Không ghi số liệu trước khi thực nghiệm.
