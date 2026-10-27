# Lab 2 — Mạng neural pre-trained (CIFAR-10)

> **Bản cấu trúc dự án dành cho nhóm 6 thành viên.** Tài liệu và khung file đã được chuẩn bị; phần huấn luyện thực tế chưa được triển khai.
> 
> **Quyết định đã chốt:** CIFAR-10 · máy cá nhân có GPU NVIDIA · Python scripts kết hợp Jupyter Notebook.

## Mục tiêu theo đề
1. Thử nhiều mô hình pre-trained (ví dụ VGG, DenseNet).
2. Khảo sát kiến trúc, các layer và số lượng tham số.
3. Thay lớp phân loại cuối thành 10 lớp CIFAR-10.
4. Thử transfer learning (đóng băng backbone) và/hoặc fine-tuning (mở một số lớp).
5. Điều chỉnh hyperparameter như learning rate và batch size.
6. Chuẩn bị dữ liệu bằng `torchvision.transforms` và `DataLoader`.
7. Chọn loss function, optimizer, viết training loop và đánh giá test.
8. Dùng TensorBoard để theo dõi training và so sánh thí nghiệm.

> Đề không chỉ định chính xác số epoch, learning rate, batch size, số thí nghiệm tối thiểu hay ngưỡng accuracy. Các giá trị phải được chốt sau khi kiểm tra GPU và/hoặc hỏi giảng viên. Không ghi số liệu trước khi chạy thật.

## Trình tự làm việc — không đảo thứ tự
- **Giai đoạn 0 — Hiểu đề:** đọc `docs/00_Lab2_Planning.md`.
- **Giai đoạn 1 — Phân công:** đọc `docs/ASSIGNMENT_LAB2.md`, xác nhận file owner.
- **Giai đoạn 2 — Cấu trúc/API:** đọc `docs/01_Project_Structure.md` và `docs/TEAM_CONTRACT.md`.
- **Giai đoạn 3 — Môi trường:** kiểm tra GPU, Python, cài PyTorch/Torchvision tương thích.
- **Giai đoạn 4 — Dữ liệu:** TV1 làm `src/data.py`, kiểm tra ảnh/nhãn/batch.
- **Giai đoạn 5 — Mô hình:** TV2 làm `src/models.py`, kiểm tra pre-trained weights và classifier 10 lớp.
- **Giai đoạn 6 — Baseline:** TV3 kết nối train/validation; TV4 kiểm tra TensorBoard; chỉ chạy thử ngắn khi các module sẵn sàng.
- **Giai đoạn 7 — Thí nghiệm:** TV5 quản lý cấu hình; so sánh kiến trúc, learning rate, batch size và fine-tuning.
- **Giai đoạn 8 — Chọn model:** chọn theo validation; test chỉ dùng cho đánh giá cuối.
- **Giai đoạn 9 — Output:** tổng hợp metrics thật, biểu đồ TensorBoard, checkpoint theo chính sách nhóm, notebook phân tích và báo cáo.
- **Giai đoạn 10 — Bàn giao:** kiểm tra hướng dẫn chạy lại, review PR và hoàn tất báo cáo.

## Phân công nhanh
| Thành viên | Branch dự kiến | Sở hữu chính |
|---|---|---|
| TV1 | `feature/data` | `src/data.py`, notebook khám phá dữ liệu |
| TV2 | `feature/model` | `src/models.py` |
| TV3 | `feature/training` | `src/train.py` |
| TV4 | `feature/evaluation` | `src/evaluate.py`, notebook phân tích |
| TV5 | `feature/experiments` | `configs/`, bảng kết quả |
| TV6 | `feature/integration` | `main.py`, `src/utils.py`, tài liệu và tích hợp |

## Đầu ra cuối cần chuẩn bị
- Code chạy lại được cho data/model/train/evaluate.
- Bảng cấu hình và kết quả thực nghiệm đo thật.
- TensorBoard logs cục bộ và ảnh/biểu đồ cần đưa vào báo cáo.
- Kết quả test cuối của cấu hình đã chọn.
- Checkpoint tốt nhất theo validation (lưu cục bộ hoặc nơi lưu artifact đã thống nhất).
- Báo cáo và hướng dẫn tái lập thí nghiệm.

## Cài đặt — lưu ý GPU
1. Kiểm tra `nvidia-smi`, `python --version`, `git --version`.
2. Tạo môi trường ảo Python.
3. Cài PyTorch/Torchvision theo hướng dẫn chính thức phù hợp với hệ điều hành, driver và CUDA runtime của máy. Không đoán phiên bản CUDA từ riêng tên GPU.
4. Cài các thư viện còn lại trong `requirements.txt`.
5. Chạy kiểm tra `torch.cuda.is_available()` trước khi huấn luyện.

## Quy tắc Git
- Không commit dataset, `.venv`, secrets, cache, log TensorBoard lớn hay checkpoint lớn theo mặc định.
- Mỗi PR phải nêu mục tiêu, file thay đổi, cách kiểm tra, output thực tế, TODO và giới hạn.
- Chỉ ghi số liệu đã chạy thật; không ghi kết quả giả/ước lượng.
- Branch names trong tài liệu là kế hoạch; tạo branch theo quy trình nhóm đã thống nhất.
