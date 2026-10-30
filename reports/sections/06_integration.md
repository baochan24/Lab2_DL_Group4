# TV6 — Tích hợp và tái lập

## 1. Môi trường thực nghiệm

Các lượt thực nghiệm được chạy trên Windows, sử dụng CPU.
CUDA không khả dụng trong môi trường đã kiểm tra.

Interpreter đã dùng:
D:\learn_DeepLearning\Lab2_DL_Group4\.venv\Scripts\python.exe

Các lượt chạy CPU sử dụng:
- torch.set_num_threads(1)
- num_workers=0
- seed=42

Phiên bản thư viện cần được đối chiếu bằng interpreter thực tế;
không dùng phiên bản của một môi trường Python khác.

## 2. Vai trò các module

| File | Vai trò |
|---|---|
| src/data.py | Dataset, transforms, chia train/validation, DataLoader |
| src/models.py | Nạp kiến trúc, thay classifier và cấu hình freeze/fine-tune |
| src/train.py | Training, validation, TensorBoard và checkpoint |
| src/evaluate.py | Đánh giá model đã nạp checkpoint |
| src/experiments.py | Tổng hợp kết quả, chọn cấu hình theo validation |
| src/utils.py | Seed, đọc/kiểm tra cấu hình, đường dẫn và JSON |
| main.py | Tích hợp pipeline qua dòng lệnh |

## 3. Kiểm tra môi trường

Từ thư mục gốc dự án trong PowerShell:

```powershell
$py = "D:\learn_DeepLearning\Lab2_DL_Group4\.venv\Scripts\python.exe"

& $py -c "import sys, torch, torchvision; print(sys.executable); print(sys.version); print(torch.__version__); print(torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

Đường dẫn interpreter trên thuộc máy thực nghiệm.
Người chạy trên máy khác cần thay bằng interpreter của mình.

Biến $py chỉ tồn tại trong phiên PowerShell hiện tại;
cần khai báo lại khi mở terminal mới.

## 4. Giao diện chạy

Kiểm tra cấu hình, không training:

```powershell
& $py main.py --config configs/E2_densenet121.yaml --check-config
```

Smoke test tích hợp với dữ liệu nhỏ:

```powershell
& $py main.py --config configs/E2_densenet121.yaml --smoke-test
```

Chạy training theo cấu hình, chỉ thực hiện khi cần tái lập:

```powershell
& $py main.py --config configs/E2_densenet121.yaml
```

main.py hiện chạy train/validation trong chế độ chính.
Đánh giá test chính thức đã được thực hiện riêng bằng cách
nạp checkpoint đã chọn và gọi evaluate_model.

main.py từ chối ghi đè khi phát hiện metrics.json hoặc
best_model.pth của lượt chính đã tồn tại.
Không xóa kết quả chỉ để vượt qua kiểm tra này.

## 5. Tổng hợp và phân tích

```powershell
& $py -m src.experiments
& $py -m tensorboard.main --logdir outputs/runs --port 6006
```

Mở TensorBoard tại http://localhost:6006.

Notebook notebooks/02_results_analysis.ipynb đọc kết quả đã lưu
để tạo bảng, biểu đồ so sánh và hiển thị final test metrics.
Notebook phân tích không chạy lại training.

## 6. Kết quả tích hợp đã xác nhận

- Smoke test qua main.py hoàn thành.
- Năm lượt thực nghiệm chính hoàn thành một epoch.
- Bảng tổng hợp được đọc từ metrics của các lượt chạy.
- E5_cpu_1epoch được chọn theo validation accuracy 92.10%.
- Checkpoint đã được nạp để đánh giá 10.000 ảnh test.
- Test accuracy: 91.81%.
- TensorBoard hiển thị đủ bốn tag cho năm lượt.
- Notebook phân tích đã chạy và hiển thị kết quả.

Kết quả test được chép từ output terminal đã làm tròn
6 chữ số thập phân vào metrics.json hiện có.
Nguồn này được ghi rõ trong trường final_test.source.

## 7. Artifact và giới hạn bàn giao

Cần giữ checkpoint, metrics, history, config và TensorBoard logs
cục bộ để đối chiếu kết quả.

Dataset, môi trường ảo và checkpoint lớn không đưa vào Git
theo mặc định. Trạng thái file được theo dõi và quy tắc ignore
phải được kiểm tra trước khi commit.

Chưa xác nhận trạng thái commit, push hoặc merge cuối cùng.
Các đường dẫn tuyệt đối lưu trong config/checkpoint có thể
cần điều chỉnh khi chuyển máy.

Chưa hỗ trợ resume giữa epoch hoặc khôi phục optimizer.
Mỗi lượt chỉ chạy một epoch và một seed.