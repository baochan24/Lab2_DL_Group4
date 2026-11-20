# BỘ QUY TẮC DỰ ÁN (PROJECT RULES & STANDARDS)
## Lab 2: Pre-trained Neural Network Architectures — Deep Learning (Group 4)

---

## 1. NGUYÊN TẮC CỐT LÕI (CORE PRINCIPLES)

1. **Tính trung thực thực nghiệm (Empirical Honesty):**
   - Tuyệt đối không suy đoán, giả lập hoặc ghi trước số liệu kết quả (loss, accuracy, thời gian huấn luyện) vào báo cáo hoặc tệp tổng hợp khi chưa chạy thực tế.
   - Mọi số liệu trong báo cáo (`reports/`) và bảng tổng hợp (`outputs/experiment_summary.csv`) phải truy vết được trực tiếp về log của vòng lặp huấn luyện (`metrics.json`, TensorBoard event logs).

2. **Tính tái lập (Reproducibility First):**
   - Cố định toàn diện seed ngẫu nhiên (`seed = 42`) trên toàn bộ các thư viện: `torch`, `torch.cuda`, `numpy`, `random`.
   - Mọi lượt huấn luyện phải được cấu hình độc lập qua tệp YAML (`configs/*.yaml`). Bất kỳ ai cũng có thể tái lập kết quả bằng một câu lệnh duy nhất từ thư mục gốc.

3. **Cơ chế chống ghi đè bảo vệ dữ liệu (Anti-Overwrite Protection):**
   - Pipeline huấn luyện (`main.py`) mặc định từ chối ghi đè khi phát hiện `best_model.pth` hoặc `metrics.json` của một thí nghiệm đã hoàn tất.
   - Không được tự ý xóa hoặc thay thế kết quả thực nghiệm đã được xác thực trừ khi có sự đồng thuận của nhóm và lý do kỹ thuật rõ ràng.

---

## 2. QUY TẮC DỮ LIỆU & TIỀN XỬ LÝ (DATA INTEGRITY & ZERO LEAKAGE)

1. **Phân chia tập dữ liệu (Dataset Partitioning):**
   - Bộ dữ liệu chuẩn: **CIFAR-10** (10 lớp, 60.000 ảnh kích thước gốc 32×32 RGB).
   - Tỷ lệ phân chia bắt buộc:
     - **Train set:** 45.000 ảnh (dùng cập nhật trọng số).
     - **Validation set:** 5.000 ảnh (dùng theo dõi quá trình học và chọn checkpoint tốt nhất).
     - **Test set:** 10.000 ảnh (giữ nguyên gốc từ CIFAR-10 test split).
   - Tách tập Train/Validation từ 50.000 ảnh train ban đầu bằng thuật toán ngẫu nhiên có `seed = 42` cố định; phân chia này phải bất biến qua tất cả các thí nghiệm so sánh.

2. **Quy tắc ngăn ngừa rò rỉ dữ liệu (Zero Data Leakage Protocol):**
   - **Tập Test độc lập tuyệt đối:** Tập Test chỉ được nạp và suy luận duy nhất một lần sau khi đã hoàn thành toàn bộ quá trình huấn luyện và đã xác định mô hình tối ưu dựa trên tập Validation.
   - Nghiêm cấm sử dụng tập Test để:
     - Tinh chỉnh siêu tham số (Learning rate, Batch size, Optimizer).
     - Quyết định số lượng epoch hoặc Early Stopping.
     - Lựa chọn checkpoint tốt nhất.
   - **Kiểm soát Data Augmentation:**
     - Data Augmentation ngẫu nhiên (`RandomCrop`, `RandomHorizontalFlip`) **chỉ được áp dụng trên tập Train**.
     - Tập Validation và Test chỉ sử dụng các phép biến đổi tất định (Deterministic transforms): Resize về 224×224, chuyển đổi Tensor, và chuẩn hóa (Normalize).

3. **Chuẩn hóa dữ liệu (Normalization Standards):**
   - Ảnh đầu vào được nội suy (Resize) về kích thước chuẩn **224 × 224** nhằm tương thích hoàn hảo với đặc trưng không gian của các trọng số ImageNet pre-trained.
   - Chuẩn hóa theo thống kê phân phối ImageNet:
     - Mean: `[0.485, 0.456, 0.406]`
     - Standard Deviation: `[0.229, 0.224, 0.225]`
   - Giao diện DataLoader chuẩn:
     - Image Tensor: `float32`, shape `[Batch_Size, 3, 224, 224]`.
     - Label Tensor: `int64`, shape `[Batch_Size]`, giá trị rời rạc trong đoạn `[0, 9]`.

---

## 3. QUY TẮC MÔ HÌNH HÓA (MODEL ARCHITECTURE & TRANSFER LEARNING)

1. **Nguồn trọng số tiền huấn luyện (Pre-trained Weights Source):**
   - Sử dụng API chính thức của `torchvision.models` với cú pháp weights hiện hành:
     - VGG16: `torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)`
     - DenseNet121: `torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)`
   - Trọng số ban đầu phải xuất phát từ tập ImageNet-1K.

2. **Thay thế Classifier Head:**
   - Đầu ra của mạng phải trả về raw **logits** với đúng 10 chiều tương ứng 10 lớp CIFAR-10 (shape `[B, 10]`).
   - Với VGG16: Thay thế `model.classifier[-1] = nn.Linear(4096, 10)`.
   - Với DenseNet121: Thay thế `model.classifier = nn.Linear(1024, 10)`.
   - **Cấm chèn Softmax ở tầng cuối:** Model chỉ xuất logits thô; không áp dụng `nn.Softmax` hoặc `F.softmax` trước hàm mất mát `nn.CrossEntropyLoss` (PyTorch CrossEntropyLoss đã tích hợp LogSoftmax và NLLLoss nội bộ để đảm bảo ổn định số học).

3. **Chiến lược đóng băng & Tinh chỉnh (Freezing & Fine-tuning Policies):**
   - **Chiến lược `classifier_only`:**
     - Đóng băng toàn bộ các lớp trích xuất đặc trưng (Backbone): Đặt `param.requires_grad = False`.
     - Chỉ mở các trọng số thuộc khối phân loại (`model.classifier`) với `param.requires_grad = True`.
   - **Chiến lược `finetune_last_layers`:**
     - Giữ đóng băng các tầng biểu diễn cấp thấp ở đầu mạng.
     - Mở có chủ đích các khối sâu nhất được chỉ định tường minh qua danh sách prefix (ví dụ với DenseNet121: `features.denseblock4` và `features.norm5`) cùng với toàn bộ `classifier`.
     - Tên prefix mở đóng băng phải được kiểm tra (validate) khớp 100% với tên layer thực tế trong cấu trúc mô hình; nếu sai lệch phải báo lỗi ngay lập tức, không được âm thầm bỏ qua.

4. **Xử lý Batch Normalization & Dropout khi đóng băng:**
   - Khi backbone bị đóng băng, các khối `nn.BatchNorm2d` thuộc backbone phải được cố định ở chế độ đánh giá (`eval()`) để không cập nhật `running_mean` và `running_var` theo phân phối batch nhỏ của CIFAR-10.
   - Các khối BatchNorm thuộc tầng được mở fine-tune tiếp tục chạy ở chế độ huấn luyện (`train()`).
   - `Dropout` hoạt động đúng theo pha: ngắt ngẫu nhiên trong lúc train, tắt trong lúc eval/test.

---

## 4. QUY TẮC HUẤN LUYỆN & TỐI ƯU HÓA (TRAINING & OPTIMIZATION)

1. **Lọc tham số cho Optimizer (Parameter Filtering):**
   - `torch.optim.Optimizer` (sử dụng Adam) **chỉ được phép nhận các tham số có `requires_grad = True`**:
     ```python
     trainable_params = [p for p in model.parameters() if p.requires_grad]
     optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
     ```
   - Tuyệt đối không truyền toàn bộ `model.parameters()` vào optimizer khi đang ở chế độ `classifier_only` để tránh lãng phí bộ nhớ lưu trữ trạng thái momentum và gradient không cần thiết.

2. **Tiêu chí lựa chọn Checkpoint tối ưu (Best Checkpoint Selection Rule):**
   - Mô hình tối ưu được lưu lại tại `outputs/checkpoints/<run_name>/best_model.pth`.
   - Tiêu chí ưu tiên:
     1. **Validation Accuracy cao nhất.**
     2. Nếu Validation Accuracy bằng nhau giữa các epoch: Chọn checkpoint có **Validation Loss thấp hơn**.
   - Cấu trúc Checkpoint bắt buộc lưu:
     - `epoch`: Chỉ số epoch đạt kết quả tốt nhất.
     - `model_state_dict`: Trọng số của mô hình.
     - `optimizer_state_dict`: Trạng thái optimizer.
     - `val_accuracy`: Độ chính xác trên tập validation.
     - `val_loss`: Hàm mất mát trên tập validation.
     - `config`: Dictionary cấu hình thí nghiệm (chỉ chứa kiểu dữ liệu nguyên thủy, không serialize object phức tạp).

3. **Chuẩn hóa Giám sát TensorBoard:**
   - Mọi lượt huấn luyện phải khởi tạo `SummaryWriter` với thư mục log: `outputs/runs/<experiment_name>`.
   - Bắt buộc ghi đúng 4 thẻ danh mục (Scalar Tags) chuẩn hóa sau mỗi epoch:
     - `Loss/train`
     - `Loss/val`
     - `Accuracy/train`
     - `Accuracy/val`
   - Đảm bảo gọi `writer.flush()` và `writer.close()` khi kết thúc vòng lặp để toàn vẹn tệp log.

---

## 5. QUY TẮC CẤU HÌNH & KHẢO SÁT THỰC NGHIỆM (EXPERIMENT SUITE)

1. **Bộ thí nghiệm chuẩn hóa (Benchmark Experiments):**
   Mọi thử nghiệm so sánh phải kiểm soát biến số (cô lập một nhân tố thay đổi duy nhất tại một thời điểm):
   - **So sánh kiến trúc:** `E1_vgg16` vs `E2_densenet121` (Cố định LR=0.001, Batch=32, Strategy=classifier_only).
   - **So sánh Learning Rate:** `E2_densenet121` (0.001) vs `E3_learning_rate` (0.0001) (Cố định DenseNet121, Batch=32, Strategy=classifier_only).
   - **So sánh Batch Size:** `E2_densenet121` (32) vs `E4_batch_size` (64) (Cố định DenseNet121, LR=0.001, Strategy=classifier_only).
   - **So sánh Chiến lược huấn luyện:** `E3_learning_rate` (classifier_only) vs `E5_cpu_1epoch` (finetune_last_layers) (Cố định DenseNet121, LR=0.0001, Batch=32).

2. **Quản lý cấu hình tệp YAML:**
   - Mọi tệp trong `configs/` phải tuân thủ schema được kiểm tra qua `src.utils.validate_config`:
     - Bắt buộc có: `experiment_name`, `model_name`, `pretrained`, `batch_size`, `learning_rate`, `epochs`, `strategy`, `seed`.
     - Các trường phụ thuộc: `finetune_prefixes` bắt buộc có nếu `strategy == "finetune_last_layers"`.

---

## 6. PHÂN CÔNG TRÁCH NHIỆM & QUY ƯỚC HỢP TÁC (TEAM CONTRACT)

| Thành viên | Phụ trách chính | Module sở hữu | Tiêu chuẩn bàn giao |
|---|---|---|---|
| **TV1** | Dữ liệu & Tiền xử lý | `src/data.py`<br>`notebooks/01_dataset_exploration.ipynb`<br>`reports/sections/01_data.md` | DataLoaders sạch, split 45k/5k/10k bất biến, ảnh chuẩn hóa ImageNet, không leak test. |
| **TV2** | Kiến trúc mô hình | `src/models.py`<br>`reports/sections/02_model.md` | Model factory hỗ trợ VGG16/DenseNet121, thay head 10 logits, đếm đúng tham số train/total, quản lý eval cho BatchNorm. |
| **TV3** | Vòng lặp huấn luyện | `src/train.py`<br>`reports/sections/03_training.md` | Training loop hoàn chỉnh, đồng bộ device, tính loss/acc, ghi 4 tags TensorBoard, lưu checkpoint chuẩn. |
| **TV4** | Đánh giá & Phân tích | `src/evaluate.py`<br>`notebooks/02_results_analysis.ipynb`<br>`reports/sections/04_evaluation.md` | Script đánh giá test độc lập bằng no_grad, đối chiếu TensorBoard curves, báo cáo accuracy/loss/weighted metrics. |
| **TV5** | Quản lý thực nghiệm | `configs/*.yaml`<br>`outputs/experiment_summary.csv`<br>`reports/sections/05_experiments.md` | Bộ config chuẩn hóa, bảng tổng hợp CSV đồng bộ 100% với logs thực tế, so sánh biện luận khoa học. |
| **TV6** | Tích hợp & Đảm bảo CL | `main.py`<br>`src/utils.py`<br>`src/predict.py`<br>`reports/Lab2_Report.md`<br>`reports/sections/06_integration.md` | Pipeline CLI tích hợp thông suốt, quản lý seed/device/check-config/smoke-test, tổng hợp báo cáo chung. |

---

## 7. QUY TRÌNH KIỂM SOÁT PHIÊN BẢN (GIT WORKFLOW)

1. **Quy tắc nhánh (Branching Rule):**
   - Nhánh `main`: Chỉ chứa code hoàn chỉnh, ổn định, đã qua kiểm thử tích hợp và được review.
   - Các nhánh tính năng theo vai trò: `feature/data`, `feature/model`, `feature/training`, `feature/evaluation`, `feature/experiments`, `feature/integration`.
2. **Quy tắc Commit & Pull Request:**
   - Commit message mang tính mô tả rõ ràng (theo chuẩn Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`).
   - Mọi Pull Request sáp nhập vào `main` bắt buộc phải vượt qua:
     - Kiểm tra cấu hình: `python tests/check_config.py`.
     - Smoke test tích hợp không lỗi runtime.
     - Review chéo từ TV6 (Integration Lead) hoặc thành viên chịu trách nhiệm module phụ thuộc.
