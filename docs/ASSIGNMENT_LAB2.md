# Phân công công việc Lab 2 — 6 thành viên

> Mỗi thành viên chịu trách nhiệm code/module, phần báo cáo và kiểm tra đầu ra được giao. Branch bên dưới là branch dự kiến; tạo theo quy trình nhóm.

## TV1 — Data & Preprocessing
**Branch:** `feature/data`

**File sở hữu**
- `src/data.py`
- `notebooks/01_dataset_exploration.ipynb`
- `data/README.md`
- `reports/sections/01_data.md`

**Việc cần làm**
1. Tải CIFAR-10 bằng Torchvision.
2. Chia train thành train/validation có seed cố định; giữ test riêng.
3. Thiết lập transforms và normalization phù hợp với pre-trained weights.
4. Tạo DataLoaders.
5. Kiểm tra shape, dtype, nhãn và số lớp.
6. Ghi lại lệnh chạy và output thực tế trong phần báo cáo.

**Bàn giao:** module data dùng được, notebook khám phá và mô tả split/transforms.

## TV2 — Model Architecture
**Branch:** `feature/model`

**File sở hữu**
- `src/models.py`
- `reports/sections/02_model.md`

**Việc cần làm**
1. Khảo sát VGG16 và DenseNet121 pre-trained.
2. In kiến trúc và xác định backbone/classifier.
3. Thay classifier cuối để đầu ra có 10 logits.
4. Hỗ trợ freeze backbone và fine-tune layer được chọn.
5. Ghi tổng số parameters và trainable parameters.

**Bàn giao:** model factory và bằng chứng kiểm tra đầu ra shape `[batch_size, 10]`.

## TV3 — Training Pipeline
**Branch:** `feature/training`

**File sở hữu**
- `src/train.py`
- `reports/sections/03_training.md`

**Việc cần làm**
1. Viết train loop: forward → loss → backward → optimizer step.
2. Tính train/validation loss và accuracy.
3. Nhận learning rate, batch size, epoch và optimizer từ config.
4. Phối hợp TV4 ghi TensorBoard.
5. Lưu checkpoint tốt nhất theo validation.

**Bàn giao:** training/validation pipeline, history và checkpoint metadata. Chưa tự tạo data/model riêng nếu module TV1/TV2 đã sẵn sàng.

## TV4 — Evaluation & TensorBoard
**Branch:** `feature/evaluation`

**File sở hữu**
- `src/evaluate.py`
- `notebooks/02_results_analysis.ipynb`
- `reports/sections/04_evaluation.md`

**Việc cần làm**
1. Đọc train/validation curves từ TensorBoard.
2. Đánh giá checkpoint đã chọn trên test sau khi chọn model xong.
3. Tổng hợp test loss, accuracy và metric bổ sung được nhóm thống nhất.
4. Vẽ biểu đồ từ metrics thật.
5. Phân tích overfitting và lỗi theo lớp nếu có thời gian.

**Bàn giao:** evaluation module, notebook phân tích, biểu đồ và nhận xét dựa trên kết quả. Không dùng test để chọn hyperparameter.

## TV5 — Experiments & Hyperparameters
**Branch:** `feature/experiments`

**File sở hữu**
- `configs/*.yaml`
- `outputs/experiment_summary.csv`
- `reports/sections/05_experiments.md`

**Việc cần làm**
1. Lập bảng thí nghiệm so sánh kiến trúc.
2. Lập thí nghiệm thay learning rate và batch size.
3. Lập thí nghiệm freeze classifier-only so với fine-tuning.
4. Ghi cấu hình mỗi run và bảo đảm mỗi kết quả truy về được config.
5. Chỉ điền metrics đã chạy thật.

**Bàn giao:** config tái lập, bảng kết quả thật và phần so sánh. Số lần chạy/epochs cụ thể cần căn theo tài nguyên và hướng dẫn giảng viên.

## TV6 — Integration & Project Lead
**Branch:** `feature/integration`

**File sở hữu**
- `main.py`
- `src/utils.py`, `src/predict.py`
- `README.md`, `requirements.txt`, `.gitignore`
- `docs/`
- `reports/Lab2_Report.md`
- `reports/sections/06_integration.md`

**Việc cần làm**
1. Kiểm tra GPU, Python, PyTorch/Torchvision.
2. Thống nhất API giữa các module trước khi ghép.
3. Tạo entry point điều phối; không viết lại logic lõi của module khác.
4. Review PR, xử lý conflict, chạy smoke test và end-to-end khi các phần đã sẵn sàng.
5. Tổng hợp báo cáo và hướng dẫn tái lập.

**Bàn giao:** pipeline tích hợp đã kiểm tra, hướng dẫn chạy và báo cáo tổng.

## Điều kiện bàn giao chung
Mỗi PR cần có: mục tiêu, file thay đổi, cách kiểm tra, output thực tế, phần báo cáo cập nhật, TODO/giới hạn và ghi chú nếu thay đổi API.
