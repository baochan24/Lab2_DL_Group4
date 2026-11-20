# 01 — Cấu trúc dự án Lab 2

## 1. Cây thư mục

```text
Lab2_DL_Group4/
├── README.md
├── main.py
├── requirements.txt
├── .gitignore
├── configs/
│   ├── README.md
│   ├── E1_vgg16.yaml
│   ├── E2_densenet121.yaml
│   ├── E3_learning_rate.yaml
│   ├── E4_batch_size.yaml
│   └── E5_finetuning.yaml
├── data/
│   └── README.md
├── src/
│   ├── README.md
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── predict.py
├── notebooks/
│   ├── README.md
│   ├── 01_dataset_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── outputs/
│   ├── README.md
│   ├── experiment_summary.csv
│   └── figures/
├── reports/
│   ├── Lab2_Report.md
│   └── sections/
│       ├── 01_data.md
│       ├── 02_model.md
│       ├── 03_training.md
│       ├── 04_evaluation.md
│       ├── 05_experiments.md
│       └── 06_integration.md
├── tests/
│   └── README.md
└── docs/
    ├── 00_Lab2_Planning.md
    ├── ASSIGNMENT_LAB2.md
    ├── 01_Project_Structure.md
    ├── TEAM_CONTRACT.md
    ├── FILE_OWNERS.md
    ├── GIT_WORKFLOW.md
    └── REVIEW_CHECKLIST.md
```

## 2. File nào do ai sở hữu?

| File/thư mục | Người phụ trách | Mục đích |
|---|---|---|
| `src/data.py` | TV1 | Tải, chia và tiền xử lý CIFAR-10; DataLoader |
| `notebooks/01_dataset_exploration.ipynb` | TV1 | Kiểm tra ảnh, nhãn, shape và preprocessing |
| `src/models.py` | TV2 | Pre-trained architecture, classifier 10 lớp, freeze/fine-tune |
| `src/train.py` | TV3 | Training và validation loop |
| `src/evaluate.py` | TV4 | Đánh giá checkpoint được chọn |
| `notebooks/02_results_analysis.ipynb` | TV4 | Phân tích metrics/biểu đồ thật |
| `configs/` | TV5 | Cấu hình các lần thực nghiệm |
| `outputs/experiment_summary.csv` | TV5 phối hợp TV4 | Metrics và cấu hình thực đo |
| `outputs/figures/` | TV4 | Biểu đồ được chọn cho báo cáo |
| `src/utils.py`, `src/predict.py` | TV6 | Tiện ích chung và demo inference |
| `main.py` | TV6 | Kết nối các module thành pipeline |
| `README.md`, `requirements.txt`, `.gitignore` | TV6 | Cài đặt, chạy và quản lý file |
| `reports/sections/01_data.md` | TV1 | Báo cáo dữ liệu |
| `reports/sections/02_model.md` | TV2 | Báo cáo kiến trúc |
| `reports/sections/03_training.md` | TV3 | Báo cáo huấn luyện |
| `reports/sections/04_evaluation.md` | TV4 | Báo cáo đánh giá |
| `reports/sections/05_experiments.md` | TV5 | Báo cáo thí nghiệm |
| `reports/sections/06_integration.md` | TV6 | Báo cáo tích hợp |
| `reports/Lab2_Report.md` | TV6 tổng hợp | Báo cáo cuối |
| `docs/` | TV6 điều phối, cả nhóm review | Kế hoạch và quy ước |
| `tests/` | Mỗi chủ module; TV6 điều phối | Kiểm thử module |

## 3. Dòng dữ liệu từ đầu đến output

```text
CIFAR-10
   ↓
src/data.py
   ↓
Train / Validation / Test DataLoaders
   ↓
src/models.py — pre-trained VGG16/DenseNet121 + classifier 10 lớp
   ↓
src/train.py — train + validation + TensorBoard + checkpoint theo validation
   ↓
TV5 quản lý configs và so sánh các experiment
   ↓
Chọn cấu hình theo validation
   ↓
src/evaluate.py — đánh giá test cuối
   ↓
outputs/experiment_summary.csv + outputs/figures/
   ↓
notebooks/02_results_analysis.ipynb
   ↓
reports/Lab2_Report.md + README hướng dẫn tái lập
```

## 4. Quy tắc kết nối module
- TV1 cung cấp DataLoaders cho TV3/TV4.
- TV2 cung cấp model có đầu ra 10 logits cho TV3/TV4.
- TV3 nhận model, loaders và config; trả history/checkpoint metadata.
- TV4 ghi nhận/đọc metrics, đánh giá test sau khi model được chọn.
- TV5 quản lý cấu hình run và tổng hợp kết quả đo.
- TV6 điều phối bằng `main.py`, không nhân bản logic lõi.

## 5. Kết quả cuối cùng cần có
1. Code chạy lại được cho data/model/train/evaluate.
2. Bảng mô tả cấu hình từng thí nghiệm.
3. Train/validation loss và accuracy trên TensorBoard.
4. Bảng metrics đo thật và kết quả test cuối.
5. Checkpoint được chọn theo validation (lưu cục bộ hoặc kho artifact đã thống nhất).
6. Notebook phân tích kết quả.
7. Báo cáo kết luận so sánh các mô hình và hyperparameters.
8. README có lệnh cài đặt, train, evaluate và mở TensorBoard.

## 6. Điều không được làm
- Không dùng test set để chọn model/hyperparameter.
- Không ghi metrics trước khi chạy thật.
- Không commit dataset, secrets, virtualenv hoặc log/checkpoint lớn theo mặc định.
- Không xem file placeholder là code đã hoàn thiện.
