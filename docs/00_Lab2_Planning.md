# Lab 2 — Planning & Problem Understanding

> **Trạng thái:** Giai đoạn 0 — hiểu đề và hoạch định.  
> **Phạm vi:** Chốt bài toán, yêu cầu, đầu ra và kế hoạch trước khi viết code huấn luyện.  
> **Nguồn chính:** Slide *Chapter 3 — Training Deep Neural Networks*, đặc biệt trang 20–24; đề *Practice 2 — Hands-on practice with pre-trained neural network architectures* ở trang 23.

## Tài liệu điều phối dự án

- [Phân công công việc chính thức cho TV1–TV6](ASSIGNMENT_LAB2.md)
- [Cấu trúc repository, quyền sở hữu file và quy ước kết nối module](01_Project_Structure.md)
- [Quy ước giao diện giữa các module](TEAM_CONTRACT.md)
- [Ma trận sở hữu file](FILE_OWNERS.md)
- [Git workflow và Pull Request](GIT_WORKFLOW.md)
- [Trạng thái review công việc](REVIEW_STATUS.md)

Các branch công việc của TV1–TV6 **chưa được tạo**. Chỉ tạo sau khi nhóm kiểm tra và đồng ý với tài liệu phân công cùng file structure.

---

## 1. Tóm tắt đề bài

Lab 2 yêu cầu thực hành với các **mạng neural đã được huấn luyện trước (pre-trained models)** bằng PyTorch/Torchvision. Nhóm cần tải và khảo sát kiến trúc mô hình, điều chỉnh mô hình cho bài toán phân loại cụ thể, chuẩn bị dữ liệu, huấn luyện, đánh giá, thử các mô hình/siêu tham số khác nhau và dùng TensorBoard để theo dõi quá trình học.

### Các yêu cầu được nêu trực tiếp trong đề

- Thử các pre-trained models khác nhau, ví dụ VGG, DenseNet, v.v.
- Điều chỉnh hyperparameters, ví dụ learning rate và batch size.
- Dùng TensorBoard để theo dõi quá trình training.
- In/xem kiến trúc mô hình và xác định các layer/parameters.
- Thay lớp phân loại cuối cho phù hợp với số lớp của dataset.
- Thử đóng băng các lớp trước (transfer learning) hoặc mở một số lớp cuối để fine-tune.
- Chuẩn bị dữ liệu bằng torchvision.transforms và torch.utils.data.DataLoader.
- Xác định loss function, optimizer, viết training loop.
- Đánh giá hiệu năng trên test dataset.

**Lưu ý:** Đề không quy định rõ số mô hình tối thiểu, số epoch, giá trị hyperparameter, ngưỡng accuracy hay định dạng báo cáo phải nộp. Các lựa chọn này cần được nhóm thống nhất hoặc kiểm tra lại với giảng viên nếu có hướng dẫn bổ sung.

---

## 2. Bài toán cần giải quyết

Ta cần sử dụng kiến thức đã học từ một mô hình pre-trained để giải quyết một nhiệm vụ phân loại ảnh mới.

Quy trình khái quát:

    Ảnh + nhãn → tiền xử lý → pre-trained backbone → classifier phù hợp → huấn luyện/validation → đánh giá test

- **Pre-trained model:** mô hình đã có trọng số học từ trước, thường trên ImageNet.
- **Transfer learning:** tận dụng trọng số đã học, thường đóng băng phần lớn backbone và huấn luyện classifier mới.
- **Fine-tuning:** mở một phần hoặc toàn bộ các lớp đã chọn để tiếp tục cập nhật trọng số trên dataset mục tiêu.
- **Classifier/head:** lớp hoặc cụm lớp đầu ra cần phù hợp với số lớp của bài toán.

Mục tiêu không chỉ là có một mô hình dự đoán được, mà còn phải **thực nghiệm có kiểm soát, ghi nhận kết quả và giải thích sự khác biệt giữa các cấu hình**.

---

## 3. Dataset — đã chốt

Slide trang 24 đưa **CIFAR-10** làm ví dụ và nhóm đã chọn CIFAR-10 làm dataset cho Lab 2.

| Hạng mục | Quyết định |
|---|---|
| Dataset | **CIFAR-10** |
| Số lớp | 10 |
| Nguồn dữ liệu | Có thể tải bằng `torchvision.datasets` |
| Tiền xử lý | Chuyển tensor và normalize phù hợp với trọng số pre-trained; resize nếu kiến trúc/cấu hình yêu cầu |
| Chia dữ liệu | Train để cập nhật trọng số; validation để chọn cấu hình; test chỉ dùng đánh giá cuối |

Không sử dụng kết quả test để chọn hyperparameters hoặc chọn mô hình tốt nhất. Tạo validation split từ phần training data và ghi rõ cách chia.

---

## 4. Câu hỏi nghiên cứu/thực nghiệm

Sau khi hoàn thành lab, nhóm cần có bằng chứng để trả lời:

1. Các kiến trúc pre-trained được thử có kết quả khác nhau như thế nào trên cùng bài toán?
2. Thay learning rate có ảnh hưởng thế nào đến loss và accuracy?
3. Thay batch size có ảnh hưởng thế nào đến quá trình học và thời gian chạy?
4. Đóng băng backbone so với fine-tuning một phần các lớp cuối có khác biệt gì?
5. Cấu hình nào được chọn dựa trên validation; kết quả cuối trên test là bao nhiêu?
6. TensorBoard cho thấy quá trình học ổn định, chưa học đủ hay có dấu hiệu overfitting nào?

Các câu hỏi 2–4 là các hướng thử nghiệm đề xuất để hiện thực hóa yêu cầu “adjust hyperparameters” và “adapt/fine-tune”. Không cần khẳng định trước kết quả; mọi kết luận phải dựa trên lần chạy thực tế.

---

## 5. Kế hoạch thực nghiệm đề xuất

Đây là **kế hoạch khởi đầu**, chưa phải kết quả và cũng không phải số lần chạy bắt buộc do đề quy định.

| ID | Thay đổi chính | Giữ cố định để so sánh | Mục đích |
|---|---|---|---|
| E1 | Mô hình pre-trained thứ nhất (ví dụ VGG16) | Dataset, split, preprocessing và quy trình đánh giá | Tạo baseline |
| E2 | Mô hình pre-trained thứ hai (ví dụ DenseNet121) | Cố gắng giữ cùng điều kiện với E1 | So sánh kiến trúc |
| E3 | Thay learning rate của mô hình đã chọn | Giữ model, batch size và cách freeze | Đánh giá ảnh hưởng learning rate |
| E4 | Thay batch size | Giữ model, learning rate và cách freeze | Đánh giá ảnh hưởng batch size |
| E5 | Fine-tune một số lớp cuối | Ghi rõ các lớp được mở và learning rate | So sánh với chỉ train classifier |

Chọn số epoch, learning rate và batch size sau khi kiểm tra tài nguyên máy. Trong mỗi so sánh, chỉ thay đổi một yếu tố chính nếu có thể, để kết luận dễ giải thích hơn. Nếu thay nhiều yếu tố cùng lúc, phải ghi rõ để tránh kết luận sai nguyên nhân.

---

## 6. Chỉ số và bằng chứng cần thu thập

### Chỉ số tối thiểu đề xuất

- Training loss theo epoch.
- Validation loss theo epoch.
- Training accuracy theo epoch.
- Validation accuracy theo epoch.
- Test loss và test accuracy của cấu hình cuối được chọn.
- Thời gian chạy và device (CPU/GPU), nếu có thể đo ổn định.

### TensorBoard

Mỗi experiment cần một tên/run riêng, ví dụ vgg16_frozen_lr1e-3, để không trộn log giữa các lần chạy. Ghi scalar cho loss/accuracy; có thể ghi thêm learning rate và graph kiến trúc nếu phù hợp.

TensorBoard là bằng chứng trực quan để so sánh quá trình học. Ảnh chụp các biểu đồ quan trọng có thể đưa vào báo cáo. Không nhất thiết đưa toàn bộ thư mục log lớn lên Git nếu đã có cách lưu trữ và tái tạo rõ ràng.

### Kiểm soát tính công bằng

- Dùng cùng dataset split giữa các thí nghiệm so sánh.
- Ghi rõ preprocessing và trọng số pre-trained sử dụng.
- Ghi lại model, seed (nếu thiết lập), learning rate, batch size, epoch và chiến lược freeze/fine-tune.
- Chọn cấu hình bằng validation; chỉ báo cáo test cuối sau khi đã chọn cấu hình.

---

## 7. Output cuối cùng cần có

Đề không liệt kê chính xác định dạng nộp, vì vậy danh sách dưới đây là **đề xuất để chứng minh đầy đủ quá trình làm lab**.

### A. Code có thể chạy lại

- Script/notebook chuẩn bị dataset và DataLoader.
- Script/module tải pre-trained model, in kiến trúc và thay classifier.
- Cấu hình freeze/fine-tune.
- Training loop có validation.
- Ghi log TensorBoard.
- Script đánh giá test và lưu kết quả.

### B. Bằng chứng thực nghiệm

- Bảng cấu hình cho từng experiment.
- Bảng metric có số liệu thật.
- Biểu đồ TensorBoard của các lần chạy chính.
- Kết quả test của cấu hình được chọn.
- Nhận xét về khác biệt giữa các mô hình và hyperparameters.

### C. Model artifact

- Checkpoint tốt nhất được chọn dựa trên validation.
- Hướng dẫn tải checkpoint và chạy inference để dự đoán một ảnh.

Checkpoint có thể lớn; trước khi commit lên Git, cần kiểm tra dung lượng và chính sách lưu file của repository. Không commit dataset đã tải về, cache, môi trường ảo hoặc log lớn một cách mặc định.

### D. Báo cáo và hướng dẫn sử dụng

- Báo cáo Markdown nêu bài toán, phương pháp, cấu hình, kết quả, phân tích và hạn chế.
- README hướng dẫn cài đặt, chạy train/evaluate và mở TensorBoard.
- Nêu rõ những gì đã chạy thật; không điền số liệu minh họa như kết quả thực nghiệm.

### Ví dụ bảng kết quả cuối (để trống cho đến khi có kết quả thật)

| Experiment | Model | Learning rate | Batch size | Strategy | Best val accuracy | Test accuracy | Runtime |
|---|---|---:|---:|---|---:|---:|---:|
| E1 | TBD | TBD | TBD | Classifier only | TBD | TBD | TBD |
| E2 | TBD | TBD | TBD | Classifier only | TBD | TBD | TBD |
| E3 | TBD | TBD | TBD | Classifier only | TBD | TBD | TBD |
| E4 | TBD | TBD | TBD | Classifier only | TBD | TBD | TBD |
| E5 | TBD | TBD | TBD | Fine-tuning | TBD | TBD | TBD |

---

## 8. Cấu trúc repository dự kiến

Đã chốt cách tổ chức **kết hợp Python scripts và notebooks**: scripts đảm nhiệm pipeline có thể chạy lại; notebooks dùng để khám phá dữ liệu, trực quan hóa và trình bày kết quả.

    Lab2_DL_Group4/
    ├── README.md
    ├── docs/
    │   ├── 00_Lab2_Planning.md
    │   └── Lab2_Report.md
    ├── configs/
    │   └── experiments.yaml
    ├── src/
    │   ├── data.py
    │   ├── models.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── utils.py
    ├── notebooks/
    │   ├── 01_dataset_exploration.ipynb
    │   └── 02_results_analysis.ipynb
    ├── results/
    │   ├── experiment_summary.csv
    │   └── figures/
    ├── requirements.txt
    └── .gitignore

Cách dùng dự kiến:
- `src/`: mã nguồn tái sử dụng để chuẩn bị dữ liệu, xây model, train và evaluate.
- `notebooks/01_dataset_exploration.ipynb`: kiểm tra mẫu ảnh, nhãn, shape và preprocessing; không đặt toàn bộ training pipeline ở đây.
- `notebooks/02_results_analysis.ipynb`: đọc bảng kết quả và tạo phân tích/biểu đồ cho báo cáo.
- `results/figures/`: chỉ giữ biểu đồ/ảnh cần thiết cho báo cáo.
- Dataset tải về, cache, môi trường ảo, TensorBoard logs lớn và checkpoint lớn mặc định không commit nếu chưa thống nhất cách quản lý.

---

## 9. Trình tự thực hiện từ đầu đến cuối

- [ ] **Phase 0 — Hiểu đề:** đọc tài liệu, ghi lại yêu cầu và đầu ra (tài liệu này).
- [ ] **Phase 1 — Chốt phạm vi:** xác nhận dataset, mô hình, môi trường chạy, số thí nghiệm và quy tắc nộp.
- [ ] **Phase 2 — Chuẩn bị repository:** README, requirements, .gitignore, nhánh làm việc và phân công.
- [ ] **Phase 3 — Kiểm tra dữ liệu:** tải dữ liệu, xem shape/label, kiểm tra preprocessing và DataLoader.
- [ ] **Phase 4 — Khảo sát model:** tải trọng số pre-trained, in kiến trúc, đếm tham số, thay classifier.
- [ ] **Phase 5 — Baseline:** chạy một cấu hình end-to-end và kiểm tra loss/accuracy/TensorBoard.
- [ ] **Phase 6 — Experiments:** thử kiến trúc và hyperparameters theo kế hoạch; lưu cấu hình và metric.
- [ ] **Phase 7 — Chọn và test:** chọn theo validation, chạy test cuối một cách độc lập.
- [ ] **Phase 8 — Báo cáo:** tạo bảng so sánh, biểu đồ, phân tích và nêu hạn chế.
- [ ] **Phase 9 — Kiểm tra và bàn giao:** chạy lại theo README, kiểm tra file, commit/push và review qua Pull Request nếu nhóm sử dụng PR.

---

## 10. Git/team workflow đề xuất

1. Không bắt đầu bằng cách đưa toàn bộ code lên main. Làm việc trên branch riêng cho từng phần.
2. Dùng commit message mô tả thay đổi, ví dụ: docs: add Lab 2 planning.
3. Trước khi mở PR, kiểm tra diff và bảo đảm không commit dataset, token, cache hoặc file checkpoint/log quá lớn.
4. Review code và kết quả trước khi merge.
5. Ghi rõ ai phụ trách data, model/training, evaluation, TensorBoard và report; tránh nhiều người sửa cùng một file nếu không cần thiết.

---

## 11. Quyết định đã chốt và việc còn mở

### Đã chốt
- [x] Dataset: **CIFAR-10**.
- [x] Môi trường: **máy cá nhân có GPU NVIDIA**.
- [x] Tổ chức mã nguồn: **kết hợp Python scripts và notebooks**.

### Còn cần xác nhận
- [ ] Tên model GPU, dung lượng VRAM, phiên bản driver/CUDA đang có.
- [ ] Phiên bản Python và cách quản lý môi trường (venv/conda).
- [ ] Hai kiến trúc đầu tiên: đề xuất VGG16 và DenseNet121.
- [ ] Số epoch, learning rate, batch size và giới hạn thời gian cho mỗi experiment.
- [ ] Yêu cầu nộp cụ thể từ giảng viên (nếu có).
- [ ] Thành viên và phân công công việc của nhóm.

**Nguyên tắc:** CIFAR-10, GPU NVIDIA và workflow scripts + notebooks là các quyết định đã chốt. Những chi tiết còn mở không được xem là yêu cầu chính thức cho đến khi nhóm xác nhận. Không ghi số liệu thực nghiệm trước khi thực sự chạy.
