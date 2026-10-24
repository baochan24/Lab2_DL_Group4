# Lab 2 — Phân công công việc chính thức (6 thành viên)

> Repository: baochan24/Lab2_DL_Group4  
> Dataset đã chốt: CIFAR-10  
> Môi trường đã chốt: máy cá nhân có GPU NVIDIA  
> Cách tổ chức: Python scripts + Jupyter notebooks  
> Trạng thái: phân công đã được nhóm xác nhận; chưa tạo sáu branch công việc.

## Nguyên tắc làm việc chung
- Mỗi thành viên chịu trách nhiệm cho module và phần báo cáo được giao.
- Làm việc trên branch riêng, tạo Pull Request (PR) để review trước khi merge.
- Không commit dataset tải về, môi trường ảo, secrets, cache, log lớn hoặc checkpoint lớn khi chưa có thống nhất.
- Mọi số liệu trong báo cáo phải đến từ lần chạy thực tế; không điền kết quả dự kiến thành kết quả thật.
- Chọn mô hình/siêu tham số dựa trên validation; chỉ dùng test để đánh giá cuối sau khi đã chọn cấu hình.
- Khi cần thay đổi giao diện module dùng chung, báo cho thành viên phụ thuộc trước khi sửa.

## TV1 — Data & Preprocessing
**Branch dự kiến:** feature/data

**Trách nhiệm**
- Khảo sát CIFAR-10: cấu trúc dữ liệu, 10 lớp, số lượng mẫu và shape ảnh.
- Xây dựng quy trình tải dữ liệu và chia train/validation/test; ghi rõ seed và cách chia.
- Chọn transforms và normalization phù hợp với trọng số pre-trained được sử dụng.
- Tạo DataLoader và kiểm tra batch, shape, kiểu dữ liệu, nhãn.
- Tạo notebook khám phá dữ liệu.

**File sở hữu**
- src/data.py
- notebooks/01_dataset_exploration.ipynb
- data/README.md
- reports/sections/01_data.md

**Bàn giao**
- Module dữ liệu có hướng dẫn gọi rõ ràng.
- Notebook thể hiện ảnh mẫu, lớp, shape và preprocessing.
- Mô tả split, transforms và các kiểm tra đã thực sự chạy.

**Phối hợp:** TV3 dùng DataLoader; TV2 cần biết kích thước/kênh ảnh và normalization.

## TV2 — Model Architecture
**Branch dự kiến:** feature/model

**Trách nhiệm**
- Khảo sát kiến trúc pre-trained; các ứng viên ban đầu là VGG16 và DenseNet121, tùy khả năng phần cứng.
- Tải weights có sẵn từ Torchvision và in cấu trúc mô hình.
- Xác định backbone, classifier và tổng số/trainable parameters.
- Thay classifier để đầu ra có 10 lớp.
- Hỗ trợ chiến lược freeze backbone và mở các lớp được chọn để fine-tune.

**File sở hữu**
- src/models.py
- reports/sections/02_model.md

**Bàn giao**
- Hàm tạo model với giao diện được ghi rõ.
- Đầu ra là logits cho 10 lớp; không thêm Softmax trước CrossEntropyLoss.
- Ghi lại kiến trúc, weights, số tham số và chính sách freeze/unfreeze.

**Phối hợp:** TV3 cần biết cách lấy model và tham số trainable; TV6 kiểm tra tích hợp.

## TV3 — Training Pipeline
**Branch dự kiến:** feature/training

**Trách nhiệm**
- Xây dựng vòng lặp training và validation.
- Tính loss/accuracy, backward và optimizer update.
- Cho phép cấu hình learning rate, batch size, epochs và optimizer.
- Ghi metrics theo epoch và phối hợp với TV4 về TensorBoard.
- Lưu checkpoint tốt nhất theo validation metric đã thống nhất.

**File sở hữu**
- src/train.py
- reports/sections/03_training.md

**Bàn giao**
- Pipeline training gọi được model và DataLoader từ module của các thành viên.
- Lịch sử train/validation và metadata cấu hình được lưu.
- Có hướng dẫn chạy smoke test; ghi rõ checkpoint được lưu ở đâu.

**Phối hợp:** TV1 cung cấp loaders; TV2 cung cấp model; TV4 thống nhất metrics/logging; TV5 cung cấp experiment config.

## TV4 — Evaluation & TensorBoard
**Branch dự kiến:** feature/evaluation

**Trách nhiệm**
- Xây dựng đánh giá trên test sau khi cấu hình đã được chọn.
- Tổng hợp test loss, test accuracy và các chỉ số bổ sung đã thống nhất.
- Theo dõi/đọc TensorBoard: train/validation loss, accuracy và dấu hiệu overfitting.
- Tạo biểu đồ/bảng từ kết quả thực tế để dùng trong báo cáo.
- Phân tích confusion matrix hoặc lỗi theo lớp nếu nhóm có thời gian và phạm vi phù hợp.

**File sở hữu**
- src/evaluate.py
- notebooks/02_results_analysis.ipynb
- reports/sections/04_evaluation.md

**Bàn giao**
- Module đánh giá nhận checkpoint đã chọn và test loader.
- Notebook đọc bảng metrics và tạo visualization.
- Ghi nhận kết quả thực tế, lệnh chạy và nhận xét dựa trên bằng chứng.

**Quy tắc:** không dùng test metrics để chọn hyperparameters hay mô hình.

## TV5 — Experiments & Hyperparameters
**Branch dự kiến:** feature/experiments

**Trách nhiệm**
- Thiết kế bảng thực nghiệm: so sánh kiến trúc, learning rate, batch size và freeze/fine-tune.
- Giữ cố định các yếu tố khác khi có thể để kết luận có ý nghĩa.
- Quản lý cấu hình run, tên run và metadata.
- Tổng hợp metrics thật vào bảng kết quả; phối hợp TV3/TV4 để bảo đảm có thể truy xuất mỗi kết quả về cấu hình tương ứng.
- Viết phân tích so sánh, không dự đoán trước kết quả.

**File sở hữu**
- configs/
- outputs/experiment_summary.csv (schema và kết quả đã đo)
- reports/sections/05_experiments.md

**Bàn giao**
- Danh sách thí nghiệm và cấu hình có thể tái lập.
- Bảng so sánh với giá trị đo thật, không có số liệu bịa.
- Nhận xét về tác động của các yếu tố dựa trên kết quả.

**Lưu ý:** số thí nghiệm, epochs và giá trị hyperparameter chưa được quy định chính xác trong ảnh đề; cần điều chỉnh theo tài nguyên và yêu cầu bổ sung của giảng viên.

## TV6 — Integration & Project Lead
**Branch dự kiến:** feature/integration

**Trách nhiệm**
- Điều phối cấu trúc repository và giao diện giữa các module.
- Kiểm tra môi trường, Python/PyTorch/Torchvision, driver và khả năng nhận GPU.
- Xây dựng entry point main.py để gọi pipeline; không sao chép logic lõi của các module.
- Quản lý README, requirements, .gitignore, tài liệu và báo cáo tổng.
- Review PR, xử lý xung đột, kiểm tra pipeline end-to-end và khả năng tái lập.
- Phối hợp để kiểm tra lưu/tải checkpoint và inference minh họa.

**File sở hữu**
- main.py
- src/utils.py, src/predict.py
- README.md, requirements.txt, .gitignore
- docs/
- reports/Lab2_Report.md
- reports/sections/06_integration.md

**Bàn giao**
- Hướng dẫn cài đặt/chạy rõ ràng.
- Pipeline tích hợp đã được chạy kiểm tra thực tế.
- Báo cáo tổng hợp, Git workflow và checklist bàn giao.

## Bảng sở hữu file nhanh

| Thành viên | Module/file chính | Phần báo cáo |
|---|---|---|
| TV1 | src/data.py, notebook khám phá dữ liệu | 01_data.md |
| TV2 | src/models.py | 02_model.md |
| TV3 | src/train.py | 03_training.md |
| TV4 | src/evaluate.py, notebook phân tích | 04_evaluation.md |
| TV5 | configs/, bảng experiment | 05_experiments.md |
| TV6 | main.py, utils, docs, README, integration | 06_integration.md, báo cáo tổng |

## Điều kiện tối thiểu trước khi mở PR
Mỗi PR cần có:
1. Mô tả mục tiêu và file đã thay đổi.
2. Cách chạy/kiểm tra.
3. Kết quả thực tế của các kiểm tra đã chạy; không ghi “pass” nếu chưa chạy.
4. Cập nhật phần báo cáo liên quan.
5. TODO, giới hạn hoặc thay đổi giao diện module (nếu có).

## Trạng thái branch
Tài liệu này chỉ ghi tên branch dự kiến. Theo yêu cầu hiện tại, sáu branch công việc chưa được tạo. Chỉ tạo chúng sau khi tài liệu phân công và file cấu trúc được kiểm tra.
