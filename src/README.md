# Trách nhiệm module `src/`

- `data.py` — TV1: CIFAR-10, split, transforms, DataLoaders.
- `models.py` — TV2: pre-trained models, thay classifier, freeze/fine-tune.
- `train.py` — TV3: train/validation loop, optimizer/loss, TensorBoard scalar phối hợp TV4, checkpoint.
- `evaluate.py` — TV4: đánh giá checkpoint đã chọn trên test, metrics.
- `utils.py` — TV6: seed, cấu hình, tiện ích dùng chung.
- `predict.py` — TV6: dự đoán một ảnh minh họa (phần bổ sung).
- `main.py` ở thư mục gốc — TV6: điều phối pipeline, không nhân bản logic module.
