"""TV6 — demo dự đoán một ảnh (phần trình bày bổ sung).

TODO:
- Nạp kiến trúc và checkpoint cuối.
- Áp dụng đúng transforms khi inference.
- Trả về nhãn dự đoán và confidence nếu dùng softmax để hiển thị.
"""
def predict_image(image_path: str, checkpoint_path: str):
    raise NotImplementedError("TV6: triển khai sau khi có checkpoint đã huấn luyện")

"""Demo dự đoán một ảnh — phần mở rộng tùy chọn.

Không thuộc pipeline thực nghiệm đã hoàn thành.
Đánh giá test được thực hiện bằng src/evaluate.py.
Hàm bên dưới chưa được triển khai.
"""