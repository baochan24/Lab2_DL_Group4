# Kiểm thử dự kiến

Các kiểm thử chưa được triển khai; không ghi là pass trước khi chạy thật.

- TV1: DataLoader trả đúng shape, dtype, nhãn 0–9; split tái lập.
- TV2: model output `[batch_size, 10]`; kiểm tra trainable parameters.
- TV3: smoke test một batch, loss hữu hạn, checkpoint tạo được.
- TV4: checkpoint reload và tính metrics.
- TV6: import/module integration và command end-to-end.
