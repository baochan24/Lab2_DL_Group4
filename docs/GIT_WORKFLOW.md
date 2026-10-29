# Git workflow nhóm

## Branch dự kiến
- `feature/data` — TV1
- `feature/model` — TV2
- `feature/training` — TV3
- `feature/evaluation` — TV4
- `feature/experiments` — TV5
- `feature/integration` — TV6

## Quy trình
1. Cập nhật branch từ `main` theo quy ước nhóm.
2. Chỉ sửa file thuộc phạm vi được giao, trừ khi đã phối hợp.
3. Commit nhỏ, thông điệp rõ ràng.
4. Trước PR: xem `git diff`, loại dataset/cache/log/checkpoint lớn và secrets.
5. PR nêu mục tiêu, file thay đổi, lệnh kiểm tra, output thực tế, TODO/giới hạn.
6. Review trước merge; không force-push branch dùng chung.
7. Nếu thay đổi API, báo trước cho thành viên phụ thuộc.

## Ví dụ commit
- `docs: clarify CIFAR-10 data workflow`
- `feat(data): add CIFAR-10 dataloaders`
- `feat(model): adapt pretrained classifier to 10 classes`
- `test(training): add one-batch smoke test`
