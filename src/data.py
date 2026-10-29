from copy import copy

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split


def get_transforms(img_size=(32, 32)):
    """Định nghĩa pipeline biến đổi ảnh (Transforms & Normalization) cho các tập."""
    # Thông số Normalization chuẩn theo ImageNet pre-trained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transform cho tập Train: Có Data Augmentation (Lật ngang, crop ngẫu nhiên)
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Transform cho tập Validation & Test: Chỉ Convert Tensor và Normalize
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_test_transform


def get_dataloaders(data_dir='./data', batch_size=64, val_split=0.1, seed=42, num_workers=2, img_size=(32, 32)):
    """Tải CIFAR-10, chia tập Train/Val/Test cố định seed và trả về DataLoaders."""
    train_transform, val_test_transform = get_transforms(img_size)

    # 1. Tải Dataset CIFAR-10
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_test_transform
    )

    # 2. Chia Train / Validation tái lập (Reproducible Split) bằng Seed
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, val_indices = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Dùng bản sao nông để train và validation có transform riêng nhưng dùng chung dữ liệu.
    val_full_train_dataset = copy(full_train_dataset)
    val_full_train_dataset.transform = val_test_transform
    val_dataset = Subset(val_full_train_dataset, val_indices.indices)

    # 3. Đóng gói DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    class_names = full_train_dataset.classes

    return train_loader, val_loader, test_loader, class_names


# ----------------------------------------------------
# Khối kiểm thử TV1 (Kiểm tra khi chạy trực tiếp file)
# ----------------------------------------------------
if __name__ == '__main__':
    print("--- [TV1 Smoke Test] Đang kiểm tra module src/data.py ---")
    
    # Chạy thử hàm tạo dataloader
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=32, seed=42)
    
    # Lấy thử 1 batch dữ liệu từ train_loader
    images, labels = next(iter(train_loader))
    
    # Kiểm tra các tiêu chí TV1 bắt buộc
    print(f"1. Kích thước Batch (Shape): {images.shape}")   # Trả về [32, 3, 32, 32]
    print(f"2. Kiểu dữ liệu ảnh (Dtype): {images.dtype}")   # Trả về torch.float32
    print(f"3. Kiểu dữ liệu nhãn: {labels.dtype}")         # Trả về torch.long
    print(f"4. Nhãn nằm trong khoảng: {labels.min().item()} -> {labels.max().item()}") # Phải từ 0 đến 9
    print(f"5. Số lớp nhãn: {len(classes)} ({classes})")
    
    print("\n Smoke Test hoàn tất thành công! ")