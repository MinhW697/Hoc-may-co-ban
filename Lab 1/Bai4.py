# Bài 4: Quy trình ML hoàn chỉnh - Phần 1: Thu thập và Kiểm tra dữ liệu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Tải tập dữ liệu digits
digits = load_digits()
X, y = digits.data, digits.target

# 2. In ra hình dạng của dữ liệu và nhãn
print("Shape dữ liệu (X):", X.shape)        # (1797, 64): 1797 ảnh, mỗi ảnh 8x8 = 64 pixel
print("Shape nhãn (y):", y.shape)           # (1797,)
print("Shape ảnh gốc (images):", digits.images.shape)  # (1797, 8, 8)
print("Các nhãn:", np.unique(y))
print("Số mẫu mỗi nhãn:", np.bincount(y))

# 3. In giá trị của mẫu đầu tiên
print("\nNhãn của mẫu đầu tiên:", y[0])
print("Giá trị pixel mẫu đầu tiên (8x8):")
print(digits.images[0])

# 4. Hiển thị một số mẫu dữ liệu dưới dạng hình ảnh
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, img, label in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"Nhãn: {label}")
    ax.axis("off")
plt.suptitle("Một số mẫu trong tập Digits")
plt.tight_layout()
plt.savefig("Bai4_digits.png")
print("\nĐã lưu hình ảnh vào Bai4_digits.png")
plt.show()
