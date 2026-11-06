# Bài 1: Làm quen với Numpy và Pandas
import numpy as np
import pandas as pd

# 1. Mảng Numpy 1 chiều chứa các số từ 1 đến 10
arr = np.arange(1, 11)
print("Mảng Numpy:", arr)
print("Kích thước:", arr.shape, "| Kiểu dữ liệu:", arr.dtype)

# 2. DataFrame chứa thông tin 5 sinh viên
df = pd.DataFrame({
    "Họ tên": ["Nguyễn Văn An", "Trần Thị Bình", "Lê Văn Cường", "Phạm Thị Dung", "Hoàng Văn Em"],
    "Tuổi": [19, 20, 19, 21, 20],
    "Điểm toán": [8.5, 7.0, 9.0, 6.5, 8.0],
})
print("\nDanh sách sinh viên:")
print(df)

# 3. Thông tin tổng quan và điểm trung bình môn toán
print("\nThông tin tổng quan (info):")
df.info()
print("\nThống kê mô tả (describe):")
print(df.describe())

diem_tb = df["Điểm toán"].mean()
print(f"\nĐiểm trung bình môn toán: {diem_tb:.2f}")
