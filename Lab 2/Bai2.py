# Bài 2: Chuẩn hóa dữ liệu (Scaling)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Tạo mảng dữ liệu: cột 0 là Tuổi (20-60), cột 1 là Thu nhập (10tr-100tr, đơn vị VNĐ)
rng = np.random.default_rng(42)
n = 10
tuoi = rng.integers(20, 61, size=n)
thu_nhap = rng.integers(10, 101, size=n) * 1_000_000
X = np.column_stack([tuoi, thu_nhap]).astype(float)

cols = ["Tuổi", "Thu nhập"]
print("Dữ liệu gốc:")
print(pd.DataFrame(X, columns=cols).astype(np.int64))

# StandardScaler: z = (x - mean) / std  -> mean = 0, std = 1
X_std = StandardScaler().fit_transform(X)

# MinMaxScaler: x' = (x - min) / (max - min)  -> đưa về khoảng [0, 1]
X_mm = MinMaxScaler().fit_transform(X)

# In kết quả để so sánh
pd.set_option("display.width", 120)
compare = pd.DataFrame({
    "Tuổi": X[:, 0].astype(int),
    "Thu nhập": X[:, 1].astype(np.int64),
    "Tuổi (Std)": X_std[:, 0],
    "Thu nhập (Std)": X_std[:, 1],
    "Tuổi (MinMax)": X_mm[:, 0],
    "Thu nhập (MinMax)": X_mm[:, 1],
})
print("\nSo sánh kết quả chuẩn hóa:")
print(compare.round(3))

print("\nThống kê sau StandardScaler (mean ≈ 0, std ≈ 1):")
print(pd.DataFrame({"mean": X_std.mean(axis=0), "std": X_std.std(axis=0)}, index=cols).round(3))
print("\nThống kê sau MinMaxScaler (min = 0, max = 1):")
print(pd.DataFrame({"min": X_mm.min(axis=0), "max": X_mm.max(axis=0)}, index=cols).round(3))
