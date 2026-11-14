# Bài 1: Làm sạch dữ liệu
import numpy as np
import pandas as pd

# Tạo DataFrame có giá trị thiếu (NaN) và giá trị ngoại lệ (outlier)
df = pd.DataFrame({
    "Tuổi": [22, 25, np.nan, 30, 28, 35, 27, 24, 31, 29, 26, np.nan, 33, 23, 150],
    "Chiều cao (cm)": [165, 170, 168, np.nan, 172, 160, 175, 169, 900, 171, 166, 174, np.nan, 167, 173],
    "Cân nặng (kg)": [55, 62, 58, 70, np.nan, 65, 68, 60, 72, 64, 57, 66, 63, 59, 61],
})
print("Dữ liệu ban đầu:")
print(df)
print("\nSố giá trị thiếu mỗi cột:")
print(df.isnull().sum())

# Lưu ý: nếu điền mean trước khi xử lý ngoại lệ thì mean bị ngoại lệ kéo lệch
# (vd: chiều cao 900 làm mean ≈ 225cm). Vì vậy ta phát hiện ngoại lệ trước,
# đánh dấu chúng là NaN, rồi điền mean của dữ liệu "sạch" cho cả hai loại.

# 2. Phát hiện ngoại lệ: |giá trị - mean| > 3 * std (Pandas tự bỏ qua NaN khi tính)
mean = df.mean()
std = df.std()
outlier_mask = (df - mean).abs() > 3 * std
print("\nNgưỡng phát hiện ngoại lệ (mean ± 3*std):")
print(pd.DataFrame({"Cận dưới": mean - 3 * std, "Cận trên": mean + 3 * std}).round(2))
print("\nCác giá trị ngoại lệ phát hiện được:")
for col in df.columns:
    for idx, val in df.loc[outlier_mask[col], col].items():
        print(f"  - Dòng {idx}, cột '{col}': {val}")

# Xử lý ngoại lệ: thay ngoại lệ bằng NaN để điền lại ở bước tiếp theo
df_clean = df.mask(outlier_mask)

# 1. Điền giá trị thiếu (và ngoại lệ đã loại) bằng giá trị trung bình (mean) của cột
col_means = df_clean.mean()
print("\nMean mỗi cột (sau khi loại ngoại lệ) dùng để điền:")
print(col_means.round(2))
df_clean = df_clean.fillna(col_means)

print("\nDữ liệu sau khi làm sạch:")
print(df_clean.round(2))
print("\nThống kê mô tả sau khi làm sạch:")
print(df_clean.describe().round(2))
