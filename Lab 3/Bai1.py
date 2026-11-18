# Bài 1: Xây dựng mô hình Hồi quy tuyến tính
import numpy as np
from sklearn.linear_model import LinearRegression

# Tạo dữ liệu giả lập: y = 2X + 1 + nhiễu
rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=(100, 1))   # 100 mẫu, 1 đặc trưng (sklearn yêu cầu X dạng 2D)
noise = rng.normal(0, 1, size=100)      # nhiễu Gauss, mean 0, std 1
y = 2 * X[:, 0] + 1 + noise

print("5 mẫu đầu tiên:")
for xi, yi in zip(X[:5, 0], y[:5]):
    print(f"  X = {xi:6.3f} -> y = {yi:7.3f}")

# Huấn luyện mô hình Hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# In hệ số của mô hình
print("\nHệ số góc (coefficient):", round(model.coef_[0], 4), "(giá trị thật: 2)")
print("Hệ số chặn (intercept):  ", round(model.intercept_, 4), "(giá trị thật: 1)")
print(f"Phương trình học được: y = {model.coef_[0]:.4f} * X + {model.intercept_:.4f}")

# Thử dự đoán với vài giá trị X mới
X_new = np.array([[0], [5], [10]])
for xi, yp in zip(X_new[:, 0], model.predict(X_new)):
    print(f"Dự đoán X = {xi:2d} -> y = {yp:.3f}")
