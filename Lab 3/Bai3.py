# Bài 3: Trực quan hóa kết quả
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dùng lại mô hình của Bài 2: diabetes, 1 đặc trưng BMI
diabetes = load_diabetes()
idx = diabetes.feature_names.index("bmi")
X = diabetes.data[:, [idx]]
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))

# Đường hồi quy: dự đoán trên dải giá trị BMI từ nhỏ nhất đến lớn nhất
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.figure(figsize=(8, 5))
# Biểu đồ tán xạ dữ liệu thực tế
plt.scatter(X_train, y_train, s=15, alpha=0.4, color="gray", label="Dữ liệu Train")
plt.scatter(X_test, y_test, s=25, color="tab:blue", label="Dữ liệu Test (thực tế)")
# Đường thẳng hồi quy trên cùng biểu đồ
plt.plot(x_line, y_line, color="tab:red", linewidth=2,
         label=f"Đường hồi quy: y = {model.coef_[0]:.1f}x + {model.intercept_:.1f}")
plt.xlabel("BMI (đã chuẩn hóa)")
plt.ylabel("Mức độ tiến triển bệnh")
plt.title(f"Hồi quy tuyến tính trên tập Diabetes (R² test = {r2:.3f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Bai3_regression.png")
print("Đã lưu biểu đồ vào Bai3_regression.png")
plt.show()
