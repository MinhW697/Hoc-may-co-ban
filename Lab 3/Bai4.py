# Bài 4: Quy trình ML hoàn chỉnh - Phần 3: Thử nghiệm mô hình Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load: giá nhà California (đơn vị y: 100.000 USD), 8 đặc trưng
#    (lần chạy đầu sklearn sẽ tải dữ liệu từ Internet và lưu cache)
housing = fetch_california_housing()
X, y = housing.data, housing.target
print("Shape dữ liệu:", X.shape)
print("Các đặc trưng:", housing.feature_names)

# Chia Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# 2. Preprocess (Scaling) + 3. Train
#    Scaler nằm trong Pipeline nên chỉ fit trên tập train -> tránh rò rỉ dữ liệu test
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
pipe.fit(X_train, y_train)

# 4. Evaluate
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nKết quả trên tập Test:")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {np.sqrt(mse):.4f}  (≈ lệch trung bình {np.sqrt(mse) * 100000:,.0f} USD)")
print(f"  MAE  = {mean_absolute_error(y_test, y_pred):.4f}")
print(f"  R²   = {r2_score(y_test, y_pred):.4f}")

# Hệ số trên dữ liệu đã chuẩn hóa -> so sánh được mức ảnh hưởng giữa các đặc trưng
coefs = pipe.named_steps["model"].coef_
print("\nHệ số của từng đặc trưng (sau scaling):")
for name, c in sorted(zip(housing.feature_names, coefs), key=lambda t: -abs(t[1])):
    print(f"  {name:<12} {c:+.4f}")

# 5. Visualize một phần kết quả
n = 200
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Giá thật vs giá dự đoán (200 mẫu test): điểm càng gần đường y = x thì dự đoán càng đúng
axes[0].scatter(y_test[:n], y_pred[:n], s=15, alpha=0.6)
lims = [0, max(y_test[:n].max(), y_pred[:n].max())]
axes[0].plot(lims, lims, "r--", label="Dự đoán hoàn hảo (y = x)")
axes[0].set_xlabel("Giá thật (100.000 USD)")
axes[0].set_ylabel("Giá dự đoán (100.000 USD)")
axes[0].set_title(f"Thật vs Dự đoán ({n} mẫu test)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# So sánh trực tiếp 30 mẫu test đầu tiên
m = 30
axes[1].plot(range(m), y_test[:m], "o-", label="Giá thật")
axes[1].plot(range(m), y_pred[:m], "s--", label="Giá dự đoán")
axes[1].set_xlabel("Mẫu thứ")
axes[1].set_ylabel("Giá (100.000 USD)")
axes[1].set_title(f"So sánh {m} mẫu test đầu tiên")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle(f"Linear Regression - California Housing (MSE = {mse:.3f})")
plt.tight_layout()
plt.savefig("Bai4_california.png")
print("\nĐã lưu biểu đồ vào Bai4_california.png")
plt.show()
