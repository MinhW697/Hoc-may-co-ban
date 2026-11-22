# Bài 1: Xây dựng mô hình Logistic Regression
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load dữ liệu: 569 khối u, 30 đặc trưng; nhãn 0 = malignant (ác tính), 1 = benign (lành tính)
data = load_breast_cancer()
X, y = data.data, data.target
print("Shape X:", X.shape, "| Shape y:", y.shape)
print("Các lớp:", data.target_names.tolist(), "| Số mẫu mỗi lớp:", np.bincount(y))

# 2. Chia Train/Test (80/20), stratify để giữ tỉ lệ 2 lớp như nhau ở train và test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# 3. Chuẩn hóa: fit scaler CHỈ trên tập train, rồi áp dụng cho cả train và test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\nSau scaling - mean train ≈ {X_train_scaled.mean():.4f} | std train ≈ {X_train_scaled.std():.4f}")

# 4. Huấn luyện Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
print("\nĐã huấn luyện xong LogisticRegression")
print(f"Số hệ số (coef_): {model.coef_.shape[1]} | Intercept: {model.intercept_[0]:.4f}")
print(f"Accuracy trên tập Train: {model.score(X_train_scaled, y_train):.4f}")

# Xác suất dự đoán của 5 mẫu test đầu tiên (cột 0: P(malignant), cột 1: P(benign))
print("\nXác suất dự đoán 5 mẫu test đầu tiên:")
print(np.round(model.predict_proba(X_test_scaled[:5]), 4))
print("Nhãn dự đoán:", model.predict(X_test_scaled[:5]), "| Nhãn thật:", y_test[:5])
