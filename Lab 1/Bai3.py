# Bài 3: Chạy thử mô hình đơn giản với Scikit-Learn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# Bước 1: Load dữ liệu
iris = load_iris()
X, y = iris.data, iris.target
print("Đặc trưng:", iris.feature_names)
print("Các lớp:", iris.target_names.tolist())
print("Shape X:", X.shape, "| Shape y:", y.shape)

# Bước 2: Khởi tạo model
# (Iris là bài toán phân loại, nhưng dùng Linear Regression để xem model hoạt động)
model = LinearRegression()

# Bước 3: Fit model (huấn luyện)
model.fit(X, y)
print("\nHệ số (coef_):", np.round(model.coef_, 4))
print("Hệ số chặn (intercept_):", round(model.intercept_, 4))

# Bước 4: Predict (dự đoán)
y_pred = model.predict(X)
print("\n5 mẫu đầu tiên:")
print("Giá trị dự đoán (số thực):", np.round(y_pred[:5], 3))
print("Nhãn thật:                ", y[:5])

# Linear Regression trả về số thực -> làm tròn để quy về nhãn lớp 0/1/2
y_pred_class = np.clip(np.round(y_pred), 0, 2).astype(int)
accuracy = (y_pred_class == y).mean()
print(f"\nĐộ chính xác sau khi làm tròn về lớp: {accuracy:.2%}")
print(f"R^2 score: {model.score(X, y):.4f}")

# Nhận xét: Linear Regression dự đoán giá trị liên tục nên không phù hợp
# trực tiếp cho phân loại; ta phải làm tròn thủ công. Với bài toán phân loại
# nên dùng các mô hình như LogisticRegression, KNN, DecisionTree...
