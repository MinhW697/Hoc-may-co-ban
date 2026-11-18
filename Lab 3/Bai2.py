# Bài 2: Đánh giá mô hình
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tải tập diabetes, chỉ lấy 1 đặc trưng: BMI (chỉ số khối cơ thể)
diabetes = load_diabetes()
feature = "bmi"
idx = diabetes.feature_names.index(feature)
X = diabetes.data[:, [idx]]   # giữ dạng 2D: (442, 1)
y = diabetes.target           # chỉ số tiến triển bệnh sau 1 năm
print("Đặc trưng sử dụng:", feature, "| Shape X:", X.shape)

# Chia Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# Huấn luyện
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\nPhương trình: y = {model.coef_[0]:.2f} * {feature} + {model.intercept_:.2f}")

# Đánh giá trên tập Test
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nKết quả trên tập Test:")
print(f"  MAE = {mae:.2f}")
print(f"  MSE = {mse:.2f}  (RMSE = {mse ** 0.5:.2f})")
print(f"  R²  = {r2:.4f}")

# Ý nghĩa các chỉ số:
# - MAE (Mean Absolute Error) = trung bình |y_thật - y_dự_đoán|.
#   Cho biết trung bình mô hình dự đoán lệch bao nhiêu đơn vị so với giá trị thật,
#   cùng đơn vị với y nên dễ diễn giải. Càng nhỏ càng tốt.
# - MSE (Mean Squared Error) = trung bình (y_thật - y_dự_đoán)².
#   Bình phương sai số nên phạt nặng các dự đoán lệch nhiều (nhạy với ngoại lệ).
#   Đơn vị là bình phương của y -> thường lấy căn (RMSE) để đưa về đơn vị gốc. Càng nhỏ càng tốt.
# - R² (R-squared, hệ số xác định) = 1 - SS_res / SS_tot.
#   Tỉ lệ phương sai của y được mô hình giải thích. R² = 1: hoàn hảo;
#   R² = 0: không tốt hơn việc luôn dự đoán bằng giá trị trung bình; R² < 0: tệ hơn cả trung bình.
#   Ở đây R² chỉ khoảng 0.2 vì chỉ dùng 1 đặc trưng (BMI): BMI có liên quan
#   đến tiến triển bệnh nhưng một mình nó chưa đủ để dự đoán chính xác.
