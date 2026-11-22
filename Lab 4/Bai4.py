# Bài 4: Quy trình ML hoàn chỉnh - Phần 4: Áp dụng Logistic Regression vào tập Digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data: 1797 ảnh 8x8 (64 pixel), 10 lớp từ 0 đến 9
digits = load_digits()
X, y = digits.data, digits.target
print("Shape X:", X.shape, "| Các lớp:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# Preprocess (StandardScaler) + 2. Logistic Regression đa lớp
#    Với solver mặc định (lbfgs), LogisticRegression tự xử lý multi-class bằng
#    hàm softmax (multinomial): mỗi lớp có 1 bộ trọng số, chọn lớp có xác suất cao nhất.
#    Scaler nằm trong Pipeline nên chỉ fit trên tập train -> tránh rò rỉ dữ liệu test.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)
print("Shape coef_:", pipe.named_steps["model"].coef_.shape, "(10 lớp x 64 pixel)")

# 3. Đánh giá trên tập Test
y_pred = pipe.predict(X_test)
print(f"\nAccuracy Train = {pipe.score(X_train, y_train):.4f}")
print(f"Accuracy Test  = {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (hàng = thật, cột = dự đoán):")
print(confusion_matrix(y_test, y_pred))

# 4. In ra 5 mẫu dự đoán sai
wrong_idx = np.where(y_pred != y_test)[0]
print(f"\nSố mẫu dự đoán sai: {len(wrong_idx)} / {len(y_test)}")
proba = pipe.predict_proba(X_test)
show = wrong_idx[:5]
print("5 mẫu dự đoán sai đầu tiên:")
for i in show:
    print(f"  Mẫu test #{i:<4} Nhãn thật = {y_test[i]} | Dự đoán = {y_pred[i]}"
          f" (xác suất {proba[i, y_pred[i]]:.2f})")

# Hiển thị ảnh của các mẫu dự đoán sai
fig, axes = plt.subplots(1, len(show), figsize=(2.4 * len(show), 3.4))
for ax, i in zip(np.atleast_1d(axes), show):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"Thật: {y_test[i]}\nDự đoán: {y_pred[i]}", color="red")
    ax.axis("off")
plt.suptitle("5 mẫu Digits bị dự đoán sai")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig("Bai4_wrong_digits.png")
print("\nĐã lưu hình vào Bai4_wrong_digits.png")
plt.show()
