# Bài 4: Quy trình ML hoàn chỉnh - Phần 2: Tiền xử lý dữ liệu
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Tải tập dữ liệu digits (tiếp tục từ Lab 1)
digits = load_digits()
X, y = digits.data, digits.target
print("Shape dữ liệu:", X.shape, "| Khoảng giá trị pixel:", X.min(), "-", X.max())

# 1. Chuẩn hóa dữ liệu (minh họa): pixel 0-16 -> mean 0, std 1
X_scaled = StandardScaler().fit_transform(X)
print("Sau khi chuẩn hóa: mean ≈ %.3f, std ≈ %.3f" % (X_scaled.mean(), X_scaled[:, X.std(axis=0) > 0].std()))

# 2. Chia tập Train/Test (80/20), giữ tỉ lệ các nhãn bằng stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# 3. Pipeline: chuẩn hóa + model
#    Scaler nằm trong Pipeline nên chỉ được fit trên tập train -> tránh rò rỉ dữ liệu test
pipelines = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf")),
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]),
}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"Độ chính xác trên tập test: {accuracy_score(y_test, y_pred):.4f}")

print("\nBáo cáo phân loại chi tiết (SVM):")
print(classification_report(y_test, pipelines["SVM"].predict(X_test)))
