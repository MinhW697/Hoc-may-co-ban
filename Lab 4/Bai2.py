# Bài 2: Đánh giá mô hình phân loại
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Dựng lại mô hình của Bài 1 (cùng dữ liệu, cùng random_state)
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Dự đoán trên tập Test
y_pred = model.predict(X_test_scaled)

# Các chỉ số (mặc định lớp dương pos_label = 1 = benign)
acc = accuracy_score(y_test, y_pred)    # tỉ lệ dự đoán đúng trên toàn bộ tập test
prec = precision_score(y_test, y_pred)  # trong các mẫu dự đoán là benign, bao nhiêu % thật sự benign
rec = recall_score(y_test, y_pred)      # trong các mẫu thật sự benign, mô hình tìm ra được bao nhiêu %
print("Kết quả trên tập Test (lớp dương = benign):")
print(f"  Accuracy  = {acc:.4f}")
print(f"  Precision = {prec:.4f}")
print(f"  Recall    = {rec:.4f}")

# Trong y tế, quan trọng nhất là không bỏ sót khối u ác tính -> xét thêm Recall của lớp malignant (0)
rec_malignant = recall_score(y_test, y_pred, pos_label=0)
print(f"  Recall lớp malignant = {rec_malignant:.4f}")

# Báo cáo chi tiết theo từng lớp
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names, digits=4))
