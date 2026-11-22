# Bài 3: Ma trận nhầm lẫn (Confusion Matrix)
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Dựng lại mô hình và dự đoán như Bài 1-2
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
y_pred = model.predict(X_test_scaled)

# Tạo ma trận nhầm lẫn: hàng = nhãn thật, cột = nhãn dự đoán, thứ tự lớp [0, 1]
#                      Dự đoán 0 (malignant)   Dự đoán 1 (benign)
#   Thật 0 (malignant)        TN                      FP
#   Thật 1 (benign)           FN                      TP
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Giải thích các thành phần (lớp dương = 1 = benign, lớp âm = 0 = malignant):
# - TP (True Positive):  thật là benign,    dự đoán benign    -> dự đoán ĐÚNG lớp dương
# - TN (True Negative):  thật là malignant, dự đoán malignant -> dự đoán ĐÚNG lớp âm
# - FP (False Positive): thật là malignant, dự đoán benign    -> kết luận nhầm "lành tính" cho khối u ác tính
#                        (lỗi nguy hiểm nhất trong bài toán này: bỏ sót bệnh nhân ung thư)
# - FN (False Negative): thật là benign,    dự đoán malignant -> báo động giả, bệnh nhân phải xét nghiệm thêm
tn, fp, fn, tp = cm.ravel()
print(f"\nTN = {tn} | FP = {fp} | FN = {fn} | TP = {tp}")

# Kiểm tra lại các chỉ số của Bài 2 từ ma trận nhầm lẫn
print(f"Accuracy  = (TP+TN)/Tổng = {(tp + tn) / cm.sum():.4f}")
print(f"Precision = TP/(TP+FP)   = {tp / (tp + fp):.4f}")
print(f"Recall    = TP/(TP+FN)   = {tp / (tp + fn):.4f}")

# Trực quan hóa bằng ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Breast Cancer)")
plt.tight_layout()
plt.savefig("Bai3_confusion_matrix.png")
print("\nĐã lưu hình vào Bai3_confusion_matrix.png")
plt.show()
