# Bài 3: Chia tập dữ liệu Train/Test
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Tải tập dữ liệu diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Chia dữ liệu: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In kích thước tập gốc và các tập sau khi chia
print("Tập dữ liệu gốc:")
print("  X:", X.shape, "| y:", y.shape)
print("Tập train (80%):")
print("  X_train:", X_train.shape, "| y_train:", y_train.shape)
print("Tập test (20%):")
print("  X_test:", X_test.shape, "| y_test:", y_test.shape)
print(f"\nTỉ lệ: train = {len(X_train) / len(X):.0%}, test = {len(X_test) / len(X):.0%}")
