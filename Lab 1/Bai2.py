# Bài 2: Phân loại bài toán Machine Learning
#
# Supervised (Có giám sát): dữ liệu huấn luyện có nhãn (đáp án) đi kèm.
# Unsupervised (Không giám sát): dữ liệu không có nhãn, mô hình tự tìm cấu trúc.

# 1. Dự đoán giá nhà dựa trên diện tích
#    -> SUPERVISED - bài toán Hồi quy (Regression).
#       Có dữ liệu diện tích kèm giá thật (nhãn là số liên tục).

# 2. Gom nhóm khách hàng dựa trên hành vi mua sắm
#    -> UNSUPERVISED - bài toán Phân cụm (Clustering).
#       Không có nhãn nhóm trước, thuật toán tự gom các khách hàng giống nhau.

# 3. Phân loại email là Spam hay không
#    -> SUPERVISED - bài toán Phân loại nhị phân (Binary Classification).
#       Email huấn luyện đã được gán nhãn Spam / Không spam.

# 4. Nhận diện chữ viết tay từ hình ảnh
#    -> SUPERVISED - bài toán Phân loại đa lớp (Multi-class Classification).
#       Mỗi ảnh được gán nhãn là chữ/số tương ứng (vd: 0-9).

phan_loai = {
    "Dự đoán giá nhà dựa trên diện tích": "Supervised (Regression)",
    "Gom nhóm khách hàng dựa trên hành vi mua sắm": "Unsupervised (Clustering)",
    "Phân loại email là Spam hay không": "Supervised (Classification)",
    "Nhận diện chữ viết tay từ hình ảnh": "Supervised (Classification)",
}

for i, (vi_du, loai) in enumerate(phan_loai.items(), 1):
    print(f"{i}. {vi_du:<45} -> {loai}")
