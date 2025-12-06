# Hướng dẫn chạy ứng dụng Nhận diện Biển báo Giao thông

Dự án bao gồm hai phần chính:
1. **Backend**: API Server sử dụng FastAPI (Python) để xử lý hình ảnh và chạy mô hình nhận diện.
2. **Frontend**: Ứng dụng Web sử dụng ReactJS (Vite) để người dùng tải ảnh lên và xem kết quả.

## Yêu cầu hệ thống

*   **Python**: Phiên bản 3.8 trở lên.
*   **Node.js**: Phiên bản 16 trở lên (kèm theo `npm`).

---

## 1. Cài đặt và chạy Backend

Backend chịu trách nhiệm load các mô hình (SVM, VGG16) và cung cấp API dự đoán.

### Bước 1: Cài đặt các thư viện cần thiết
Tại thư mục gốc của dự án, chạy lệnh sau để cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Server
Di chuyển vào thư mục backend và khởi động server:

```bash
cd backend
uvicorn main:app --reload
```

*   Server sẽ chạy tại địa chỉ: `http://127.0.0.1:8000`
*   API docs (Swagger UI) có thể xem tại: `http://127.0.0.1:8000/docs`

**Lưu ý:** Backend cần truy cập các file mô hình đã được huấn luyện nằm ở thư mục gốc của dự án:
*   `hog_results/svm_kernel-rbf_C-10_gamma-scale/`
*   `vgg16_results_64x64/`

---

## 2. Cài đặt và chạy Frontend

Frontend là giao diện web cho phép người dùng tương tác.

### Bước 1: Di chuyển vào thư mục frontend
Mở một terminal **mới** (giữ terminal backend đang chạy) và chạy:

```bash
cd frontend
```

### Bước 2: Cài đặt các thư viện Node.js
Chạy lệnh:

```bash
npm install
```

### Bước 3: Chạy ứng dụng ở chế độ Development
Chạy lệnh:

```bash
npm run dev
```

*   Ứng dụng sẽ chạy tại địa chỉ (thường là): `http://localhost:5173` (kiểm tra terminal để biết chính xác port).

---

## 3. Sử dụng

1.  Mở trình duyệt và truy cập địa chỉ của Frontend (ví dụ: `http://localhost:5173`).
2.  Nhấn nút **Upload Image** để chọn ảnh biển báo giao thông từ máy tính.
3.  Kết quả nhận diện từ cả mô hình SVM (HOG features) và VGG16 sẽ hiển thị trên màn hình.

