# Nghiên cứu và xây dựng mô hình học sâu đa nhiệm dựa trên CLIP trong việc nhận diện và định lượng bệnh lý trên lá cây cà phê

## Giới thiệu
Dự án này sử dụng mô hình Multi-Task Learning (MTL) dựa trên kiến trúc CLIP (Contrastive Language-Image Pretraining) để đồng thời nhận diện loại bệnh và đánh giá mức độ nghiêm trọng trên lá cây cà phê.

## Cấu trúc thư mục
11220189_Hồ Thị Kim Anh_Nghiên cứu và xây dựng mô hình học sâu đa nhiệm dựa trên CLIP trong việc nhận diện và định lượng bệnh lý trên lá cây cà phê.zip
├── 11220189_Hồ Thị Kim Anh_Nghiên cứu và xây dựng mô hình học sâu đa nhiệm dựa trên CLIP trong việc nhận diện và định lượng bệnh lý trên lá cây cà phê.pdf
├── Source/
│   ├── app.py
│   ├── model.py
│   ├── Multi_task_learning.ipynb
│   ├── requirements.txt
│   └── models/
│       └── model.pt
└── README.md

- `Source/`: Chứa mã nguồn ứng dụng.
  - `app.py`: File chạy chính (Gradio interface).
  - `model.py`: Định nghĩa kiến trúc mạng Multi-task.
  - `Multi_task_learning.ipynb`: Notebook huấn luyện và đánh giá mô hình.
  - `models/`: Chứa trọng số mô hình đã huấn luyện (`model.pt`).
  - `requirements.txt`: Danh sách các thư viện cần thiết.
- `11220189_Hồ Thị Kim Anh_Nghiên cứu và xây dựng mô hình học sâu đa nhiệm dựa trên CLIP trong việc nhận diện và định lượng bệnh lý trên lá cây cà phê.pdf`: File báo cáo chi tiết.

## Hướng dẫn cài đặt và chạy
1. **Cài đặt môi trường**:
   Nên sử dụng Python 3.9+. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r Source/requirements.txt
   ```

2. **Tải trọng số mô hình**:
   - Tải file `model.pt` từ mục [Releases](https://github.com/HTKA0314/coffee-leaf-disease-clip-mtl/releases) (hoặc liên kết lưu trữ đám mây).
   - Đặt file `model.pt` vào thư mục: `Source/models/model.pt`

3. **Chạy ứng dụng**:
   Di chuyển vào thư mục `Source` và chạy:
   ```bash
   python app.py
   ```
   Sau khi chạy, truy cập đường dẫn local (thường là `http://127.0.0.1:7860`) để sử dụng giao diện.

## Tính năng
- Nhận diện 4 loại bệnh chính: Rỉ sắt, Sâu vẽ bùa, Phoma, Đốm mắt cua.
- Phân loại mức độ nghiêm trọng (0-4).
- Đưa ra khuyến nghị canh tác, sinh học và hóa học dựa trên kết quả chẩn đoán.
