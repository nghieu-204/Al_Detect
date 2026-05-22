# Hệ thống AI Camera Detect (Smart VMS AI Worker)

Hệ thống AI xử lý luồng camera thực tế (RTSP) dựa trên kiến trúc đa luồng, kết hợp YOLOv8 và bộ lọc vùng (ROI) đa giác. Hệ thống cho phép bám vết đối tượng, nhận diện phương tiện/khuôn mặt và gửi kết quả theo thời gian thực (realtime) qua MQTT.

---

## 1. Môi trường & Cấu hình
- **Yêu cầu môi trường:** Python 3.8+, CUDA & cuDNN (nếu dùng GPU).
- **requirements.txt:** Chứa các thư viện cần thiết (`ultralytics`, `opencv-python`, `paho-mqtt`, `shapely`, `imageio-ffmpeg`, `torch`).
- **config.json:** File cấu hình tập trung chứa thông số MQTT Broker (IP, Port, Account, Company ID), Topic cấu trúc, và cấu hình AI (imgsz, confidence, đường dẫn model YOLO).

## 2. Cấu trúc Source Code

### Utils (Các hàm hỗ trợ)
- **`utils/helpers.py`**: Các hàm xử lý chung như cấu hình MQTT Client, bóc tách chuỗi JSON, lọc danh sách module AI phù hợp, chuyển đổi tọa độ điểm ROI thành polygon (Shapely).
- **`utils/roi_filter.py`**: Xử lý logic lọc Bounding Box. Kiểm tra xem tọa độ hộp giới hạn (Bounding Box) sinh ra từ AI có nằm trong/cắt ngang vùng đa giác cấu hình (ROI Polygon) hay không.

### Threads (Luồng xử lý)
Hệ thống sử dụng thư viện `threading`, `queue`, và `subprocess` để tối ưu Multi-threading.
- **`threads/camera_capture.py`**: Chuyên biệt để đọc RTSP (Capture Thread). Mở kết nối luồng camera qua `ffmpeg`/`opencv` và lấy raw frame liên tục.
- **`threads/batch_inference.py`**: Gom lô dữ liệu (Batching) từ tất cả các camera để đẩy vào Model AI nhận diện đồng loạt, tối đa hoá hiệu suất GPU thay vì suy luận từng frame lẻ.
- **`threads/tracking_thread.py`**: Nhận Bounding Box sau khi AI phân tích, tích hợp giải thuật Tracking (như ByteTrack của YOLO), và đóng gói dữ liệu thành JSON đẩy vào hàng đợi MQTT.

### Entry Point
- **`main.py`**: File khởi chạy chính. Khởi tạo MQTT, lắng nghe danh sách Camera trực tuyến (`ONLINE`) từ backend. Khởi tạo song song các Thread Capture, Inference, và Tracking cho từng camera tương ứng.

---

## 3. Cài đặt và Chạy hệ thống

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Cài đặt FFmpeg:** Đảm bảo hệ điều hành đã được cài đặt FFmpeg và đưa vào biến môi trường (PATH) để hỗ trợ decode GPU.
3. **Khởi chạy hệ thống:**
   ```bash
   python main.py
   ```

---

## 4. Luồng xử lý dữ liệu chi tiết

### 4.1 Nhận camera từ MQTT
- **Subscribe topic:** Hệ thống đăng ký lắng nghe trên topic được chỉ định để nhận thông tin cấu hình từ trung tâm.
- **Lọc camera ONLINE:** Chỉ những camera có trạng thái `ONLINE` mới được đưa vào danh sách xử lý.
- **Lấy RTSP stream phù hợp:** Lựa chọn luồng RTSP tương ứng với module AI (ví dụ: FACE, ALPR) cần thiết để đảm bảo đúng nguồn dữ liệu.

### 4.2 Xử lý RTSP stream
- **Dùng OpenCV / Subprocess:** Đọc frame từ luồng camera.
- **Tách thread riêng:** Mỗi camera chạy trên một thread độc lập để đọc frame, không gây ảnh hưởng lẫn nhau.
- **Tự động reconnect:** Khi mất kết nối, thread tự động dò lại độ phân giải và thử kết nối lại dần theo thời gian.
- **Giảm buffer (latency):** Tắt các bộ đệm mặc định để giảm thiểu độ trễ hình ảnh.

### 4.3 Quản lý Queue
- **Cấu trúc hàng đợi:** Sử dụng `queue.Queue(maxsize=2)` (hoặc `maxsize=1`).
- **Chiến lược giữ frame:** Luôn giữ các frame mới nhất. Khi hàng đợi đầy, loại bỏ các frame cũ trước khi đưa frame mới vào.
- **Mục đích:** Tránh hiện tượng delay tích lũy theo thời gian, đảm bảo mô hình luôn infer trên khung hình thực (realtime) nhất.

### 4.4 YOLO Inference
- **Sử dụng model TensorRT:** Nạp mô hình dạng `.engine` giúp tối ưu hóa phần cứng GPU.
- **Đầu vào:** Lấy frame trực tiếp từ queue của các camera.
- **Mục tiêu nhận diện:** `car`, `motorcycle`, `bus`, `truck` (hoặc `face`, tuỳ module).

### 4.5 MQTT Publish
- **Gửi kết quả:** Đóng gói kết quả detection (Bounding Box, ID theo dõi) thành JSON.
- **Topic:** `smart_vms/ai/bbox/{camera_code}`
- **QoS:** `1` (Đảm bảo tin cậy).

---

## 5. Các biện pháp Tối ưu hệ thống

### 5.1 Tách thread xử lý
- **Thread 1:** Đọc RTSP (Capture Thread).
- **Thread 2:** Inference AI và publish kết quả (Batch Inference/Tracking).
- **Hiệu quả:** Giúp tránh bị block các luồng, chạy song song với nhau mượt mà.

### 5.2 Queue giảm latency
- Không xử lý frame cũ, dồn sức mạnh tính toán để giữ frame mới nhất. Đảm bảo Bbox sinh ra khớp với hình ảnh đang diễn ra tại thời điểm đó.

### 5.3 Tối ưu hóa mô hình AI
- **Chuyển đổi sang TensorRT:** Convert mô hình sang định dạng `.engine`.
- **Lượng tử hóa:** Set `half = True` để nén trọng số mô hình từ FP32 xuống còn FP16.
- **Giảm độ phân giải:** Hạ kích thước đầu vào `imgsz = 640` thay vì full HD `1920` để tăng tốc độ xử lý nhiều lần.

### 5.4 Reconnect RTSP tự động
- Thiết kế cơ chế tự động thăm dò và thử lại kết nối. Tránh crash toàn bộ hệ thống khi mất mạng camera cục bộ.

### 5.5 Decode frame bằng GPU (FFmpeg Hardware Acceleration)
- **Vấn đề:** Decode Full HD bằng CPU của OpenCV rất chậm và tốn chi phí copy chuyển ảnh từ RAM lên VRAM của GPU, gây ra độ trễ cao (delay Bbox).
- **Giải pháp:** 
  - Dùng **FFmpeg (`-hwaccel cuda`)** để decode trực tiếp trên GPU.
  - Tích hợp lệnh resize từ 1920x1080 xuống 640x320 ngay trong quá trình decode bằng GPU.
  - Đọc raw frame thông qua tiến trình con (subprocess), giảm đáng kể băng thông copy về RAM và khối lượng tính toán.
