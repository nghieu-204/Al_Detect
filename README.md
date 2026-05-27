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
- **`threads/camera_capture.py`**: Chuyên biệt để đọc RTSP (Capture Thread). Sử dụng tiến trình `ffmpeg.exe` độc lập cho mỗi camera để quản lý kết nối, tự động decode bằng GPU CUDA/NVDEC (với CPU fallback tự động nếu lỗi khởi tạo GPU), resize trực tiếp về 640x360 và xuất BGR24 qua stdout. Áp dụng cơ chế hàng đợi `maxsize=1` cùng logic empty-put để loại bỏ độ trễ tích lũy.
- **`threads/inference_thread.py`**: Luồng xử lý suy diễn YOLO cho riêng từng camera. Đọc khung hình từ hàng đợi capture, chạy suy diễn YOLO và chuyển kết quả sang TrackingThread.
- **`threads/tracking_thread.py`**: Nhận Bounding Box sau khi AI phân tích, tích hợp giải thuật Tracking (ByteTrack), lọc theo ROI và đóng gói dữ liệu thành JSON đẩy lên MQTT.

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
- **Lấy RTSP stream phù hợp:** Lựa chọn luồng RTSP tương ứng với module AI cần thiết để đảm bảo đúng nguồn dữ liệu.

### 4.2 Xử lý RTSP stream
- **Dùng OpenCV đọc frame:** Sử dụng `cv2.VideoCapture` với backend FFmpeg để kết nối và đọc luồng RTSP ổn định.
- **Tách thread riêng để đọc frame:** Mỗi camera chạy trên một thread độc lập, không gây ảnh hưởng lẫn nhau.
- **Reconnect khi mất kết nối:** Thread tự động thử kết nối lại với thời gian chờ tăng dần (Exponential Backoff) khi mất tín hiệu.
- **Giảm buffer để giảm latency:** Tắt các bộ đệm mặc định của FFmpeg (`-fflags nobuffer`, `-flags low_delay`) để giảm thiểu độ trễ hình ảnh.

### 4.3 Queue
- **Sử dụng `queue.Queue(maxsize=1)`:** Mỗi camera có một hàng đợi riêng với kích thước tối đa là 1 phần tử.
- **Luôn giữ các frame mới nhất:** Khi hàng đợi đầy, frame cũ bị loại bỏ ngay lập tức để nhường chỗ cho frame mới (thuật toán Empty-Put).
- **Mục đích:**
  - Tránh hiện tượng delay tích lũy (Latency Accumulation) theo thời gian.
  - Đảm bảo mô hình AI luôn suy diễn trên khung hình thực gần nhất (Realtime).

### 4.4 YOLO Inference
- **Sử dụng model TensorRT:** Nạp mô hình dạng `.engine` giúp tối ưu hóa phần cứng GPU tối đa.
- **Input frame từ queue:** Lấy frame trực tiếp từ hàng đợi của các camera và gom thành Batch trước khi đẩy vào GPU.
- **Detect các class:** `car` (ô tô), `motorcycle` (xe máy), `bus` (xe buýt), `truck` (xe tải).

### 4.5 MQTT Publish
- **Gửi kết quả detection:** Đóng gói kết quả detection (Bounding Box, ID theo dõi, tọa độ chuẩn hóa) thành JSON.
- **Topic:** `smart_vms/ai/bbox/{camera_code}`
- **QoS:** `0` (Tốc độ cao, giảm overhead mạng cho luồng dữ liệu liên tục).

---

## 5. Các biện pháp Tối ưu hệ thống

### 5.1 Tách thread xử lý
- **Thread 1:** Đọc RTSP (Capture Thread) — chạy độc lập cho từng camera.
- **Thread 2:** Inference AI (Batch Inference Thread) — gom tất cả camera để xử lý song song trên GPU.
- **Thread 3:** Tracking & Publish (Tracking Thread) — chạy độc lập cho từng camera.
- **Hiệu quả:** Tránh bị block lẫn nhau, toàn bộ pipeline chạy song song hoàn toàn.

### 5.2 Queue giảm latency
- Không xử lý frame cũ, dồn toàn bộ sức mạnh tính toán để xử lý frame mới nhất.
- Đảm bảo Bbox sinh ra khớp với hình ảnh đang diễn ra tại thời điểm đó.

### 5.3 Tối ưu hóa mô hình
- **Chuyển đổi sang TensorRT:** Convert mô hình sang định dạng `.engine`.
- **Lượng tử hóa FP16:** Set `half=True` để nén trọng số mô hình từ FP32 xuống còn FP16, tăng gấp đôi tốc độ suy diễn và tiết kiệm một nửa VRAM.
- **Giảm độ phân giải:** Hạ kích thước đầu vào `imgsz=640` thay vì full HD `1920` để giảm khối lượng tính toán và tăng tốc độ xử lý nhiều lần.

### 5.4 Reconnect RTSP tự động
- Thiết kế cơ chế tự động phát hiện mất kết nối và thử lại theo thời gian chờ tăng dần (Exponential Backoff, tối đa 30 giây).
- Tránh crash toàn bộ hệ thống khi mất mạng camera cục bộ.

### 5.5 Giải mã và Stream bằng GPU/CPU FFMPEG Subprocess
- **Vấn đề**: Mặc định OpenCV decode RTSP bằng CPU rất tốn tài nguyên và dễ tích lũy độ trễ hình ảnh khi mạng lag.
- **Giải pháp**:
  - Mỗi camera sử dụng một tiến trình `ffmpeg.exe` riêng để tự động quản lý kết nối và dòng dữ liệu RTSP.
  - Tự động kiểm tra phần cứng và driver GPU. Nếu hỗ trợ, FFMPEG sẽ giải mã phần cứng bằng CUDA/NVDEC (`-hwaccel cuda`) trực tiếp trên GPU.
  - Nếu GPU khởi tạo lỗi hoặc không khả dụng, hệ thống tự động fallback chuyển sang decode bằng CPU giúp duy trì tính ổn định liên tục.
  - Tích hợp resize về `640x360` và convert sang pixel format `bgr24` trực tiếp trong luồng FFMPEG, truyền luồng frame thô nhị phân qua ống dẫn `stdout` đến Python.
  - Python đọc trực tiếp buffer tĩnh (`640 * 360 * 3` bytes) bằng chế độ blocking cực nhanh qua `np.frombuffer`, loại bỏ hoàn toàn chi phí decode gián tiếp.
  - Sử dụng cơ chế hàng đợi `maxsize=1` cùng logic Empty-Put: Chỉ đẩy frame mới nhất vào hàng đợi cho AI xử lý, giải phóng frame cũ ngay lập tức nếu AI chưa tiêu thụ xong, triệt tiêu hoàn toàn độ trễ tích lũy.

### 5.6 Tối ưu hóa Target Lock & Độ trễ Bám vết
- Giảm ngưỡng tin cậy của giải thuật ByteTrack (`track_high_thresh = 0.25`, `new_track_thresh = 0.3`) giúp khóa mục tiêu nhanh hơn ngay khi vật thể xuất hiện từ xa.
- Ghi đè (override) tọa độ dự báo trơn mượt của bộ lọc Kalman bằng chính tọa độ thô từ mô hình YOLO khi khớp vết. Điều này loại bỏ hoàn toàn độ trễ hiển thị bbox "lững thững" theo sau vật thể của Kalman Filter truyền thống.
