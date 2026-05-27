import time
import threading
import queue
from utils.roi_filter import build_roi_polygon

class InferenceThread(threading.Thread):
    """
    Luồng xử lý suy diễn YOLO cho riêng từng camera.
    Đọc khung hình từ CameraCaptureThread, chạy suy diễn YOLO và chuyển kết quả sang TrackingThread.
    """
    def __init__(
        self,
        camera_code: str,
        cap_thread,
        tracking_queue: queue.Queue,
        camera_zones: dict,
        yolo_model,
        model_lock: threading.Lock,
        confidence: float,
        device: str,
        data_lock: threading.Lock
    ):
        super().__init__(name=f"InferenceThread_{camera_code}", daemon=True)
        self.camera_code = camera_code
        self.cap_thread = cap_thread
        self.tracking_queue = tracking_queue
        self.camera_zones = camera_zones
        self.yolo_model = yolo_model
        self.model_lock = model_lock
        self.confidence = confidence
        self.device = device
        self.data_lock = data_lock
        self.running = True

    def run(self):
        print(f"[InferenceThread_{self.camera_code}] Started.")
        is_cuda = (self.device == "cuda")
        
        while self.running:
            # Lấy khung hình mới nhất từ hàng đợi capture
            try:
                frame_data = self.cap_thread.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame = frame_data["frame"]
            capture_time = frame_data["capture_time"]
            h, w = frame.shape[:2]

            # Chạy suy diễn bằng YOLO (dùng model_lock để an toàn luồng với TensorRT/CUDA)
            with self.model_lock:
                results = self.yolo_model(
                    frame,
                    imgsz=640,
                    conf=self.confidence,
                    verbose=False,
                    device=self.device,
                    half=is_cuda
                )

            from core.metrics import GLOBAL_METRICS
            GLOBAL_METRICS.increment_inference(1)

            frame_result = results[0]
            
            # Lấy danh sách các vùng (zones) động từ trạng thái chia sẻ
            with self.data_lock:
                zones = self.camera_zones.get(self.camera_code)

            # Xây dựng trước roi_pts để dùng cho bám vết sau này
            if zones:
                for zone in zones:
                    if zone["roi_pts"] is None or zone["frame_size"] != (w, h):
                        zone["roi_pts"] = build_roi_polygon(w, h, zone["polygon"])
                        zone["frame_size"] = (w, h)

            raw_detections = []
            if frame_result.boxes is not None:
                for box in frame_result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Chỉ lọc các lớp 'xe đạp' (1), 'ô tô' (2), 'xe máy' (3) và 'xe tải' (7)
                    if class_id not in [1, 2, 3, 7]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw_detections.append(([x1, y1, x2, y2], class_id, confidence))

            # Đóng gói dữ liệu gửi sang tracking thread
            tracking_data = {
                "frame": frame,
                "detections": raw_detections,
                "camera_zones": zones,
                "capture_time": capture_time
            }

            # Đưa vào tracking queue bằng "empty put" (lấy frame mới bỏ frame cũ)
            try:
                if self.tracking_queue.full():
                    self.tracking_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.tracking_queue.put_nowait(tracking_data)
            except queue.Full:
                pass

        print(f"[InferenceThread_{self.camera_code}] Stopped.")

    def stop(self):
        self.running = False
