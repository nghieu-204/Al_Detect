import time
import threading
import queue
from utils.roi_filter import build_roi_polygon

class BatchInferenceThread(threading.Thread):
    """
    Luồng tập trung thực hiện gom các khung hình từ hàng đợi capture của tất cả các camera đang hoạt động,
    gom nhóm chúng lại (batch), chạy suy diễn mô hình YOLOv8 trong một lượt duy nhất,
    và đẩy kết quả nhận diện cùng khung hình vào hàng đợi queue (maxsize=1) của từng camera để bám vết.
    """
    def __init__(
        self,
        capture_threads: dict,
        camera_zones: dict,
        tracking_queues: dict,
        yolo_model,
        device: str,
        confidence: float,
        data_lock: threading.Lock
    ):
        super().__init__(name="BatchInferenceThread", daemon=True)
        self.capture_threads = capture_threads
        self.camera_zones = camera_zones
        self.tracking_queues = tracking_queues
        self.yolo_model = yolo_model
        self.device = device
        self.confidence = confidence
        self.data_lock = data_lock
        self.running = True

    def run(self):
        print("[BatchInference] Centralized Batch Inference thread started.")
        while self.running:
            loop_start = time.time()
            
            # Truy vấn an sau các luồng capture đang hoạt động
            with self.data_lock:
                active_codes = list(self.capture_threads.keys())
            
            batch_frames = []
            batch_camera_codes = []
            batch_capture_times = []
            
            for camera_code in active_codes:
                cap_thread = self.capture_threads.get(camera_code)
                if cap_thread is None:
                    continue
                
                # Lấy khung hình mới nhất từ hàng đợi một cách không chặn (non-blocking)
                # frame_queue có maxsize=1 và dùng empty-put nên luôn chứa frame mới nhất
                try:
                    frame_data = cap_thread.frame_queue.get_nowait()
                    batch_frames.append(frame_data["frame"])
                    batch_camera_codes.append(camera_code)
                    batch_capture_times.append(frame_data["capture_time"])
                except queue.Empty:
                    continue
            
            if not batch_frames:
                # Nếu không có luồng nào có frame mới, ngủ một lúc rồi thử lại
                time.sleep(0.01)
                continue
            
            # Chạy suy diễn theo batch trên GPU/CPU
            is_cuda = (self.device == "cuda")
            results = self.yolo_model(
                batch_frames,
                imgsz=640,
                conf=self.confidence,
                verbose=False,
                device=self.device,
                half=is_cuda
            )
            
            from core.metrics import GLOBAL_METRICS
            GLOBAL_METRICS.increment_inference(len(batch_frames))
            
            for camera_code, frame, capture_time, frame_result in zip(batch_camera_codes, batch_frames, batch_capture_times, results):
                h, w = frame.shape[:2]
                
                # Lấy danh sách các vùng (zones) động từ trạng thái chia sẻ
                with self.data_lock:
                    zones = self.camera_zones.get(camera_code)
                    
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
                
                # Đóng gói dữ liệu để gửi sang luồng bám vết (tracking thread) của camera tương ứng
                tracking_data = {
                    "frame": frame,
                    "detections": raw_detections,
                    "camera_zones": zones,
                    "capture_time": capture_time
                }
                
                # Đưa vào hàng đợi của camera bằng thuật toán "empty put" (lấy frame mới bỏ frame cũ)
                t_queue = self.tracking_queues.get(camera_code)
                if t_queue is not None:
                    try:
                        if t_queue.full():
                            t_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        t_queue.put_nowait(tracking_data)
                    except queue.Full:
                        pass
            
            # Kiểm soát tốc độ suy diễn FPS (mục tiêu tốc độ toàn vòng lặp ~30 FPS)
            elapsed = time.time() - loop_start
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)
            
        print("[BatchInference] Centralized Batch Inference thread stopped.")

    def stop(self):
        self.running = False
