import time
import threading
import queue
import cv2
import numpy as np
from ultralytics import YOLO

class LPDThread(threading.Thread):
    """
    Luồng toàn cục nhận các yêu cầu crop ảnh từ `lpd_queue`,
    chạy mô hình nhận diện biển số xe (best.pt),
    chuyển đổi tọa độ cục bộ về tọa độ gốc của camera
    và lưu kết quả vào bảng đăng ký dùng chung `track_plates_registry`.
    """
    def __init__(
        self,
        lpd_queue: queue.Queue,
        lpd_model,
        confidence: float,
        device: str,
        model_lock: threading.Lock,
        registry: dict,
        registry_lock: threading.Lock
    ):
        super().__init__(name="LPDThread", daemon=True)
        self.lpd_queue = lpd_queue
        self.model = lpd_model
        self.confidence = confidence
        self.device = device
        self.model_lock = model_lock
        self.registry = registry
        self.registry_lock = registry_lock
        self.running = True

    def run(self):
        is_cuda = (self.device == "cuda")
        print("[LPDThread] Started processing queue.")
        last_cleanup_time = time.time()

        while self.running:
            # Dọn dẹp registry định kỳ (mỗi 5 phút) để tránh tràn bộ nhớ RAM
            now = time.time()
            if now - last_cleanup_time > 300.0:
                last_cleanup_time = now
                with self.registry_lock:
                    expired_keys = [k for k, v in self.registry.items() if now - v.get("timestamp", 0) > 600.0]
                    for k in expired_keys:
                        del self.registry[k]

            requests = []
            try:
                # Lấy yêu cầu crop đầu tiên (blocking với timeout ngắn)
                first_request = self.lpd_queue.get(timeout=0.1)
                requests.append(first_request)
            except queue.Empty:
                continue

            # Gom tiếp các yêu cầu crop đang chờ sẵn trong hàng đợi (non-blocking)
            max_batch_size = 8
            while len(requests) < max_batch_size:
                try:
                    req = self.lpd_queue.get_nowait()
                    requests.append(req)
                except queue.Empty:
                    break

            # Lọc bỏ các yêu cầu không hợp lệ (ảnh crop rỗng)
            valid_requests = []
            for req in requests:
                crop_img = req.get("crop_img")
                if crop_img is not None and crop_img.size > 0:
                    valid_requests.append(req)
                else:
                    self.lpd_queue.task_done()

            if not valid_requests:
                continue

            if len(valid_requests) > 1:
                print(f"[LPDThread] Gom batch thành công: {len(valid_requests)} ảnh crop xe để nhận diện biển số.")

            # Thực hiện suy diễn batch
            batch_imgs = [req["crop_img"] for req in valid_requests]
            results = None
            try:
                with self.model_lock:
                    results = self.model(
                        batch_imgs,
                        imgsz=640,
                        conf=self.confidence,
                        verbose=False,
                        device=self.device,
                        half=is_cuda
                    )
            except Exception as e:
                # Nếu mô hình TensorRT tĩnh (batch=1) báo lỗi, tự động fallback sang chạy tuần tự
                results = []
                for req in valid_requests:
                    try:
                        with self.model_lock:
                            res = self.model(
                                req["crop_img"],
                                imgsz=640,
                                conf=self.confidence,
                                verbose=False,
                                device=self.device,
                                half=is_cuda
                            )
                            results.append(res[0])
                    except Exception as seq_err:
                        print(f"[LPDThread] Lỗi fallback tuần tự cho track_{req['track_id']}: {seq_err}")
                        results.append(None)

            # Xử lý kết quả của từng yêu cầu trong batch
            for i, req in enumerate(valid_requests):
                try:
                    result = results[i] if (results and i < len(results)) else None
                    if result is None:
                        continue

                    camera_code = req["camera_code"]
                    track_id = req["track_id"]
                    crop_img = req["crop_img"]
                    vehicle_bbox = req["vehicle_bbox"]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Lấy phát hiện biển số có độ tin cậy cao nhất
                        best_box = None
                        best_conf = -1.0
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf > best_conf:
                                best_conf = conf
                                best_box = box
                        
                        if best_box is not None:
                            # Lấy trực tiếp tọa độ pixel tuyệt đối trên ảnh crop
                            px1, py1, px2, py2 = map(float, best_box.xyxy[0])
                            vx1, vy1, vx2, vy2 = vehicle_bbox
                            
                            # Kích thước xe gốc (trước khi bị clip bởi camera boundary)
                            w_veh_orig = float(vx2 - vx1)
                            h_veh_orig = float(vy2 - vy1)
                            
                            # Tọa độ góc trên bên trái của ảnh crop thực tế
                            cx1 = max(0, int(vx1))
                            cy1 = max(0, int(vy1))
                            
                            # Tọa độ của biển số xe trên hệ quy chiếu của xe gốc (chưa clip)
                            px1_on_veh = px1 + (cx1 - vx1)
                            py1_on_veh = py1 + (cy1 - vy1)
                            px2_on_veh = px2 + (cx1 - vx1)
                            py2_on_veh = py2 + (cy1 - vy1)
                            
                            if w_veh_orig > 0 and h_veh_orig > 0:
                                rel_x1 = px1_on_veh / w_veh_orig
                                rel_y1 = py1_on_veh / h_veh_orig
                                rel_x2 = px2_on_veh / w_veh_orig
                                rel_y2 = py2_on_veh / h_veh_orig
                            else:
                                rel_x1, rel_y1, rel_x2, rel_y2 = 0.0, 0.0, 0.0, 0.0
                                
                            # Đảm bảo các giá trị không bị vượt quá giới hạn [0, 1] nếu có sai lệch nhỏ
                            rel_x1 = max(0.0, min(rel_x1, 1.0))
                            rel_y1 = max(0.0, min(rel_y1, 1.0))
                            rel_x2 = max(0.0, min(rel_x2, 1.0))
                            rel_y2 = max(0.0, min(rel_y2, 1.0))
                            
                            # print(f"[LPDThread] [{camera_code}] Track {track_id} Crop: {crop_img.shape[1]}x{crop_img.shape[0]}, Plate local: {px1:.1f},{py1:.1f},{px2:.1f},{py2:.1f}, Relative: {rel_x1:.3f},{rel_y1:.3f},{rel_x2:.3f},{rel_y2:.3f}")
                            
                            # Lưu vào Registry dùng chung
                            registry_key = f"{camera_code}_{track_id}"
                            with self.registry_lock:
                                self.registry[registry_key] = {
                                    "relative_bbox": [rel_x1, rel_y1, rel_x2, rel_y2],
                                    "confidence": best_conf,
                                    "timestamp": time.time()
                                }
                            # print(f"[LPDThread] [{camera_code}] SUCCESS: Found plate for track_{track_id} (conf: {best_conf:.2f})")
                    else:
                        pass
                        # print(f"[LPDThread] [{camera_code}] FAILED: No plate found for track_{track_id}")
                except Exception as post_err:
                    print(f"[LPDThread] Lỗi hậu xử lý kết quả cho track_{req['track_id']}: {post_err}")
                finally:
                    self.lpd_queue.task_done()

        print("[LPDThread] Stopped.")

    def stop(self):
        self.running = False
