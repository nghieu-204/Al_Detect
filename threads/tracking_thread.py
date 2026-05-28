import time
import threading
import queue
import json
import numpy as np
from utils.roi_filter import inside_roi

class TrackingThread(threading.Thread):
    """
    Luồng bám vết (Tracking) riêng biệt cho từng camera.
    Nhận các phát hiện thô (raw detections) từ hàng đợi queue (maxsize=1),
    sử dụng ByteTrack để liên kết/bám sát vật thể, lọc theo ROI và gửi lên MQTT.
    """
    def __init__(
        self,
        camera_code: str,
        tracking_queue: queue.Queue,
        mqtt_client,
        bbox_topic_template: str,
        ai_module: str,
        class_names: dict,
        lpd_queue=None,
        track_plates_registry=None,
        registry_lock=None
    ):
        super().__init__(name=f"TrackingThread_{camera_code}", daemon=True)
        self.camera_code = camera_code
        self.tracking_queue = tracking_queue
        self.mqtt_client = mqtt_client
        self.bbox_topic_template = bbox_topic_template
        self.ai_module = ai_module
        self.class_names = class_names
        self.lpd_queue = lpd_queue
        self.track_plates_registry = track_plates_registry
        self.registry_lock = registry_lock
        self.track_lpd_states = {}  # track_id -> {"attempts": int, "last_attempt_time": float}
        self.running = True

    def run(self):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from types import SimpleNamespace

        print(f"[TrackingThread_{self.camera_code}] Started.")
        
        # Cấu hình các thông số của ByteTrack để khóa mục tiêu nhạy hơn
        args = SimpleNamespace(
            track_buffer=30,           # Số lượng frame giữ vết khi mất dấu
            track_high_thresh=0.25,    # Giảm xuống 0.25 để nhận diện các xe ở xa/bị che khuất sớm hơn
            track_low_thresh=0.1,      # Ngưỡng tin cậy cho liên kết giai đoạn 2
            new_track_thresh=0.3,      # Cho phép tạo vết mới ngay khi tự tin đạt >= 0.3
            match_thresh=0.8,          # Ngưỡng khoảng cách tối đa để khớp
            fuse_score=True            # Kết hợp điểm tin cậy
        )
        tracker = BYTETracker(args)

        while self.running:
            try:
                # Lấy dữ liệu thô từ queue (timeout ngắn để có thể dừng luồng khi cần)
                data = self.tracking_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame = data["frame"]
            raw_dets = data["detections"]
            zones = data["camera_zones"]
            capture_time = data.get("capture_time", time.time())
            
            from core.metrics import GLOBAL_METRICS
            latency = time.time() - capture_time
            GLOBAL_METRICS.add_tracking_latency(self.camera_code, latency)
            h, w = frame.shape[:2]

            # Nếu có kết quả phát hiện thô, tiến hành cập nhật ByteTrack
            bboxes_xyxy = []
            if len(raw_dets) > 0:
                scores = []
                classes = []
                for box, class_id, conf in raw_dets:
                    bboxes_xyxy.append(box)
                    scores.append(conf)
                    classes.append(class_id)

                # Chuyển đổi từ định dạng xyxy [x1, y1, x2, y2] sang center-x, center-y, width, height (xywh)
                bboxes_xywh = []
                for box in bboxes_xyxy:
                    x1, y1, x2, y2 = box
                    xc = (x1 + x2) / 2.0
                    yc = (y1 + y2) / 2.0
                    width = x2 - x1
                    height = y2 - y1
                    bboxes_xywh.append([xc, yc, width, height])

                # Tạo class giả lập có các thuộc tính tương thích với cấu trúc BYTETracker
                class CustomResults:
                    def __init__(self, xywh, conf, cls):
                        self.xywh = np.array(xywh, dtype=np.float32)
                        self.conf = np.array(conf, dtype=np.float32)
                        self.cls = np.array(cls, dtype=np.float32)

                    def __len__(self):
                        return len(self.xywh)

                    def __getitem__(self, index):
                        return CustomResults(self.xywh[index], self.conf[index], self.cls[index])

                tracker_input = CustomResults(bboxes_xywh, scores, classes)
                try:
                    tracked_outputs = tracker.update(tracker_input, img=frame)
                except Exception as e:
                    print(f"[TrackingThread_{self.camera_code}] ByteTrack update error: {e}")
                    tracked_outputs = []
            else:
                # Không có phát hiện nào, chạy cập nhật tracker trống
                class EmptyResults:
                    def __init__(self):
                        self.xywh = np.empty((0, 4), dtype=np.float32)
                        self.conf = np.empty((0,), dtype=np.float32)
                        self.cls = np.empty((0,), dtype=np.float32)
                    def __len__(self): return 0
                    def __getitem__(self, index): return self
                try:
                    tracked_outputs = tracker.update(EmptyResults(), img=frame)
                except Exception:
                    tracked_outputs = []

            # Lọc kết quả bám vết qua đa giác ROI
            has_roi = bool(zones)
            detections = []
            for row in tracked_outputs:
                # row cấu trúc: [x1_kf, y1_kf, x2_kf, y2_kf, track_id, score, class_id, idx]
                x1_kf, y1_kf, x2_kf, y2_kf, track_id, score, class_id, idx = row
                class_id = int(class_id)
                track_id = int(track_id)

                # Mặc định sử dụng tọa độ dự đoán của bộ lọc Kalman (ByteTrack)
                x1, y1, x2, y2 = x1_kf, y1_kf, x2_kf, y2_kf

                # Nếu tracking khớp với phát hiện thô hiện tại từ YOLO, sử dụng trực tiếp tọa độ của YOLO
                # để loại bỏ hoàn toàn độ trễ do bộ lọc Kalman làm mượt (giúp bbox bám sát vật thể tức thì)
                original_idx = int(idx)
                if len(bboxes_xyxy) > 0 and 0 <= original_idx < len(bboxes_xyxy):
                    x1, y1, x2, y2 = bboxes_xyxy[original_idx]

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                inside_any = True
                if has_roi:
                    inside_any = False
                    for zone in zones:
                        if zone.get("roi_pts") is not None:
                            if inside_roi(cx, cy, zone["roi_pts"]):
                                inside_any = True
                                break

                if not inside_any:
                    continue

                # 1. Kiểm tra kết quả biển số trong registry dùng chung
                plate_info = None
                has_plate = False
                if self.track_plates_registry is not None:
                    registry_key = f"{self.camera_code}_{track_id}"
                    with self.registry_lock:
                        raw_plate = self.track_plates_registry.get(registry_key)
                    if raw_plate is not None:
                        if raw_plate.get("confidence", 0.0) > 0.7:
                            has_plate = True
                        
                        # Tái cấu trúc tọa độ biển số theo vị trí xe ở frame hiện tại (chống trễ hình)
                        rel_bbox = raw_plate["relative_bbox"]
                        rx1, ry1, rx2, ry2 = rel_bbox
                        
                        w_veh = float(x2 - x1)
                        h_veh = float(y2 - y1)
                        
                        g_px1 = max(0.0, min(float(x1) + rx1 * w_veh, float(w)))
                        g_py1 = max(0.0, min(float(y1) + ry1 * h_veh, float(h)))
                        g_px2 = max(0.0, min(float(x1) + rx2 * w_veh, float(w)))
                        g_py2 = max(0.0, min(float(y1) + ry2 * h_veh, float(h)))
                        
                        print(f"[TrackingThread_{self.camera_code}] Track {track_id} Veh: {x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}, Plate: {g_px1:.1f},{g_py1:.1f},{g_px2:.1f},{g_py2:.1f}")
                        
                        plate_info = {
                            "bbox": [g_px1 / w, g_py1 / h, g_px2 / w, g_py2 / h],
                            "confidence": raw_plate["confidence"]
                        }

                # 2. Gửi yêu cầu nhận diện biển số xe (nếu chưa tìm thấy biển số tin cậy)
                if self.lpd_queue is not None and not has_plate:
                    state = self.track_lpd_states.setdefault(track_id, {"attempts": 0, "last_attempt_time": 0.0, "last_seen": 0.0})
                    state["last_seen"] = time.time()
                    
                    attempts = state["attempts"]
                    last_attempt_time = state["last_attempt_time"]
                    
                    width_px = x2 - x1
                    height_px = y2 - y1
                    size_ok = (width_px > 30) and (height_px > 30)
                    time_ok = (time.time() - last_attempt_time) > 0.5
                    
                    if attempts < 3 and size_ok and time_ok and not self.lpd_queue.full():
                        cx1 = max(0, int(x1))
                        cy1 = max(0, int(y1))
                        cx2 = min(w, int(x2))
                        cy2 = min(h, int(y2))
                        
                        if (cx2 - cx1) > 10 and (cy2 - cy1) > 10:
                            crop_img = frame[cy1:cy2, cx1:cx2].copy()
                            request_data = {
                                "camera_code": self.camera_code,
                                "track_id": track_id,
                                "crop_img": crop_img,
                                "vehicle_bbox": [x1, y1, x2, y2],
                                "frame_size": (w, h),
                                "timestamp": time.time()
                            }
                            try:
                                self.lpd_queue.put_nowait(request_data)
                                state["attempts"] = attempts + 1
                                state["last_attempt_time"] = time.time()
                                print(f"[TrackingThread_{self.camera_code}] Pushed vehicle crop for track_{track_id} to LPD queue (size: {width_px}x{height_px}, attempt: {attempts + 1})")
                            except queue.Full:
                                pass

                # Chuẩn hóa tọa độ bbox [0.0 - 1.0]
                bbox = [
                    float(x1) / w,
                    float(y1) / h,
                    float(x2) / w,
                    float(y2) / h
                ]
                cls_name = self.class_names.get(class_id, "unknown")
                obj_id = f"obj_{track_id}"

                detection_item = {
                    "id": obj_id,
                    "cls": cls_name,
                    "class": cls_name,
                    "label": cls_name,
                    "class_id": class_id,
                    "confidence": float(score),
                    "bbox": bbox,
                    "color": "#00ff00"
                }

                # Đính kèm thông tin biển số xe nếu tìm thấy trong registry (để lưu trữ/xử lý backend)
                if plate_info is not None:
                    detection_item["license_plate"] = {
                        "bbox": plate_info["bbox"],
                        "confidence": plate_info["confidence"]
                    }
                    print(f"[TrackingThread_{self.camera_code}] Attaching plate for track_{track_id} to MQTT (conf: {plate_info['confidence']:.2f})")

                detections.append(detection_item)

                # Đồng thời thêm biển số xe như một đối tượng detect độc lập để Web vẽ BBox trực tiếp
                if plate_info is not None:
                    plate_detection = {
                        "id": f"plate_{track_id}",
                        "cls": "license_plate",
                        "class": "license_plate",
                        "label": "license_plate",
                        "class_id": 99,  # ID giả lập cho lớp biển số
                        "confidence": plate_info["confidence"],
                        "bbox": plate_info["bbox"],
                        "color": "#ff0000"  # Vẽ BBox biển số màu đỏ cho nổi bật
                    }
                    detections.append(plate_detection)

            # Dọn dẹp trạng thái LPD của các track đã lâu không xuất hiện (sau 60 giây)
            now = time.time()
            expired_tids = [tid for tid, st in self.track_lpd_states.items() if now - st.get("last_seen", 0.0) > 60.0]
            for tid in expired_tids:
                del self.track_lpd_states[tid]

            # Publish kết quả bám vết của camera này lên MQTT
            self.publish_detections(detections, zones)

        print(f"[TrackingThread_{self.camera_code}] Stopped.")

    def publish_detections(self, detections, zones):
        formatted_zones = []
        formatted_polygons = []
        if zones:
            for z in zones:
                poly_pts = z.get("polygon") or []
                formatted_zones.append({
                    "id": z.get("zone_id", "default"),
                    "points": [{"x": p[0], "y": p[1]} for p in poly_pts]
                })
            formatted_polygons = [z.get("polygon") or [] for z in zones]

        topic = self.bbox_topic_template.format(camera_code=self.camera_code)
        payload = {
            "camera_code": self.camera_code,
            "ai_module": self.ai_module,
            "ai_modules": [self.ai_module],
            "timestamp": time.time(),
            "detections": detections,
            "zones": formatted_zones,
            "polygons": formatted_polygons,
            "roi": formatted_zones,
            "active_areas": formatted_zones
        }
        self.mqtt_client.publish(
            topic,
            json.dumps(payload, ensure_ascii=False),
            qos=0
        )
        # Xoá log in ra terminal để tránh trôi màn hình

    def stop(self):
        self.running = False
