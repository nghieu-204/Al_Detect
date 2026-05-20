import time
import threading
import queue
import json
from roi_filter import build_roi_polygon, inside_roi

class BatchInferenceThread(threading.Thread):
    """
    Centralized thread that pools frames from all active camera capture queues,
    batches them together, runs YOLOv8 model inference in a single batch,
    and publishes detections to MQTT.
    """
    def __init__(
        self,
        capture_threads: dict,
        camera_zones: dict,
        mqtt_client,
        yolo_model,
        device: str,
        confidence: float,
        ai_module: str,
        bbox_topic_template: str,
        data_lock: threading.Lock
    ):
        super().__init__(name="BatchInferenceThread", daemon=True)
        self.capture_threads = capture_threads
        self.camera_zones = camera_zones
        self.mqtt_client = mqtt_client
        self.yolo_model = yolo_model
        self.device = device
        self.confidence = confidence
        self.ai_module = ai_module
        self.bbox_topic_template = bbox_topic_template
        self.data_lock = data_lock
        self.running = True

    def run(self):
        print("[BatchInference] Centralized Batch Inference thread started.")
        while self.running:
            loop_start = time.time()
            
            batch_frames = []
            batch_camera_codes = []
            
            # Safely query active capture threads
            with self.data_lock:
                active_codes = list(self.capture_threads.keys())
                
            for camera_code in active_codes:
                # Get the capture thread reference
                cap_thread = self.capture_threads.get(camera_code)
                if cap_thread is None:
                    continue
                
                # Non-blocking fetch of the latest frame from the queue
                try:
                    frame = cap_thread.frame_queue.get_nowait()
                    batch_frames.append(frame)
                    batch_camera_codes.append(camera_code)
                except queue.Empty:
                    # Capture queue is empty; skip this camera for this batch cycle
                    continue
            
            # If no frames are ready, sleep briefly and retry
            if not batch_frames:
                time.sleep(0.01)
                continue
            
            # Perform batch inference
            if self.yolo_model is not None:
                is_cuda = (self.device == "cuda")
                results = self.yolo_model(batch_frames, imgsz=640, conf=self.confidence, verbose=False, device=self.device, half=is_cuda)
                
                for i, camera_code in enumerate(batch_camera_codes):
                    frame_result = results[i]
                    frame = batch_frames[i]
                    h, w = frame.shape[:2]
                    
                    # Fetch zones dynamically from shared state
                    with self.data_lock:
                        zones = self.camera_zones.get(camera_code)
                        
                    has_roi = bool(zones)
                    if has_roi:
                        for zone in zones:
                            if zone["roi_pts"] is None or zone["frame_size"] != (w, h):
                                zone["roi_pts"] = build_roi_polygon(w, h, zone["polygon"])
                                zone["frame_size"] = (w, h)
                                
                    detections = []
                    
                    for box in frame_result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Filter only 'person' class (ID 0)
                        if class_id != 0:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        inside_any = True
                        if has_roi:
                            inside_any = False
                            for zone in zones:
                                if inside_roi(cx, cy, zone["roi_pts"]):
                                    inside_any = True
                                    break
                                    
                        if not inside_any:
                            continue
                            
                        # Normalize coordinates
                        bbox = [
                            x1 / w,
                            y1 / h,
                            x2 / w,
                            y2 / h
                        ]
                        
                        detections.append({
                            "id": f"obj_{int(time.time() * 1000)}_{class_id}",
                            "cls": self.yolo_model.names[class_id],
                            "class": self.yolo_model.names[class_id],
                            "label": self.yolo_model.names[class_id],
                            "class_id": class_id,
                            "confidence": confidence,
                            "bbox": bbox,
                            "color": "#00ff00"
                        })
                        
                    # Publish detections
                    if detections:
                        print(f"[{camera_code}] Publishing {len(detections)} active detections to MQTT (Batched)")
                    self.publish_detections(camera_code, detections, zones)
                    
            # Control inference FPS (target ~30 FPS overall loop speed)
            elapsed = time.time() - loop_start
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)
            
        print("[BatchInference] Centralized Batch Inference thread stopped.")

    def publish_detections(self, camera_code, detections, zones):
        """
        Publishes bounding box detections and active ROI zones to the MQTT broker.
        """
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

        topic = self.bbox_topic_template.format(camera_code=camera_code)
        payload = {
            "camera_code": camera_code,
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

    def stop(self):
        self.running = False
