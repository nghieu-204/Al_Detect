import json
import time
import threading
import queue
import torch
from ultralytics import YOLO

from threads.camera_capture import CameraCaptureThread
from threads.batch_inference import BatchInferenceThread
from threads.tracking_thread import TrackingThread
from utils.helpers import (
    MQTT_CONFIG,
    new_client,
    normalize_ai_modules,
    select_rtsp,
    camera_code_from_zone_topic,
    zone_points
)

class SystemManager:
    """
    Quản lý trạng thái hệ thống: kết nối MQTT, cấu hình camera, hàng đợi
    và điều phối các luồng con (Capture, Tracking, Batch Inference).
    """
    def __init__(self, config):
        self.config = config
        self.data_lock = threading.Lock()
        
        # Các biến toàn cục cũ giờ được gom vào đối tượng (instance)
        self.active_cameras = {}
        self.capture_threads = {}
        self.camera_zones = {}
        self.tracking_queues = {}
        self.tracking_threads = {}
        self.class_names_ref = {}
        
        print("Initializing AI Worker service...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"AI Device detected: {self.device.upper()}")
        
        print(f"Loading YOLO model: {self.config['ai']['model_path']} on device: {self.device}")
        self.yolo_model = YOLO(self.config["ai"]["model_path"])
        self.class_names_ref.update(self.yolo_model.names)
        
        self.batch_inference_thread = None
        
        # Khởi tạo MQTT client
        self.mqtt_client = new_client(MQTT_CONFIG, f"roi_ai_service_{int(time.time())}")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker.")
            client.subscribe(MQTT_CONFIG["cameras_topic"], qos=1)
            print(f"Subscribed to cameras list topic: {MQTT_CONFIG['cameras_topic']}")
        else:
            print(f"MQTT Connection failed with return code: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode('utf-8'))
            
            if topic == MQTT_CONFIG["cameras_topic"]:
                self.update_cameras(payload.get("cameras", []))
                
            elif "zones" in topic:
                camera_code = camera_code_from_zone_topic(topic)
                if not camera_code:
                    return
                    
                print(f"[{camera_code}] Received zones update on topic: {topic}")
                
                # Gửi thông tin vùng mặc định nếu không có cấu hình zone nào
                zones_list = payload.get("zones") or []
                if not zones_list:
                    print(f"[{camera_code}] No zones found. Publishing default mock ROI to MQTT broker...")
                    default_payload = {
                        "camera_code": camera_code,
                        "zones": [
                            {
                                "id": f"default_mock_roi_{camera_code}",
                                "is_active": True,
                                "ai_modules": [MQTT_CONFIG["ai_module"]],
                                "points": [
                                    {"x": 0.2, "y": 0.2},
                                    {"x": 0.8, "y": 0.2},
                                    {"x": 0.8, "y": 0.8},
                                    {"x": 0.2, "y": 0.8}
                                ]
                            }
                        ],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    client.publish(topic, json.dumps(default_payload), qos=1, retain=True)
                    return
                
                zones = []
                for zone in zones_list:
                    if not isinstance(zone, dict) or not zone.get("is_active", True):
                        continue
                    modules = normalize_ai_modules(zone.get("ai_modules") or zone.get("aiModules"))
                    if modules and MQTT_CONFIG["ai_module"] not in modules:
                        continue
                    points = zone_points(zone)
                    if points:
                        zones.append({
                            "zone_id": zone.get("id") or zone.get("zone_id"),
                            "polygon": points,
                            "roi_pts": None,
                            "frame_size": None
                        })
                
                with self.data_lock:
                    self.camera_zones[camera_code] = zones
                    print(f"[{camera_code}] Updated active ROI zones list ({len(zones)} zones)")
                    
        except Exception as e:
            print(f"Error handling message on {msg.topic}: {e}")

    def update_cameras(self, new_camera_list):
        with self.data_lock:
            new_online_codes = set()
            parsed_cameras = {}
            
            for cam in new_camera_list:
                modules = normalize_ai_modules(cam.get("ai_modules") or cam.get("aiModules"))
                status = str(cam.get("status") or "").upper()
                code = str(cam.get("code") or cam.get("name") or cam.get("id") or "").strip()
                rtsp = select_rtsp(cam, MQTT_CONFIG["ai_module"])
                
                active_filter = self.config.get("active_cameras")
                if active_filter and code not in active_filter:
                    continue
                    
                if code and rtsp and MQTT_CONFIG["ai_module"] in modules and status == "ONLINE":
                    new_online_codes.add(code)
                    parsed_cameras[code] = {
                        "code": code,
                        "name": cam.get("name") or code,
                        "rtsp": rtsp
                    }
            
            # Xoá bỏ camera không còn trực tuyến hoặc không nằm trong danh sách
            for code in list(self.capture_threads.keys()):
                if code not in new_online_codes:
                    print(f"Stopping capture thread for camera: {code}")
                    self.capture_threads[code].stop()
                    self.capture_threads[code].join(timeout=1.0)
                    del self.capture_threads[code]
                    
                    if code in self.tracking_threads:
                        print(f"Stopping tracking thread for camera: {code}")
                        self.tracking_threads[code].stop()
                        self.tracking_threads[code].join(timeout=1.0)
                        del self.tracking_threads[code]
                    if code in self.tracking_queues:
                        del self.tracking_queues[code]
                    
                    if code in self.camera_zones:
                        del self.camera_zones[code]
                    
                    topic = MQTT_CONFIG["polygons_topic"].format(camera_code=code)
                    self.mqtt_client.unsubscribe(topic)
                        
            # Bật luồng capture, tracking cho camera mới
            for code in new_online_codes:
                if code not in self.capture_threads:
                    cam_info = parsed_cameras[code]
                    print(f"Starting capture thread for camera: {code} ({cam_info['rtsp']})")
                    
                    topic = MQTT_CONFIG["polygons_topic"].format(camera_code=code)
                    self.mqtt_client.subscribe(topic, qos=1)
                    print(f"Subscribed to zones topic: {topic}")
                    
                    cap_thread = CameraCaptureThread(
                        camera_code=code,
                        rtsp_url=cam_info["rtsp"]
                    )
                    cap_thread.start()
                    self.capture_threads[code] = cap_thread
                    
                    # Cấu hình hàng đợi tracking (maxsize=1 để giảm latency giống như yêu cầu)
                    t_queue = queue.Queue(maxsize=1)
                    self.tracking_queues[code] = t_queue
                    
                    track_thread = TrackingThread(
                        camera_code=code,
                        tracking_queue=t_queue,
                        mqtt_client=self.mqtt_client,
                        bbox_topic_template=MQTT_CONFIG["bbox_topic_template"],
                        ai_module=MQTT_CONFIG["ai_module"],
                        class_names=self.class_names_ref
                    )
                    track_thread.start()
                    self.tracking_threads[code] = track_thread
                    
            self.active_cameras = parsed_cameras

    def start(self):
        # Kết nối MQTT broker
        print(f"Connecting to broker: {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}")
        self.mqtt_client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], 60)
        self.mqtt_client.loop_start()

        # Bật luồng Batch Inference
        self.batch_inference_thread = BatchInferenceThread(
            capture_threads=self.capture_threads,
            camera_zones=self.camera_zones,
            tracking_queues=self.tracking_queues,
            yolo_model=self.yolo_model,
            device=self.device,
            confidence=self.config["ai"]["confidence"],
            data_lock=self.data_lock
        )
        self.batch_inference_thread.start()

        # Bật luồng ghi log FPS
        self.metrics_running = True
        self.metrics_thread = threading.Thread(target=self._log_metrics_loop, daemon=True, name="MetricsLogger")
        self.metrics_thread.start()

    def _log_metrics_loop(self):
        from core.metrics import GLOBAL_METRICS
        while self.metrics_running:
            time.sleep(1.0)
            GLOBAL_METRICS.reset_and_print()

    def stop(self):
        print("Stopping AI Worker service...")
        self.metrics_running = False
        if getattr(self, 'metrics_thread', None) is not None:
            self.metrics_thread.join(timeout=1.0)
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        
        if self.batch_inference_thread is not None:
            print("Stopping batch inference thread...")
            self.batch_inference_thread.stop()
            self.batch_inference_thread.join(timeout=1.0)
            
        with self.data_lock:
            for code, thread in list(self.tracking_threads.items()):
                print(f"Stopping tracking thread: {code}")
                thread.stop()
            for thread in self.tracking_threads.values():
                thread.join(timeout=1.0)
            self.tracking_threads.clear()
            self.tracking_queues.clear()

            for code, thread in list(self.capture_threads.items()):
                print(f"Stopping capture thread: {code}")
                thread.stop()
            for thread in self.capture_threads.values():
                thread.join(timeout=1.0)
            self.capture_threads.clear()
            
        print("Shutdown complete.")
