import os
import sys
import time
import json
import threading
import torch
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# Import local helpers
from threads.camera_capture import CameraCaptureThread
from threads.batch_inference import BatchInferenceThread
from mock_bbox_publisher import (
    MQTT_CONFIG,
    new_client,
    normalize_ai_modules,
    select_rtsp,
    topic_to_subscribe,
    camera_code_from_zone_topic,
    zone_points
)

# Load configuration from config.json
CONFIG_PATH = "config.json"
def load_config():
    default_config = {
        "mqtt": {
            "broker": "192.168.1.250",
            "port": 1883,
            "username": "atin",
            "password": "team1@123#",
            "company_id": 10
        },
        "ai": {
            "model_path": "models/yolov8n.pt",
            "confidence": 0.4,
            "ai_module": "FACE"
        },
        "active_cameras": ["1", "2", "Demo01"]
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                for key in default_config:
                    if key in config:
                        if isinstance(default_config[key], dict) and isinstance(config[key], dict):
                            default_config[key].update(config[key])
                        else:
                            default_config[key] = config[key]
        except Exception as e:
            print(f"Error loading config.json: {e}. Using defaults.")
    return default_config

CONFIG = load_config()

# Update MQTT_CONFIG with config.json settings
MQTT_CONFIG.update({
    "broker": CONFIG["mqtt"]["broker"],
    "port": CONFIG["mqtt"]["port"],
    "username": CONFIG["mqtt"]["username"],
    "password": CONFIG["mqtt"]["password"],
    "company_id": CONFIG["mqtt"]["company_id"],
    "cameras_topic": f"smart_vms/cameras/company/{CONFIG['mqtt']['company_id']}",
    "ai_module": CONFIG["ai"]["ai_module"]
})

# Global variables protected by a lock
data_lock = threading.Lock()
active_cameras = {}  # camera_code -> camera_details
capture_threads = {}  # camera_code -> CameraCaptureThread
camera_zones = {}    # camera_code -> list of zones

# Global YOLO model, device and Batch Inference thread reference
yolo_model = None
device = "cpu"
batch_inference_thread = None

def update_cameras(new_camera_list, mqtt_client):
    """
    Synchronizes running capture threads with the camera list received from the MQTT broker.
    The single centralized BatchInferenceThread automatically polls all active capture queues.
    """
    global active_cameras, capture_threads
    
    with data_lock:
        new_online_codes = set()
        parsed_cameras = {}
        
        # Parse cameras list from broker
        for cam in new_camera_list:
            modules = normalize_ai_modules(cam.get("ai_modules") or cam.get("aiModules"))
            status = str(cam.get("status") or "").upper()
            code = str(cam.get("code") or cam.get("name") or cam.get("id") or "").strip()
            rtsp = select_rtsp(cam, MQTT_CONFIG["ai_module"])
            
            # Filter active cameras based on config.json
            active_filter = CONFIG.get("active_cameras")
            if active_filter and code not in active_filter:
                continue
                
            if code and rtsp and MQTT_CONFIG["ai_module"] in modules and status == "ONLINE":
                new_online_codes.add(code)
                parsed_cameras[code] = {
                    "code": code,
                    "name": cam.get("name") or code,
                    "rtsp": rtsp,
                    "is_mock": False
                }
        
        # Stop capture threads for cameras that are no longer online/configured
        for code in list(capture_threads.keys()):
            if code not in new_online_codes:
                print(f"Stopping capture thread for camera: {code}")
                capture_threads[code].stop()
                capture_threads[code].join(timeout=1.0)
                del capture_threads[code]
                
                if code in camera_zones:
                    del camera_zones[code]
                # Unsubscribe zones topic
                topic = MQTT_CONFIG["polygons_topic"].format(camera_code=code)
                mqtt_client.unsubscribe(topic)
                    
        # Start capture threads for newly active cameras
        for code in new_online_codes:
            if code not in capture_threads:
                cam_info = parsed_cameras[code]
                print(f"Starting capture thread for camera: {code} ({cam_info['rtsp']})")
                
                # Subscribe to polygon/zones configuration updates
                topic = MQTT_CONFIG["polygons_topic"].format(camera_code=code)
                mqtt_client.subscribe(topic, qos=1)
                print(f"Subscribed to zones topic: {topic}")
                
                # Start capture thread
                cap_thread = CameraCaptureThread(
                    camera_code=code,
                    rtsp_url=cam_info["rtsp"],
                    is_mock=cam_info["is_mock"]
                )
                cap_thread.start()
                capture_threads[code] = cap_thread
                
        active_cameras = parsed_cameras


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker.")
        # Subscribe to camera list
        client.subscribe(MQTT_CONFIG["cameras_topic"], qos=1)
        print(f"Subscribed to cameras list topic: {MQTT_CONFIG['cameras_topic']}")
    else:
        print(f"MQTT Connection failed with return code: {rc}")


def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode('utf-8'))
        
        if topic == MQTT_CONFIG["cameras_topic"]:
            update_cameras(payload.get("cameras", []), client)
            
        elif "zones" in topic:
            camera_code = camera_code_from_zone_topic(topic)
            if not camera_code:
                return
                
            print(f"[{camera_code}] Received zones update on topic: {topic}")
            print(f"[{camera_code}] Raw payload: {json.dumps(payload, ensure_ascii=False)}")
            
            # If no zones are configured, publish the default mock ROI to the MQTT broker
            # so the Web VMS receives it and draws the polygon on the canvas.
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
            
            # Parse polygon configurations
            zones = []
            for zone in zones_list:
                if not isinstance(zone, dict) or not zone.get("is_active", True):
                    continue
                modules = normalize_ai_modules(zone.get("ai_modules") or zone.get("aiModules"))
                print(f"[{camera_code}] Parsing zone {zone.get('id')}. Modules: {modules}")
                if modules and MQTT_CONFIG["ai_module"] not in modules:
                    print(f"[{camera_code}] Zone skipped: Configured AI module '{MQTT_CONFIG['ai_module']}' not in zone modules {modules}")
                    continue
                points = zone_points(zone)
                if points:
                    zones.append({
                        "zone_id": zone.get("id") or zone.get("zone_id"),
                        "polygon": points,  # normalized ratios [0.0, 1.0]
                        "roi_pts": None,    # built on frame dimensions lazily
                        "frame_size": None
                    })
            
            with data_lock:
                camera_zones[camera_code] = zones
                print(f"[{camera_code}] Updated active ROI zones list ({len(zones)} zones)")
                
    except Exception as e:
        print(f"Error handling message on {msg.topic}: {e}")


def main():
    global yolo_model, device, batch_inference_thread
    print("Initializing AI Worker service...")
    
    # Initialize YOLO model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(CONFIG["ai"]["model_path"])
    print(f"YOLOv8 initialized ({CONFIG['ai']['model_path']}). Device: {device.upper()}")
    
    # Initialize MQTT connection
    mqtt_client = new_client(MQTT_CONFIG, f"roi_ai_service_{int(time.time())}")
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    print(f"Connecting to broker: {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}")
    mqtt_client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], 60)
    mqtt_client.loop_start()

    # Start the centralized Batch Inference thread
    batch_inference_thread = BatchInferenceThread(
        capture_threads=capture_threads,
        camera_zones=camera_zones,
        mqtt_client=mqtt_client,
        yolo_model=yolo_model,
        device=device,
        confidence=CONFIG["ai"]["confidence"],
        ai_module=MQTT_CONFIG["ai_module"],
        bbox_topic_template=MQTT_CONFIG["bbox_topic_template"],
        data_lock=data_lock
    )
    batch_inference_thread.start()

    try:
        # The main thread now simply sleeps and acts as the supervisor
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("Stopping AI Worker service...")
    finally:
        # Shutdown client and threads
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        
        # Stop centralized batch inference first
        if batch_inference_thread is not None:
            print("Stopping batch inference thread...")
            batch_inference_thread.stop()
            batch_inference_thread.join(timeout=1.0)
            
        with data_lock:
            # Stop all capture threads
            for code, thread in list(capture_threads.items()):
                print(f"Stopping capture thread: {code}")
                thread.stop()
            for thread in capture_threads.values():
                thread.join(timeout=1.0)
            capture_threads.clear()
            
        print("Shutdown complete.")

if __name__ == "__main__":
    main()