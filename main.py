import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import time
import json
import threading

# Import module quản lý hệ thống
from core.system_manager import SystemManager

# Tải cấu hình từ config.json
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

def main():
    # 1. Tải cấu hình
    config = load_config()
    
    # 2. Cập nhật MQTT_CONFIG toàn cục từ file helpers (phục vụ cho logic nội bộ)
    from utils.helpers import MQTT_CONFIG
    MQTT_CONFIG.update({
        "broker": config["mqtt"]["broker"],
        "port": config["mqtt"]["port"],
        "username": config["mqtt"]["username"],
        "password": config["mqtt"]["password"],
        "company_id": config["mqtt"]["company_id"],
        "cameras_topic": f"smart_vms/cameras/company/{config['mqtt']['company_id']}",
        "ai_module": config["ai"]["ai_module"]
    })

    # 3. Khởi tạo System Manager
    manager = SystemManager(config)
    
    # 4. Chạy hệ thống
    manager.start()

    try:
        # Luồng chính (main thread) bây giờ chỉ ngủ và chờ tín hiệu kết thúc (Supervisor loop)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nNhận tín hiệu dừng từ người dùng (Ctrl+C)...")
    finally:
        # 5. Dừng hệ thống an toàn
        manager.stop()

if __name__ == "__main__":
    main()