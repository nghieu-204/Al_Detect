import json
import re
import time
import paho.mqtt.client as mqtt
from typing import Any

MQTT_CONFIG = {
    "broker": "192.168.1.250",
    "port": 1883,
    "username": "atin",
    "password": "team1@123#",
    "company_id": 10,
    "ai_module": "FACE",
    "cameras_topic": "smart_vms/cameras/company/10",
    "polygons_topic": "smart_vms/cameras/{camera_code}/zones",
    "bbox_topic_template": "smart_vms/ai/bbox/{camera_code}",
}

def new_client(cfg: dict[str, Any], client_id: str) -> mqtt.Client:
    client = mqtt.Client(client_id=client_id, clean_session=True)
    if cfg["username"]:
        client.username_pw_set(cfg["username"], cfg["password"])
    return client

def normalize_ai_modules(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                items = json.loads(value)
            except Exception:
                items = [value]
        else:
            items = [value]
    elif value is None:
        items = []
    else:
        items = [value]
    return [str(item).strip().upper() for item in items if str(item).strip()]

def parse_json_value(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except Exception:
                return default
    return value if value is not None else default

def select_rtsp(cam: dict[str, Any], ai_module: str) -> str:
    restream_urls = parse_json_value(cam.get("restream_urls") or cam.get("restreamUrls"), {})
    if isinstance(restream_urls, dict):
        for key in (ai_module, ai_module.lower(), ai_module.upper()):
            value = restream_urls.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(restream_urls, list):
        for value in restream_urls:
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = cam.get("stream_url") or cam.get("streamUrl") or cam.get("rtsp") or cam.get("url") or cam.get("link")
    return value.strip() if isinstance(value, str) else ""

def topic_to_subscribe(topic_template: str) -> str:
    return re.sub(r"\{[^}]+\}", "+", topic_template)

def camera_code_from_zone_topic(topic: str) -> str | None:
    parts = topic.split("/")
    if len(parts) >= 2 and parts[-1] == "zones":
        return parts[-2]
    return None

def point(value: Any) -> list[float] | None:
    try:
        if isinstance(value, dict):
            x = value.get("x", value.get("X"))
            y = value.get("y", value.get("Y"))
            if x is not None and y is not None:
                return [float(x), float(y)]
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    except Exception:
        return None
    return None

def zone_points(zone: dict[str, Any]) -> list[list[float]] | None:
    for key in ("points", "polygon", "active_area", "area"):
        raw = zone.get(key)
        if isinstance(raw, list):
            points = [p for p in (point(item) for item in raw) if p is not None]
            if len(points) >= 3:
                return points
        if isinstance(raw, dict):
            for sub_key in ("points", "active_area", "area"):
                sub = raw.get(sub_key)
                if isinstance(sub, list):
                    points = [p for p in (point(item) for item in sub) if p is not None]
                    if len(points) >= 3:
                        return points
    return None
