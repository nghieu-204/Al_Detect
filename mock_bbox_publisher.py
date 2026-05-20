#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from typing import Any

import paho.mqtt.client as mqtt


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


def load_cameras(cfg: dict[str, Any], timeout: float = 5.0) -> list[dict[str, Any]]:
    done = threading.Event()
    cameras: list[dict[str, Any]] = []

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(cfg["cameras_topic"], qos=1)
        else:
            done.set()

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            for cam in payload.get("cameras") or []:
                if not isinstance(cam, dict):
                    continue
                modules = normalize_ai_modules(cam.get("ai_modules") or cam.get("aiModules"))
                status = str(cam.get("status") or "").upper()
                code = str(cam.get("code") or cam.get("name") or cam.get("id") or "").strip()
                rtsp = select_rtsp(cam, cfg["ai_module"])
                if code and rtsp and cfg["ai_module"] in modules and status == "ONLINE":
                    cameras.append({"id": cam.get("id"), "code": code, "name": cam.get("name"), "rtsp": rtsp})
        finally:
            done.set()

    client = new_client(cfg, f"mock_camera_loader_{int(time.time())}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(cfg["broker"], cfg["port"], 60)
    client.loop_start()
    done.wait(timeout)
    client.loop_stop()
    client.disconnect()
    return cameras


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


def load_polygons(cfg: dict[str, Any], timeout: float = 5.0) -> dict[str, list[dict[str, Any]]]:
    polygons: dict[str, list[dict[str, Any]]] = {}
    done = threading.Event()

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(topic_to_subscribe(cfg["polygons_topic"]), qos=1)
        else:
            done.set()

    def on_message(client, userdata, msg):
        camera_code = camera_code_from_zone_topic(msg.topic)
        if not camera_code:
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            zones = []
            for zone in payload.get("zones") or []:
                if not isinstance(zone, dict) or not zone.get("is_active", True):
                    continue
                modules = normalize_ai_modules(zone.get("ai_modules") or zone.get("aiModules"))
                if modules and cfg["ai_module"] not in modules:
                    continue
                points = zone_points(zone)
                if points:
                    zones.append({"zone_id": zone.get("id") or zone.get("zone_id"), "polygon": points})
            if zones:
                polygons[camera_code] = zones
        except Exception:
            pass

    client = new_client(cfg, f"mock_polygon_loader_{int(time.time())}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(cfg["broker"], cfg["port"], 60)
    client.loop_start()
    done.wait(timeout)
    client.loop_stop()
    client.disconnect()
    return polygons


def fake_bbox_message(camera_code: str, ai_module: str) -> dict[str, Any]:
    return {
        "camera_code": camera_code,
        "ai_module": ai_module,
        "ai_modules": [ai_module],
        "timestamp": time.time(),
        "detections": [
            {
                "id": "obj_mock_1",
                "cls": "car",
                "class": "car",
                "label": "car",
                "class_id": 0,
                "confidence": 0.95,
                "bbox": [0.35, 0.35, 0.65, 0.75],
                "color": "#00ff00",
            }
        ],
    }


def publish_fake_bbox(client: mqtt.Client, cfg: dict[str, Any], camera_code: str) -> None:
    topic = cfg["bbox_topic_template"].format(camera_code=camera_code)
    message = fake_bbox_message(camera_code, cfg["ai_module"])
    info = client.publish(topic, json.dumps(message, ensure_ascii=False), qos=0)
    info.wait_for_publish(timeout=2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-code", default="")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    cfg = MQTT_CONFIG
    cameras = load_cameras(cfg)
    polygons = load_polygons(cfg)

    if args.camera_code.strip():
        camera_codes = [args.camera_code.strip()]
    else:
        camera_codes = [camera["code"] for camera in cameras if camera.get("code")]

    if not camera_codes:
        raise SystemExit("No ONLINE camera found. Pass --camera-code to publish manually.")

    for camera in cameras:
        camera_code = camera.get("code")
        if camera_code not in camera_codes:
            continue
        camera_polygons = polygons.get(camera_code, [])
        print(
            f"camera name={camera.get('name') or camera_code} "
            f"rtsp={camera.get('rtsp')} "
            f"polygons={json.dumps(camera_polygons, ensure_ascii=False)}"
        )

    client = new_client(cfg, f"mock_bbox_publisher_{int(time.time())}")
    client.connect(cfg["broker"], cfg["port"], 60)
    client.loop_start()
    try:
        while True:
            for camera_code in camera_codes:
                publish_fake_bbox(client, cfg, camera_code)
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
