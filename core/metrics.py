import threading
import time

class PerformanceMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.capture_counts = {}
        self.tracking_counts = {}
        self.inference_count = 0
        self.latency_sum = {}
        self.latency_counts = {}

    def increment_capture(self, code):
        with self.lock:
            self.capture_counts[code] = self.capture_counts.get(code, 0) + 1

    def increment_inference(self, num_frames):
        with self.lock:
            self.inference_count += num_frames

    def add_tracking_latency(self, code, latency):
        with self.lock:
            self.tracking_counts[code] = self.tracking_counts.get(code, 0) + 1
            self.latency_sum[code] = self.latency_sum.get(code, 0) + latency
            self.latency_counts[code] = self.latency_counts.get(code, 0) + 1

    def reset_and_print(self):
        with self.lock:
            cap_str = " ".join([f"{k}({v})" for k, v in self.capture_counts.items()]) if self.capture_counts else "None"
            ai_str = str(self.inference_count)
            trk_str = " ".join([f"{k}({v})" for k, v in self.tracking_counts.items()]) if self.tracking_counts else "None"
            
            lat_str_list = []
            for k in self.latency_counts.keys():
                avg_latency = self.latency_sum[k] / self.latency_counts[k]
                lat_str_list.append(f"{k}({avg_latency * 1000:.0f}ms)")
            lat_str = " ".join(lat_str_list) if lat_str_list else "None"

            # Reset
            self.capture_counts.clear()
            self.tracking_counts.clear()
            self.inference_count = 0
            self.latency_sum.clear()
            self.latency_counts.clear()
            
        print(f"[METRICS] FPS -> Capture: {cap_str} | AI: {ai_str} | Tracking: {trk_str} | Latency: {lat_str}")

GLOBAL_METRICS = PerformanceMetrics()
