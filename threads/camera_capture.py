import time
import threading
import queue
import cv2
import numpy as np

class CameraCaptureThread(threading.Thread):
    """
    Luồng chạy ngầm để đọc khung hình từ luồng RTSP.
    Sử dụng OpenCV (cv2.VideoCapture) để decode frame bằng CPU.
    Lưu trữ khung hình mới nhất vào hàng đợi Queue(maxsize=1) an toàn đa luồng.
    """
    def __init__(self, camera_code: str, rtsp_url: str):
        super().__init__(name=f"CaptureThread_{camera_code}", daemon=True)
        self.camera_code = camera_code
        self.rtsp_url = rtsp_url
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Các tham số thử lại kết nối khi mất kết nối (Exponential Backoff)
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0

    def run(self):
        print(f"[{self.camera_code}] Capture thread started.")
        self._run_rtsp()

    def _push_frame(self, frame):
        """
        Đẩy khung hình mới vào hàng đợi maxsize=1 (thuật toán Empty-Put).
        Nếu đầy, loại bỏ khung hình cũ nhất để luôn giữ frame mới nhất.
        """
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _run_rtsp(self):
        import os
        # Ép buộc OpenCV dùng TCP thay vì UDP để tránh mất gói tin
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        while self.running:
            print(f"[{self.camera_code}] Đang kết nối tới RTSP: {self.rtsp_url}")

            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm latency

            if not cap.isOpened():
                print(f"[{self.camera_code}] Không thể kết nối RTSP. Thử lại sau {self.reconnect_delay:.1f}s...")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                continue

            print(f"[{self.camera_code}] Kết nối thành công (OpenCV CPU Decode).")
            self.reconnect_delay = 1.0  # Đặt lại khi kết nối thành công

            try:
                while self.running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"[{self.camera_code}] Mất kết nối RTSP.")
                        break

                    # Resize về 640x320
                    frame = cv2.resize(frame, (640, 320))

                    from core.metrics import GLOBAL_METRICS
                    GLOBAL_METRICS.increment_capture(self.camera_code)

                    frame_data = {
                        "frame": frame,
                        "capture_time": time.time()
                    }
                    self._push_frame(frame_data)

            except Exception as e:
                print(f"[{self.camera_code}] Gặp lỗi khi đọc luồng video: {e}")
            finally:
                cap.release()

            if self.running:
                time.sleep(2.0)

    def stop(self):
        self.running = False
