import time
import threading
import queue
import cv2
import numpy as np

class CameraCaptureThread(threading.Thread):
    """
    Background thread to read frames from RTSP stream or generate mock frames.
    Buffers the latest frame into a thread-safe Queue(maxsize=1).
    """
    def __init__(self, camera_code: str, rtsp_url: str, is_mock: bool = False):
        super().__init__(name=f"CaptureThread_{camera_code}", daemon=True)
        self.camera_code = camera_code
        self.rtsp_url = rtsp_url
        self.is_mock = is_mock
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Connection retry parameters
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0

    def run(self):
        print(f"[{self.camera_code}] Capture thread started.")
        if self.is_mock:
            self._run_mock()
        else:
            self._run_rtsp()

    def _push_frame(self, frame):
        """
        Pushes a new frame to the queue. If full, discards the old frame
        to keep the queue size at 1 and containing only the most recent frame.
        """
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _run_mock(self):
        # Generate synthetic frame with moving shapes at ~15 FPS
        while self.running:
            start_time = time.time()
            
            # Create a base gray-blue frame
            frame = np.zeros((360, 640, 3), dtype=np.uint8) + 40
            frame[:, :, 0] += 15 # slightly blue hue
            
            # Draw standard reference grid lines
            for x in range(0, 640, 80):
                cv2.line(frame, (x, 0), (x, 360), (60, 60, 60), 1)
            for y in range(0, 360, 60):
                cv2.line(frame, (0, y), (640, y), (60, 60, 60), 1)
                
            # Draw camera name and timestamp
            cv2.putText(
                frame,
                f"CAMERA FEED: {self.camera_code.upper()} (MOCK)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            
            # Draw a moving shape (circle + rectangle representing a person)
            t = time.time()
            cx = int(320 + 200 * np.cos(t * 0.8))
            cy = int(180 + 100 * np.sin(t * 1.1))
            
            # Draw head
            cv2.circle(frame, (cx, cy - 25), 12, (220, 220, 220), -1)
            # Draw body
            cv2.rectangle(frame, (cx - 15, cy - 10), (cx + 15, cy + 30), (220, 220, 220), -1)
            
            cv2.putText(
                frame,
                "Simulated Target",
                (cx - 50, cy - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            self._push_frame(frame)
                
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0.005, (1.0 / 15.0) - elapsed)
            time.sleep(sleep_time)

    def _run_rtsp(self):
        while self.running:
            print(f"[{self.camera_code}] Connecting to RTSP: {self.rtsp_url}")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Set buffer size to 1 to read only latest frame
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                print(f"[{self.camera_code}] Failed to open RTSP. Retrying in {self.reconnect_delay:.1f}s...")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                continue
            
            print(f"[{self.camera_code}] RTSP connection successful.")
            self.reconnect_delay = 1.0 # Reset delay on success
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.camera_code}] RTSP stream read error.")
                    break
                
                self._push_frame(frame)
                    
            cap.release()
            if self.running:
                time.sleep(2.0)

    def stop(self):
        self.running = False
