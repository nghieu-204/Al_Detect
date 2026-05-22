import time
import threading
import queue
import cv2
import numpy as np

class CameraCaptureThread(threading.Thread):
    """
    Luồng chạy ngầm để đọc khung hình từ luồng RTSP.
    Lưu trữ khung hình mới nhất vào hàng đợi Queue(maxsize=1) an toàn đa luồng.
    """
    def __init__(self, camera_code: str, rtsp_url: str):
        super().__init__(name=f"CaptureThread_{camera_code}", daemon=True)
        self.camera_code = camera_code
        self.rtsp_url = rtsp_url
        self.running = True
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Các tham số thử lại kết nối khi mất kết nối
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0

    def run(self):
        print(f"[{self.camera_code}] Capture thread started.")
        self._run_rtsp()

    def _push_frame(self, frame):
        """
        Đẩy khung hình mới vào hàng đợi maxsize=2.
        Nếu đầy, loại bỏ khung hình cũ nhất để luôn giữ frame mới nhất (empty-put).
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
        import imageio_ffmpeg
        import subprocess

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        while self.running:
            print(f"[{self.camera_code}] Đang kết nối tới RTSP (Probe): {self.rtsp_url}")
            
            # Sử dụng OpenCV để dò thông số chiều rộng và chiều cao của khung hình
            probe_cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            ret, first_frame = probe_cap.read()
            if not ret or first_frame is None:
                print(f"[{self.camera_code}] Không thể dò độ phân giải RTSP. Thử lại sau {self.reconnect_delay:.1f} giây...")
                probe_cap.release()
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                continue
                
            height, width = first_frame.shape[:2]
            probe_cap.release()
            print(f"[{self.camera_code}] Đã dò được độ phân giải gốc: {width}x{height}. Đặt độ phân giải đầu ra từ GPU là 640x320.")
            self.reconnect_delay = 1.0  # Đặt lại độ trễ kết nối khi thành công
            
            # Khung hình từ FFmpeg sẽ luôn là 640x320 do lệnh resize
            width, height = 640, 320
            
            # Cấu hình các lệnh chạy FFmpeg
            command_cuda = [
                ffmpeg_path,
                "-rtsp_transport", "tcp",         # Ép buộc dùng TCP
                "-fflags", "nobuffer",            # Tắt bộ đệm đầu vào
                "-flags", "low_delay",            # Chế độ trễ thấp
                "-probesize", "32",               # Giảm kích thước dữ liệu phân tích luồng xuống tối thiểu
                "-analyzeduration", "0",          # Tắt thời gian phân tích định dạng
                "-threads", "1",                  # Ép giải mã tuần tự để triệt tiêu bộ đệm đa luồng
                "-hwaccel", "cuda",               # Sử dụng tăng tốc phần cứng GPU CUDA
                "-hwaccel_output_format", "cuda", # Giữ giải mã trên GPU
                "-i", self.rtsp_url,              # Luồng RTSP đầu vào
                "-vf", "scale_cuda=640:320,hwdownload,format=nv12,format=bgr24", # Resize trên GPU
                "-f", "image2pipe",               # Định dạng đầu ra
                "-vcodec", "rawvideo",            # Giải mã video thô
                "-pix_fmt", "bgr24",              # Định dạng pixel BGR cho OpenCV/YOLO
                "-"                               # Xuất ra stdout
            ]
            
            # Khởi chạy giải mã GPU CUDA
            process = subprocess.Popen(command_cuda, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            if process.poll() is not None:
                print(f"[{self.camera_code}] LỖI: GPU CUDA không hỗ trợ giải mã hoặc kết nối thất bại. Thử lại sau...")
                try:
                    process.terminate()
                except Exception:
                    pass
                time.sleep(2.0)
                continue
            
            print(f"[{self.camera_code}] Kết nối thành công và sử dụng tăng tốc phần cứng GPU CUDA.")
            frame_size = width * height * 3
            try:
                while self.running:
                    raw_frame = process.stdout.read(frame_size)
                    if len(raw_frame) != frame_size:
                        print(f"[{self.camera_code}] Lỗi đọc luồng video từ FFmpeg (RTSP disconnect).")
                        break
                    
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                    
                    # Đóng gói dữ liệu và tính capture_time
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
                # Giải phóng tiến trình FFmpeg
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
            if self.running:
                time.sleep(2.0)

    def stop(self):
        self.running = False
