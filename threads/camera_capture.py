import time
import threading
import queue
import numpy as np

def check_cuda_supported(ffmpeg_path):
    """
    Kiểm tra xem bản FFMPEG và GPU có hỗ trợ giải mã tăng tốc phần cứng CUDA hay không.
    Trả về True nếu hỗ trợ, False nếu không hoặc xảy ra lỗi/timeout.
    """
    import subprocess
    cmd = [
        ffmpeg_path,
        "-y",
        "-hwaccel", "cuda",
        "-f", "lavfi",
        "-i", "testsrc=duration=1:size=640x360:rate=1",
        "-f", "null",
        "-"
    ]
    try:
        # Chạy kiểm tra nhanh với timeout 1.5 giây
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.5
        )
        return (res.returncode == 0)
    except Exception:
        return False

class CameraCaptureThread(threading.Thread):
    """
    Luồng chạy ngầm để đọc khung hình từ luồng RTSP bằng FFMPEG subprocess.
    Giải mã H264/H265 bằng GPU CUDA (nếu có hỗ trợ) và tự động fallback sang CPU.
    Resize về 640x360 và chuyển đổi sang BGR24 thô trực tiếp qua stdout.
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
        import imageio_ffmpeg
        import subprocess

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        width, height = 640, 360
        frame_size = width * height * 3

        # Kiểm tra hỗ trợ CUDA một lần duy nhất tại mỗi lần chạy luồng
        use_cuda = check_cuda_supported(ffmpeg_path)
        if use_cuda:
            print(f"[{self.camera_code}] Phát hiện FFMPEG hỗ trợ CUDA. Sẽ giải mã bằng GPU.")
        else:
            print(f"[{self.camera_code}] FFMPEG không hỗ trợ CUDA hoặc driver không tương thích. Sẽ giải mã bằng CPU.")

        while self.running:
            print(f"[{self.camera_code}] Đang khởi tạo tiến trình FFMPEG...")

            # Xây dựng lệnh FFMPEG tương ứng
            if use_cuda:
                cmd = [
                    ffmpeg_path,
                    "-y",
                    "-rtsp_transport", "tcp",
                    "-timeout", "15000000",  # Timeout kết nối socket 15 giây
                    "-fflags", "nobuffer",
                    "-flags", "low_delay",
                    "-max_delay", "0",
                    "-hwaccel", "cuda",
                    "-i", self.rtsp_url,
                    "-vf", f"scale={width}:{height}",
                    "-pix_fmt", "bgr24",
                    "-f", "rawvideo",
                    "-"
                ]
            else:
                cmd = [
                    ffmpeg_path,
                    "-y",
                    "-rtsp_transport", "tcp",
                    "-timeout", "15000000",  # Timeout kết nối socket 15 giây
                    "-fflags", "nobuffer",
                    "-flags", "low_delay",
                    "-max_delay", "0",
                    "-i", self.rtsp_url,
                    "-vf", f"scale={width}:{height}",
                    "-pix_fmt", "bgr24",
                    "-f", "rawvideo",
                    "-"
                ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_size * 2
            )

            # Chờ kết nối và đọc khung hình đầu tiên
            connected = False
            try:
                # Đọc frame đầu tiên (chặn cho đến khi kết nối thành công hoặc lỗi/thoát)
                in_bytes = process.stdout.read(frame_size)
                if len(in_bytes) == frame_size:
                    connected = True
                    print(f"[{self.camera_code}] Kết nối và giải mã thành công bằng FFMPEG ( {'GPU/CUDA' if use_cuda else 'CPU'} ).")
                    self.reconnect_delay = 1.0  # Reset backoff
                    
                    frame = np.frombuffer(in_bytes, dtype=np.uint8).reshape((height, width, 3))
                    frame_data = {
                        "frame": frame.copy(),
                        "capture_time": time.time()
                    }
                    self._push_frame(frame_data)
                    from core.metrics import GLOBAL_METRICS
                    GLOBAL_METRICS.increment_capture(self.camera_code)
                else:
                    print(f"[{self.camera_code}] Không thể đọc đủ dữ liệu frame đầu tiên từ FFMPEG (nhận được {len(in_bytes)} bytes).")
                    if use_cuda:
                        print(f"[{self.camera_code}] Khởi tạo GPU/CUDA hoặc giải mã thất bại. Chuyển sang chế độ CPU fallback.")
                        use_cuda = False
            except Exception as e:
                print(f"[{self.camera_code}] Lỗi khi kết nối FFMPEG: {e}")
                if use_cuda:
                    print(f"[{self.camera_code}] Lỗi kết nối GPU/CUDA. Chuyển sang chế độ CPU fallback.")
                    use_cuda = False

            # Đọc tiếp các frame tiếp theo nếu kết nối thành công
            if connected:
                try:
                    from core.metrics import GLOBAL_METRICS
                    while self.running:
                        in_bytes = process.stdout.read(frame_size)
                        if len(in_bytes) != frame_size:
                            break

                        if self.frame_queue.empty():
                            frame = np.frombuffer(in_bytes, dtype=np.uint8).reshape((height, width, 3))
                            GLOBAL_METRICS.increment_capture(self.camera_code)

                            frame_data = {
                                "frame": frame.copy(),
                                "capture_time": time.time()
                            }
                            self._push_frame(frame_data)
                except Exception as e:
                    print(f"[{self.camera_code}] Lỗi trong vòng lặp đọc FFMPEG: {e}")

            # Giải phóng tiến trình
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()

            if self.running:
                print(f"[{self.camera_code}] Tiến trình FFMPEG dừng. Thử lại sau {self.reconnect_delay:.1f}s...")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    def stop(self):
        self.running = False
