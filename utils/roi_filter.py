import cv2
import numpy as np
from typing import List, Tuple, Any

def parse_point(p: Any) -> Tuple[float, float]:
    """
    Chuyển đổi điểm biểu diễn thành một tuple số thực (x, y).
    Hỗ trợ dạng dict như {'x': x, 'y': y} hoặc dạng danh sách như [x, y].
    """
    if isinstance(p, dict):
        x = p.get("x", p.get("X"))
        y = p.get("y", p.get("Y"))
        if x is not None and y is not None:
            return float(x), float(y)
    elif isinstance(p, (list, tuple)) and len(p) >= 2:
        return float(p[0]), float(p[1])
    raise ValueError(f"Invalid point structure: {p}")

def build_roi_polygon(width: int, height: int, polygon: List[Any]) -> np.ndarray:
    """
    Chuyển đổi đa giác ROI chuẩn hóa (danh sách các tọa độ tương đối [x, y] trong khoảng [0.0, 1.0])
    thành mảng tọa độ pixel tuyệt đối để sử dụng với các hàm OpenCV.
    """
    pts = []
    for p in polygon:
        try:
            rx, ry = parse_point(p)
            x = int(rx * width)
            y = int(ry * height)
            pts.append([x, y])
        except Exception:
            continue
    return np.array(pts, dtype=np.int32)

def inside_roi(cx: int, cy: int, roi_pts: np.ndarray) -> bool:
    """
    Kiểm tra xem điểm trung tâm (cx, cy) có nằm trong đa giác ROI được chỉ định hay không.
    """
    if roi_pts is None or len(roi_pts) == 0:
        return False
    result = cv2.pointPolygonTest(roi_pts, (cx, cy), False)
    return result >= 0
