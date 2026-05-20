import cv2
import numpy as np
from typing import List, Tuple, Union, Dict, Any

def parse_point(p: Any) -> Tuple[float, float]:
    """
    Parses a point representation into a float tuple (x, y).
    Supports dict like {'x': x, 'y': y} and iterable like [x, y].
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
    Converts a normalized ROI polygon (list of relative [x, y] coordinates in [0.0, 1.0])
    into an absolute pixel coordinate array for OpenCV functions.
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
    Checks if a central point (cx, cy) is within the given ROI polygon coordinates.
    """
    if roi_pts is None or len(roi_pts) == 0:
        return False
    result = cv2.pointPolygonTest(roi_pts, (cx, cy), False)
    return result >= 0
