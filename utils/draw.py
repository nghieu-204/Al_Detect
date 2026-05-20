import cv2
import numpy as np
from typing import List, Dict, Any, Optional

def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    roi_polygons: Optional[List[np.ndarray]] = None,
    color: tuple = (0, 255, 0),
    roi_color: tuple = (255, 0, 255)
) -> np.ndarray:
    """
    Annotates a frame with detection bounding boxes and ROI polygons.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw ROIs
    if roi_polygons:
        for poly in roi_polygons:
            if poly is not None and len(poly) > 0:
                cv2.polylines(annotated, [poly], isClosed=True, color=roi_color, thickness=2)

    # Draw BBoxes
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        
        # Denormalize coordinates
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        # Draw bbox rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label & confidence
        label = det.get("label") or det.get("cls") or f"obj_{det.get('class_id', '')}"
        conf = det.get("confidence", 0.0)
        text = f"{label} {conf:.2f}"
        
        # Put text
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
    return annotated
