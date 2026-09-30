import cv2
import numpy as np
from typing import List, Tuple, Any, Dict

class BedDetector:
    def __init__(self, roi_polygon: List[Tuple[int, int]]):
        self.bed_roi = np.array(roi_polygon, np.int32)


    def check_overlap(self, bbox: List[float], keypoints: np.ndarray, conf_thresh: float = 0.35) -> Dict[str, Any]:
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        center_in_bed = cv2.pointPolygonTest(self.bed_roi, (cx, cy), False) >= 0

        valid_kpts = 0
        kpt_inside = 0

        for kp in keypoints:
            if kp[2] >= conf_thresh:
                valid_kpts+=1
                if cv2.pointPolygonTest(self.bed_roi, (int(kp[0]), int(kp[1])), False) >= 0:
                    kpt_inside+=1

        overlap_ratio = (kpt_inside / valid_kpts) if valid_kpts > 0 else 0.0
        return {
            "center_in_bed": center_in_bed,
            "bed_overlap_ratio": overlap_ratio
        }

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        cv2.polylines(frame, [self.bed_roi], isClosed = True, color=(0,255,255), thickness=2)
        cv2.putText(frame, "BED ROI", (self.bed_roi[0][0], self.bed_roi[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame