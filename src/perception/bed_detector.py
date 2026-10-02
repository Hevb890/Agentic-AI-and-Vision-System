import cv2
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from ultralytics import YOLO

class BedDetector:
    def __init__(self, roi_polygon: Optional[List[Tuple[int, int]]] = None):
        if roi_polygon is not None:
            self.bed_roi = np.array(roi_polygon, np.int32)
        else:
            self.bed_roi = None

    def auto_detect_bed(
            self,
            frame: np.ndarray,
            model_path: str = "yolov8n.pt",
            conf_thresh: float = 0.30,
            padding_ratio: float = 0.10
    ) -> np.ndarray:
        h, w, _ = frame.shape
        model = YOLO(model_path)

        results = model.predict(source=frame, classes = [59], conf=conf_thresh, verbose = False)

        if len(results[0].boxes) > 0:
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

            pw = int((x2 - x1) * padding_ratio)
            ph = int((y2 - y1) * padding_ratio)

            x1 = max(0, x1 - pw)
            y1 = max(0, y1 - ph)
            x2 = min(w, x2 + pw)
            y2 = min(h, y2 + ph)
        else:
            x1, y1 = int(w * 0.20), int(h * 0.25)
            x2, y2 = int(w * 0.80), int(h * 0.85)

        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        self.bed_roi = np.array(polygon, np.int32)
        return self.bed_roi

    def check_overlap(self, bbox: List[float], keypoints: np.ndarray, conf_thresh: float = 0.35) -> Dict[str, Any]:
        if self.bed_roi is None:
            raise ValueError("[BedDetector ERROR] bed_roi is not initialized! Call auto_detect_bed(frame) first.")
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
        if self.bed_roi is None:
            return frame
        cv2.polylines(frame, [self.bed_roi], isClosed = True, color=(0,255,255), thickness=2)

        text_x = self.bed_roi[0][0]
        text_y = max(10, self.bed_roi[0][1] - 10)
        cv2.putText(frame, "BED ROI", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame