import cv2
import numpy as np
import torch
from typing import List, Optional, Dict, Tuple, Any
from ultralytics import YOLO

from .bed_detector import BedDetector
from .tracker import PersonTracker
from .pose_estimator import PoseEstimator

class PerceptionPipeline:
    def __init__(self, model_path: str="yolov8n-pose.pt", bed_roi_polygon: Optional[List[Tuple[int, int]]] = None, conf_threshold: float = 0.35, enable_clahe: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        self.conf_threshold = conf_threshold
        self.enable_clahe = enable_clahe

        self.bed_detector = BedDetector(roi_polygon=bed_roi_polygon)
        self.pose_estimator = PoseEstimator(conf_threshold=conf_threshold)
        self.tracker = PersonTracker()

        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        self.patient_track_id: Optional[int] = None
        self.frames_since_patient_seen: int = 0
        self.patient_loss_threshold: int = 900

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.enable_clahe:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def reanchor_patient_id(self, new_track_id: Optional[int] = None):
        self.patient_track_id = new_track_id
        self.frames_since_patient_seen = 0

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_sec: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        enhanced_frame = self._preprocess_frame(frame)

        if self.bed_detector.bed_roi is None:
            self.bed_detector.auto_detect_bed(enhanced_frame)
            
        results = self.model.track(
            enhanced_frame,
            persist=True,
            classes=[0],
            verbose=False,
            device=self.device
        )

        detections = []
        annotated_frame = frame.copy()

        annotated_frame = self.bed_detector.draw_roi(annotated_frame)

        if results and len(results) > 0 and results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(boxes))

            keypoints_data = results[0].keypoints.data.cpu().numpy()  
            
            raw_detections = []
            for bbox, track_id, kpts in zip(boxes, track_ids, keypoints_data):
                spatial_info = self.bed_detector.check_overlap(bbox, kpts, conf_thresh=self.conf_threshold)
                raw_detections.append({
                    "bbox": bbox,
                    "track_id": track_id,
                    "kpts": kpts,
                    "spatial_info": spatial_info
                })

            if self.patient_track_id is None and len(raw_detections) > 0:
                bed_candidates = [
                    d for d in raw_detections 
                    if d["spatial_info"]["bed_overlap_ratio"] > 0.30 or d["spatial_info"]["center_in_bed"]
                ]
                if bed_candidates:
                    best_candidate = max(bed_candidates, key=lambda x: x["spatial_info"]["bed_overlap_ratio"])
                    self.patient_track_id = int(best_candidate["track_id"])

            patient_found = False

            for det in raw_detections:
                bbox = det["bbox"]
                track_id = det["track_id"]
                kpts = det["kpts"]
                spatial_info = det["spatial_info"]

                is_primary = (track_id == self.patient_track_id)
                if is_primary:
                    patient_found = True
                    self.frames_since_patient_seen = 0

                torso_angle, torso_conf = self.pose_estimator.calculate_torso_angle(kpts)
                overall_conf = float(np.mean(kpts[:, 2]))

                det_record = {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "track_id": int(track_id),
                    "is_primary_subject": is_primary,
                    "bbox": bbox.tolist(),
                    "torso_angle": torso_angle,
                    "torso_confidence": torso_conf,
                    "overall_confidence": overall_conf,
                    "in_bed_roi": spatial_info["center_in_bed"],
                    "bed_overlap_ratio": spatial_info["bed_overlap_ratio"],
                    "raw_keypoints": kpts.tolist()
                }

                detections.append(det_record)
                color = (0, 255, 0) if is_primary else (0, 165, 255)
                cv2.rectangle(
                    annotated_frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color, 2
                )
                for kp in kpts:
                    if kp[2] >= self.conf_threshold:
                        cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

                angle_str = f"Angle: {torso_angle:.1f}°" if torso_angle is not None else "Angle: OCCLUDED"
                label_str = f"ID:{track_id} {'(Patient)' if is_primary else '(Caregiver)'} | {angle_str}"
                
                cv2.putText(
                    annotated_frame, label_str,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            if not patient_found and self.patient_track_id is not None:
                self.frames_since_patient_seen += 1
                if self.frames_since_patient_seen > self.patient_loss_threshold:
                    self.patient_track_id = None
                    self.frames_since_patient_seen = 0

        status_text = f"Patient ID: {self.patient_track_id if self.patient_track_id is not None else 'SEARCHING...'}"
        cv2.putText(annotated_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return annotated_frame, detections