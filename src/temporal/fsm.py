from enum import Enum
from typing import Dict, Any, Optional

class ActivityState(str, Enum):
    LYING_IN_BED = "LYING_IN_BED"
    SITTING_ON_BED = "SITTING_ON_BED"
    SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
    STANDING = "STANDING"
    WALKING = "WALKING"
    OUT_OF_BED = "OUT_OF_BED"
    UNKNOWN = "UNKNOWN"

class StateMachine:
    def __init__(self, angle_lying_threshold: float = 30.0, angle_sitting_threshold: float = 60.0):
        self.angle_lying_threshold = angle_lying_threshold
        self.angle_sitting_threshold = angle_sitting_threshold

    def evaluate_frame(self, detection: Dict[str, Any]) -> ActivityState:
        if not detection or detection.get("torso_confidence", 0.0) < 0.3:
            return ActivityState.UNKNOWN

        angle = detection.get("torso_angle")
        in_bed_roi = detection.get("in_bed_roi", False)
        overlap_ratio = detection.get("bed_overlap_ratio", 0.0)

        if angle is None:
            return ActivityState.UNKNOWN

        if in_bed_roi or overlap_ratio > 0.4:
            if angle <= self.angle_lying_threshold:
                return ActivityState.LYING_IN_BED
            elif angle > self.angle_lying_threshold:
                return ActivityState.SITTING_ON_BED

        else:
            if angle <= self.angle_lying_threshold:
                return ActivityState.UNKNOWN
            elif angle > self.angle_sitting_threshold:
                return ActivityState.STANDING
            else:
                return ActivityState.SITTING_OUTSIDE_BED

        return ActivityState.UNKNOWN        