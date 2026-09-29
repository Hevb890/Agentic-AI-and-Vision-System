import numpy as np
from typing import Tuple, Optional

class PoseEstimator:
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6 
    LEFT_HIP, RIGHT_HIP = 11, 12

    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold

    def calculate_torso_angle(self, kpts: np.ndarray) -> Tuple[Optional[float],float]:
        ls, rs = kpts[self.LEFT_SHOULDER], kpts[self.RIGHT_SHOULDER]
        lh, rh = kpts[self.LEFT_HIP], kpts[self.RIGHT_HIP]

        torso_conf = float(np.mean([ls[2], rs[2], lh[2], rh[2]]))
        if torso_conf < self.conf_threshold:
            return None, torso_conf 

        shoulder_center = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
        hip_center = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]

        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]

        angle = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
        return angle, torso_conf
