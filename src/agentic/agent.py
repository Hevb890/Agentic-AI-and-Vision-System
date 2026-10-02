from collections import deque
import numpy as np
import time
from typing import Dict, List, Any, Optional

from temporal import ActivityState
from .vlm_client import VLMClient
from .prompts import (
    BED_EXIT_VERIFICATION_PROMPT, 
    AMBIGUITY_RESOLUTION_PROMPT,
    ROLE_DISAMBIGUATION_PROMPT
)

class AgenticOrchestration:
    def __init__(
        self, 
        fps: float = 10.0, 
        buffer_seconds: int = 8, 
        enable_vlm: bool = True,
        vlm_cooldown_sec: float = 5.0,
        pose_conf_thresh: float = 0.35
    ):
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)  
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.enable_vlm = enable_vlm
        self.vlm_client = VLMClient() if enable_vlm else None

        self.last_state = ActivityState.UNKNOWN
        self.bed_exit_events: List[Dict[str, Any]] = []

        # Confidence & Cooldown management for VLM API calls
        self.pose_conf_thresh = pose_conf_thresh
        self.vlm_cooldown_sec = vlm_cooldown_sec
        self.last_vlm_call_time = 0.0

    def push_frame(self, frame: np.ndarray):
        self.frame_buffer.append(frame.copy())

    def _should_trigger_vlm(self) -> bool:
        """Enforces API cooldown limits."""
        now = time.time()
        if now - self.last_vlm_call_time >= self.vlm_cooldown_sec:
            self.last_vlm_call_time = now
            return True
        return False

    def inspect_transition(
        self,
        current_state: ActivityState,
        timestamp_sec: float,
        detection_records: Optional[List[Dict[str, Any]]] = None,
        patient_track_id: Optional[int] = None
    ) -> Dict[str, Any]:
        agent_decision = {
            "state_override": None,
            "event_triggered": None,
            "vlm_inspection": None,
            "caregiver_present": False
        }

        # --- 1. Evaluate State Transition Candidates ---
        is_bed_exit_candidate = (
            self.last_state in [ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED]
            and current_state in [ActivityState.STANDING, ActivityState.OUT_OF_BED, ActivityState.WALKING]
        )
        is_unknown_state = (current_state == ActivityState.UNKNOWN and self.last_state != ActivityState.UNKNOWN)

        # --- 2. Evaluate Ambiguity & Confidence Metrics ---
        detections = detection_records or []
        in_bed_count = sum(1 for d in detections if d.get("in_bed_roi") or d.get("bed_overlap_ratio", 0) > 0.20)
        
        is_multi_person_ambiguity = (in_bed_count >= 2)
        is_unanchored_identity = (patient_track_id is None and len(detections) >= 1)
        
        low_confidence_detected = False
        for d in detections:
            if d.get("is_primary_subject") and d.get("overall_confidence", 1.0) < self.pose_conf_thresh:
                low_confidence_detected = True

        # --- 3. Execute VLM Inspections Based on Triggers ---
        if is_bed_exit_candidate:
            agent_decision["event_triggered"] = "BED_EXIT_CANDIDATE"
            if self.enable_vlm and len(self.frame_buffer) > 0 and self._should_trigger_vlm():
                vlm_res = self.vlm_client.verify_event(
                    frames=list(self.frame_buffer),
                    system_prompt=BED_EXIT_VERIFICATION_PROMPT,
                    user_question=f"Confirm if patient exited bed unassisted at timestamp {timestamp_sec:.1f}s."
                )
                agent_decision["vlm_inspection"] = vlm_res
                agent_decision["caregiver_present"] = vlm_res.get("caregiver_present", False)

                # Only register bed exit if verified AND NOT assisted by caregiver
                if vlm_res.get("verified", False) and not agent_decision["caregiver_present"]:
                    self.bed_exit_events.append({
                        "timestamp_sec": round(timestamp_sec, 2),
                        "event": "UNASSISTED_BED_EXIT",
                        "reasoning": vlm_res.get("reasoning", "")
                    })

        elif is_multi_person_ambiguity or is_unanchored_identity:
            agent_decision["event_triggered"] = "ROLE_OR_IDENTITY_AMBIGUITY"
            if self.enable_vlm and len(self.frame_buffer) > 0 and self._should_trigger_vlm():
                vlm_res = self.vlm_client.verify_event(
                    frames=list(self.frame_buffer),
                    system_prompt=ROLE_DISAMBIGUATION_PROMPT,
                    user_question=f"Disambiguate patient vs caregiver at timestamp {timestamp_sec:.1f}s."
                )
                agent_decision["vlm_inspection"] = vlm_res
                agent_decision["caregiver_present"] = vlm_res.get("caregiver_present", False)

        elif is_unknown_state or low_confidence_detected:
            agent_decision["event_triggered"] = "LOW_CONFIDENCE_OR_UNKNOWN_STATE"
            if self.enable_vlm and len(self.frame_buffer) > 0 and self._should_trigger_vlm():
                vlm_res = self.vlm_client.verify_event(
                    frames=list(self.frame_buffer),
                    system_prompt=AMBIGUITY_RESOLUTION_PROMPT,
                    user_question=f"Determine state and resolve occlusion at timestamp {timestamp_sec:.1f}s."
                )
                agent_decision["vlm_inspection"] = vlm_res

        self.last_state = current_state
        return agent_decision

    def get_bed_exit_logs(self) -> List[Dict[str, Any]]:
        return self.bed_exit_events