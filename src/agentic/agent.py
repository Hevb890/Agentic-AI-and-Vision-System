from collections import deque
import numpy as np
from typing import Dict, List, Any, Optional

from temporal import ActivityState
from .vlm_client import VLMClient
from .prompts import BED_EXIT_VERIFICATION_PROMPT, AMBIGUITY_RESOLUTION_PROMPT

class AgenticOrchestration:
    def __init__(self, fps: float = 10.0, buffer_seconds: int = 8, enable_vlm: bool = True):
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)  
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.enable_vlm = enable_vlm
        self.vlm_client = VLMClient() if enable_vlm else None

        self.last_state = ActivityState.UNKNOWN
        self.bed_exit_events: List[Dict[str, Any]] = []

    def push_frame(self, frame: np.ndarray):
        self.frame_buffer.append(frame.copy())


    def inspect_transition(
        self,
        current_state: ActivityState,
        timestamp_sec: float,
        detection_record: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        agent_decision = {
            "state_override": None,
            "event_triggered": None,
            "vlm_inspection": None
        }
        is_bed_exit_candidate = (
            self.last_state in [ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED]
            and current_state in [ActivityState.STANDING, ActivityState.OUT_OF_BED, ActivityState.WALKING]
        )
        is_unknown_state = (current_state == ActivityState.UNKNOWN and self.last_state != ActivityState.UNKNOWN)

        if is_bed_exit_candidate:
            agent_decision["event_triggered"] = "BED_EXIT_CANDIDATE"
            if self.enable_vlm and len(self.frame_buffer) > 0:
                vlm_res = self.vlm_client.verify_event(
                    frames=list(self.frame_buffer),
                    system_prompt=BED_EXIT_VERIFICATION_PROMPT,
                    user_question=f"Confirm if patient exited bed at timestamp {timestamp_sec:.1f}s."
                )
                agent_decision["vlm_inspection"] = vlm_res

                if vlm_res.get("verified", False):
                    self.bed_exit_events.append({
                        "timestamp_sec": round(timestamp_sec, 2),
                        "event": "BED_EXIT",
                        "reasoning": vlm_res.get("reasoning", "")
                    })

        elif is_unknown_state:
            agent_decision["event_triggered"] = "AMBIGUOUS_STATE"
            if self.enable_vlm and len(self.frame_buffer) > 0:
                vlm_res = self.vlm_client.verify_event(
                    frames=list(self.frame_buffer),
                    system_prompt=AMBIGUITY_RESOLUTION_PROMPT,
                    user_question=f"Determine state at timestamp {timestamp_sec:.1f}s."
                )
                agent_decision["vlm_inspection"] = vlm_res

        self.last_state = current_state
        return agent_decision

    def get_bed_exit_logs(self) -> List[Dict[str, Any]]:
        return self.bed_exit_events