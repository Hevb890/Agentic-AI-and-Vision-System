from typing import Dict, List, Any, Optional
from .fsm import ActivityState

class DurationEngine:
    def __init__(self, fps: float = 10.0):
        self.fps = fps
        self.frame_detection = 1.0 / fps
        self.durations: Dict[str, float] = {state.value: 0.0 for state in ActivityState}
        self.timeline: List[Dict[str, Any]] = []
        self.current_segment: Optional[Dict[str, Any]] = None

    def update(self, state: ActivityState, timestamp_sec: float):
        self.durations[state.value] += self.frame_detection

        if self.current_segment is None:
            self.current_segment = {"state": state.value, "start_time": timestamp_sec, "end_time": timestamp_sec}
        elif self.current_segment["state"] == state.value:
            self.current_segment["end_time"] = timestamp_sec
        else:
            self.timeline.append(self.current_segment)
            self.current_segment = {"state": state.value, "start_time": timestamp_sec, "end_time": timestamp_sec}

    def get_summary(self) -> Dict[str, Any]:
        if self.current_segment and self.current_segment not in self.timeline:
            self.timeline.append(self.current_segment)
        return{
            "duration_seconds": {k: round(v,2) for k, v in self.durations.items()},
            "timeline": self.timeline
        }