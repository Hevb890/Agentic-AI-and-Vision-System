import cv2
import logging
from typing import List, Optional, Any, Dict, Tuple

from perception.pipeline import PerceptionPipeline
from temporal import StateMachine, TemporalWindowFilter, DurationEngine
from agentic import AgenticOrchestration
from alerts import AlertRulesEngine, OutputFormatter

logger = logging.getLogger(__name__)

class PatientMonitoringPipeline:
    def __init__(
            self,
            model_path: str = "yolov8n-pose.pt",
            bed_roi: Optional[List[Tuple[int, int]]] = None,
            fps: float = 10.0,
            enable_vlm: bool = False
    ):
        self.fps = fps
        self.perception_pipeline = PerceptionPipeline(model_path, bed_roi)
        self.fsm = StateMachine()
        self.window_filter = TemporalWindowFilter(window_size=int(fps * 1.5))
        self.duration_engine = DurationEngine(fps)
        self.agent = AgenticOrchestration(fps, enable_vlm)
        self.alerts = AlertRulesEngine()

    def process_video(self, video_path: str, output_json_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        frame_idx = 0

        logger.info(f"Starting pipeline execution on Video: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_idx / video_fps

            annotated_frame, detections = self.perception_pipeline.process_frame(frame, frame_idx, timestamp_sec)
            primary_det = next((d for d in detections if d.get("is_primary_subject")), None)

            draft_state = self.fsm.evaluate_frame(primary_det)
            smoothed_state = self.window_filter.add_and_smooth(draft_state)
            self.duration_engine.update(smoothed_state, timestamp_sec)

            self.agent.push_frame(frame)
            self.agent.inspect_transition(
                current_state=smoothed_state,
                timestamp_sec=timestamp_sec,
                detection_record=primary_det
            )

            frame_idx += 1

        cap.release()

        summary = self.duration_engine.get_smmary()
        bed_exits = self.agent.get_bed_exit_logs()
        alert_info = self.alerts.evaluate_status(summary["durations_seconds"], bed_exists)

        final_payload = OutputFormatter.build_summary(
            durations=summary["durations_seconds"],
            timeline=summary["timeline"],
            bed_exits=bed_exits,
            alert_info=alert_info,
            video_metadata={"video_path": video_path, "total_frames": frame_idx, "fps": video_fps}
        )

        OutputFormatter.save_json(final_payload, output_json_path)
        logger.info(f"Pipeline finished. Saved summary to {output_json_path}")
        return final_payload