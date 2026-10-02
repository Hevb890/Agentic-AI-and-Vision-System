import cv2
import logging
from typing import List, Optional, Any, Dict, Tuple
import time

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
        bed_exists = self.agent.get_bed_exit_logs()
        alert_info = self.alerts.evaluate_status(summary["durations_seconds"], bed_exists)

        final_payload = OutputFormatter.build_summary(
            durations=summary["durations_seconds"],
            timeline=summary["timeline"],
            bed_exits=bed_exists,
            alert_info=alert_info,
            video_metadata={"video_path": video_path, "total_frames": frame_idx, "fps": video_fps}
        )

        OutputFormatter.save_json(final_payload, output_json_path)
        logger.info(f"Pipeline finished. Saved summary to {output_json_path}")
        return final_payload

    def process_live_stream(
            self,
            camera_index: int = 0,
            output_video_path: str = "live_output.mp4",
            output_json_path: str = "live_summary.json"
    ) -> Dict[str,Any]:
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

        if not cap.isOpened():
            logger.error(f"Cannot open webcam with index {camera_index}")
            raise RuntimeError(f"Could not access camera index {camera_index}. Check macOS Privacy Permissions.")
      
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (frame_width, frame_height))

        frame_idx = 0
        start_time = time.time()

        logger.info(f"Starting Live Webcam Pipeline. Recording to {output_video_path}. Press 'q' to stop.")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame from live stream.")
                    break

                timestamp_sec = round(time.time() - start_time, 2)
                annotated_frame, detections = self.perception_pipeline.process_frame(
                    frame, frame_idx, timestamp_sec
                )

                primary_det = next((d for d in detections if d.get("is_primary_subject")), None)

                draft_state = self.fsm.evaluate_frame(primary_det)
                smoothed_res = self.window_filter.add_and_smooth(draft_state)

                if isinstance(smoothed_res, tuple):
                    smoothed_state = smoothed_res[0]
                else:
                    smoothed_state = smoothed_res
                self.duration_engine.update(smoothed_state, timestamp_sec)

                self.agent.push_frame(frame)
                self.agent.inspect_transition(
                    current_state=smoothed_state,
                    timestamp_sec=timestamp_sec,
                    detection_record=primary_det
                )

                cv2.putText(
                    annotated_frame, f"STATE: {smoothed_state} | TIME: {timestamp_sec:.1f}s", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

                out_writer.write(annotated_frame)

                cv2.imshow("MacBook Live Patient Monitor", annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r'):
                    logger.info("Recalibrating Bed ROI via 'r' key command...")
                    self.perception_pipeline.bed_detector.auto_detect_bed(frame)
                frame_idx += 1

                if key == ord('q'):
                    logger.info("User requested stop via 'q' key.")
                    break

        finally:
            cap.release()
            out_writer.release()
            cv2.destroyAllWindows()
        summary = self.duration_engine.get_summary()
        bed_exits = self.agent.get_bed_exit_logs()
        alert_info = self.alerts.evaluate_status(summary["durations_seconds"], bed_exits)

        final_payload = OutputFormatter.build_summary(
            durations=summary["durations_seconds"],
            timeline=summary["timeline"],
            bed_exits=bed_exits,
            alert_info=alert_info,
            video_metadata={
                "video_path": output_video_path,
                "total_frames": frame_idx,
                "fps": self.fps,
                "duration_seconds": round(time.time() - start_time, 2)
            }
        )
        OutputFormatter.save_json(final_payload, output_json_path)
        logger.info(f"Live session ended. Saved recorded video to '{output_video_path}' and JSON to '{output_json_path}'.")
        return final_payload