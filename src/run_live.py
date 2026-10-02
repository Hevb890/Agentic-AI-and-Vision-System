from pipeline import PatientMonitoringPipeline

if __name__ == "__main__":
    # Initialize monitoring pipeline
    pipeline = PatientMonitoringPipeline(
        model_path="yolov8n-pose.pt",
        bed_roi=None,  # Triggers automatic Bed ROI detection on frame 0
        fps=10.0,
        enable_vlm=False
    )

    # Start live feed using MacBook webcam (camera_index=0)
    pipeline.process_live_stream(
        camera_index=0,
        output_video_path="data/raw_videos/recorded_patient_session.mp4",
        output_json_path="data/processed_data/session_summary.json"
    )