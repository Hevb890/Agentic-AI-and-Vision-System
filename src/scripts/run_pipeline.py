import argparse
from pipeline import PatientMonitoringPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Patient Monitoring AI Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="data/processed/summary.json", help="Output JSON path")
    parser.add_argument("--enable-vlm", action="store_true", help="Enable VLM verification in Tier 3")
    args = parser.parse_args()

    default_bed_roi = [(150, 250), (750, 250), (750, 700), (150, 700)]

    pipeline = PatientMonitoringPipeline(
        model_path="yolov8n-pose.pt",
        bed_roi=default_bed_roi,
        enable_vlm=args.enable_vlm
    )

    pipeline.process_video(args.video, args.output)

if __name__ == "__main__":
    main()