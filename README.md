# Agentic-AI-and-Vision-System

ai-bed-exit-monitor/
├── .github/ # CI/CD workflows (linting, testing, docker build)
│ └── workflows/
│ └── ci.yml
├── configs/ # Configuration files (YAML / Hydra)
│ ├── config.yaml # Main application config
│ ├── models.yaml # Model weights, thresholds, confidence levels
│ └── rules.yaml # Alert rules & temporal state timeout settings
├── data/ # Local data directory (git-ignored)
│ ├── raw_videos/ # Input video streams/files
│ ├── processed/ # Extracted frames / cropped ROIs
│ └── ground_truth/ # Labels for evaluation (annotations/JSONs)
├── docs/ # Documentation & architecture assets
│ ├── architecture.png # Architecture diagram
│ └── metrics_report.md # Failure cases & evaluation breakdowns
├── src/ # Main source code package
│ ├── **init**.py
│ ├── common/ # Shared utilities & helpers
│ │ ├── **init**.py
│ │ ├── io.py # Video/Frame ingestion & file handling
│ │ ├── logger.py # Structured logging (Loguru / Python logging)
│ │ └── visualization.py # Drawing bounding boxes, keypoints, ROI overlays
│ │
│ ├── perception/ # Tier 1: Computer Vision & Pose Pipelines
│ │ ├── **init**.py
│ │ ├── bed_detector.py # ROI polygon definition & spatial IoU checking
│ │ ├── pose_estimator.py # YOLO-Pose / MediaPipe keypoint extraction
│ │ └── tracker.py # ByteTrack / DeepSORT tracking wrapper
│ │
│ ├── temporal/ # Tier 2: State Tracking & Temporal Processing
│ │ ├── **init**.py
│ │ ├── fsm.py # Finite State Machine & state transition rules
│ │ ├── duration_engine.py # Active state duration aggregator
│ │ └── window_filter.py # Sliding window temporal smoothing
│ │
│ ├── agentic/ # Tier 3: Agent Orchestration & VLM Integration
│ │ ├── **init**.py
│ │ ├── agent.py # Re-query loop controller & frame context fetcher
│ │ ├── prompts.py # System prompts for VLM verification
│ │ └── vlm_client.py # OpenAI GPT-4o / Gemini VLM API wrapper
│ │
│ ├── alerts/ # Tier 4: Contextual Alerting & Outputs
│ │ ├── **init**.py
│ │ ├── alert_rules.py # NORMAL / MONITOR / ALERT policy logic
│ │ └── output_formatter.py # Generates final JSON schema & summaries
│ │
│ └── pipeline.py # Main end-to-end execution pipeline
│
├── tests/ # Unit and Integration Tests
│ ├── unit/ # Tests for individual modules (FSM, IoU, parser)
│ ├── integration/ # End-to-end pipeline run tests
│ └── test_data/ # Mock clips or frame samples for testing
│
├── scripts/ # Executable scripts for CLI usage & evaluation
│ ├── run_pipeline.py # Primary CLI entrypoint for running on a video
│ └── evaluate.py # Computes precision, recall, MAE duration errors
│
├── .gitignore # Git ignore rules
├── Dockerfile # Containerization setup
├── README.md # Setup instructions, run guide, architecture overview
├── requirements.txt # Production dependencies
└── setup.py / pyproject.toml # Package definition for modular imports
