import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class OutputFormatter:
    @staticmethod
    def build_summary(
        durations: Dict[str, float],
        timeline: List[Dict[str, Any]],
        bed_exits: List[Dict[str, Any]],
        alert_info: Dict[str, Any],
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        summary_payload = {
            "system_status": alert_info.get("alert_level", "ALERT"),
            "alert_messages": alert_info.get("messages", []),
            "metrics": {
                "total_bed_exits": alert_info.get("bed_exit_count", 0),
                "activity_durations_seconds": {
                    k: round(v, 2) for k, v in durations.items()
                }
            },
            "event_log": bed_exits,
            "activity_timeline": timeline
        }

        if video_metadata:
            summary_payload["metadata"] = video_metadata
        return summary_payload

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"[OutputFormatter] Successfully saved output to {file_path}")
        except IOError as e:
            logger.error(f"[OutputFormatter] Failed to write JSON to {file_path}: {e}")
            raise