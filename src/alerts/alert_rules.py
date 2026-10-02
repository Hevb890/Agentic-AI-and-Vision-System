from typing import List, Any, Dict

class AlertRulesEngine:
    def __init__(self, prolonged_sitting_threshold: float = 60.0, out_of_bed_alert_threshold: float = 60.0):
        self.prolonged_sitting_threshold_sec = prolonged_sitting_threshold
        self.out_of_bed_threshold = out_of_bed_alert_threshold

    def evaluate_status(self, durations: Dict[str, float], bed_exit_events: List[Dict[str, Any]]) -> Dict[str, Any]:

        is_alert = False
        is_monitor = False
        messages = []


        out_time = durations.get("OUT_OF_BED", 0.0) + durations.get("WALKING", 0.0) + durations.get("STANDING", 0.0)
        sitting_time = durations.get("SITTING_ON_BED", 0.0)

        if len(bed_exit_events) > 0:
            messages.append(f"Bed exit detected ({len(bed_exit_events)} event(s))")
            if out_time > self.out_of_bed_threshold:
                is_alert = True
                messages.append(f"Patient out of bed area for prolonged duration ({int(out_time)}s).")

        
        if sitting_time > self.prolonged_sitting_threshold_sec:
            is_monitor = True
            messages.append(f"Patient sitting on bed edge for > {int(sitting_time)}s.")

        if is_alert:
            alert_level = "ALERT"
        elif is_monitor:
            alert_level = "MONITOR"
        else:
            alert_level = "NORMAL"

        return {
            "alert_level": alert_level,
            "messages": messages,
            "bed_exit_count": len(bed_exit_events)
        }