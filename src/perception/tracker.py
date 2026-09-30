import numpy as np
from typing import Optional, List, Dict, Any

class PersonTracker:
    def __init__(self):
        self.primary_subject_id: Optional[int] = None

    def filter_primary_subject(self, track_ids: np.ndarray) -> List[bool]:
        if self.primary_subject_id is None and len(track_ids) > 0:
            self.primary_subject_id = int(track_ids[0])

        return [int(tid) == self.primary_subject_id for tid in track_ids]