from collections import deque, Counter
from typing import List
from .fsm import ActivityState

class TemporalWindowFilter:
    def __init__(self, window_size: int = 15):
        self.window = deque(maxlen=window_size)

    def add_and_smooth(self, draft_state: ActivityState) -> ActivityState:
        self.window.append(draft_state)
        counts = Counter(self.window)
        most_common_state = counts.most_common(1)[0]
        return most_common_state
        
