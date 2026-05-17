"""
Utility modules
"""

from .run_manager import RunManager
from .analysis import ResultsAnalyzer
from .training_curves import TrainingCurveRecorder, TrainingCurveAggregator
from .feature_importance import FeatureImportanceRecorder, FeatureImportanceAggregator

class TimingTracker:
    """Lightweight phase timer; embed in workflows; export as dict or JSON"""
    def __init__(self):
        from time import perf_counter
        self._pc = perf_counter
        self._t0 = self._pc()
        self._stack = {}
        self.metrics = {}

    def start(self, key: str):
        self._stack[key] = self._pc()

    def end(self, key: str):
        if key in self._stack:
            dt = self._pc() - self._stack.pop(key)
            self.metrics[key] = self.metrics.get(key, 0.0) + dt
            return dt
        return 0.0

    def total(self):
        return self._pc() - self._t0

    def to_dict(self):
        d = dict(self.metrics)
        d['total_runtime'] = self.total()
        return d

__all__ = [
    'RunManager',
    'ResultsAnalyzer',
    'TrainingCurveRecorder',
    'TrainingCurveAggregator',
    'FeatureImportanceRecorder',
    'FeatureImportanceAggregator',
    'TimingTracker'
]
