#!/usr/bin/env python3
"""
Timing utilities module
Provides unified time tracking and performance analysis
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path


@dataclass
class TimeRecord:
    """Time record"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def stop(self):
        """Stop timing"""
        if self.end_time is None:
            self.end_time = time.perf_counter()
            self.duration = self.end_time - self.start_time
        return self.duration
    
    def get_duration(self) -> float:
        """Get duration"""
        if self.duration is not None:
            return self.duration
        elif self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return time.perf_counter() - self.start_time


class TimingTracker:
    """
    Timing tracker
    Supports nested time records and performance statistics
    """
    
    def __init__(self, name: str = "main"):
        """
        Initialize timing tracker
        
        Args:
            name: Tracker name
        """
        self.name = name
        self.records: Dict[str, TimeRecord] = {}
        self.active_records: List[str] = []
        self.completed_records: List[TimeRecord] = []
        self.start_time = time.perf_counter()
        
    def start(self, name: str, metadata: Optional[Dict] = None) -> TimeRecord:
        """
        Start timing
        
        Args:
            name: Timing name
            metadata: Additional metadata
        
        Returns:
            Time record object
        """
        record = TimeRecord(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata or {}
        )
        self.records[name] = record
        self.active_records.append(name)
        return record
    
    def stop(self, name: str) -> float:
        """
        Stop timing
        
        Args:
            name: Timing name
        
        Returns:
            Duration (seconds)
        """
        if name in self.records:
            record = self.records[name]
            duration = record.stop()
            
            # Remove from active records
            if name in self.active_records:
                self.active_records.remove(name)
            
            # Add to completed records
            self.completed_records.append(record)
            
            return duration
        return 0.0
    
    @contextmanager
    def measure(self, name: str, metadata: Optional[Dict] = None):
        """
        Context manager for timing
        
        Args:
            name: Timing name
            metadata: Additional metadata
        
        Example:
            with tracker.measure("feature_extraction"):
                pass
        """
        record = self.start(name, metadata)
        try:
            yield record
        finally:
            self.stop(name)
    
    def add_metric(self, name: str, key: str, value: Any):
        """
        Add metric to a time record
        
        Args:
            name: Timing name
            key: Metric key
            value: Metric value
        """
        if name in self.records:
            self.records[name].metadata[key] = value
    
    def get_duration(self, name: str) -> Optional[float]:
        """
        Get duration for a record
        
        Args:
            name: Timing name
        
        Returns:
            Duration (seconds)
        """
        if name in self.records:
            return self.records[name].get_duration()
        return None
    
    def calculate_throughput(self, name: str, samples: int) -> Optional[float]:
        """
        Calculate throughput
        
        Args:
            name: Timing name
            samples: Number of samples
        
        Returns:
            Throughput (samples/second)
        """
        duration = self.get_duration(name)
        if duration and duration > 0:
            throughput = samples / duration
            self.add_metric(name, "samples", samples)
            self.add_metric(name, "throughput", throughput)
            self.add_metric(name, "per_sample_time", duration / samples)
            return throughput
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get timing summary
        
        Returns:
            Summary dict
        """
        total_time = time.perf_counter() - self.start_time
        
        summary = {
            "total_time": total_time,
            "records": {}
        }
        
        # Add statistics for all records
        for name, record in self.records.items():
            duration = record.get_duration()
            record_data = {
                "duration": duration,
                "percentage": (duration / total_time * 100) if total_time > 0 else 0
            }
            
            # Add metadata
            if record.metadata:
                record_data["metrics"] = record.metadata
            
            summary["records"][name] = record_data
        
        return summary
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Get detailed time report
        
        Returns:
            Detailed report dict
        """
        summary = self.get_summary()
        
        # Compute key performance metrics
        performance_metrics = {}
        
        # Feature extraction performance
        if "feature_extraction" in self.records:
            fe_record = self.records["feature_extraction"]
            if "samples" in fe_record.metadata:
                samples = fe_record.metadata["samples"]
                duration = fe_record.get_duration()
                performance_metrics["feature_extraction"] = {
                    "total_time": duration,
                    "samples": samples,
                    "throughput": f"{samples/duration:.2f} samples/sec" if duration > 0 else "N/A",
                    "per_sample_avg": f"{duration/samples*1000:.2f} ms" if samples > 0 else "N/A"
                }
        
        # Training performance
        training_records = [name for name in self.records if "training" in name or "fold" in name]
        if training_records:
            total_training_time = sum(self.records[name].get_duration() for name in training_records)
            performance_metrics["training"] = {
                "total_time": total_training_time,
                "phases": len(training_records)
            }
            
            # Fold statistics
            fold_times = []
            for name in training_records:
                if "fold" in name:
                    fold_times.append(self.records[name].get_duration())
            
            if fold_times:
                performance_metrics["training"]["fold_times"] = fold_times
                performance_metrics["training"]["avg_fold_time"] = sum(fold_times) / len(fold_times)
        
        # Prediction performance
        prediction_records = [name for name in self.records if "predict" in name]
        if prediction_records:
            prediction_metrics = {}
            for name in prediction_records:
                record = self.records[name]
                if "samples" in record.metadata:
                    samples = record.metadata["samples"]
                    duration = record.get_duration()
                    prediction_metrics[name] = {
                        "time": duration,
                        "samples": samples,
                        "throughput": f"{samples/duration:.2f} samples/sec" if duration > 0 else "N/A"
                    }
            performance_metrics["prediction"] = prediction_metrics
        
        return {
            "summary": summary,
            "performance_metrics": performance_metrics,
            "timeline": self._get_timeline()
        }
    
    def _get_timeline(self) -> List[Dict]:
        """
        Get timeline
        
        Returns:
            Timeline list
        """
        timeline = []
        for record in self.completed_records:
            timeline.append({
                "name": record.name,
                "start": record.start_time - self.start_time,
                "end": (record.end_time - self.start_time) if record.end_time else None,
                "duration": record.duration
            })
        
        # Sort by start time
        timeline.sort(key=lambda x: x["start"])
        return timeline
    
    def save_report(self, filepath: Path, format: str = "json"):
        """
        Save timing report
        
        Args:
            filepath: Save path
            format: Format (json/txt)
        """
        report = self.get_detailed_report()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=True)
        elif format == "txt":
            with open(filepath, 'w') as f:
                f.write(self._format_text_report(report))
    
    def _format_text_report(self, report: Dict) -> str:
        """
        Format text report
        
        Args:
            report: Report dict
        
        Returns:
            Formatted text
        """
        lines = []
        lines.append("="*60)
        lines.append(f"Time Statistics Report - {self.name}")
        lines.append("="*60)
        
        summary = report["summary"]
        lines.append(f"\nTotal time: {summary['total_time']:.2f} s")
        lines.append("\nPhase durations:")
        lines.append("-"*40)
        
        for name, data in summary["records"].items():
            lines.append(f"  {name:30s}: {data['duration']:8.2f}s ({data['percentage']:5.1f}%)")
            if "metrics" in data and "throughput" in data["metrics"]:
                lines.append(f"    Throughput: {data['metrics']['throughput']:.2f} samples/sec")
        
        if "performance_metrics" in report:
            lines.append("\nPerformance metrics:")
            lines.append("-"*40)
            
            metrics = report["performance_metrics"]
            
            if "feature_extraction" in metrics:
                fe = metrics["feature_extraction"]
                lines.append(f"  Feature extraction:")
                lines.append(f"    - Total time: {fe['total_time']:.2f}s")
                lines.append(f"    - Throughput: {fe['throughput']}")
                lines.append(f"    - Avg per sample: {fe['per_sample_avg']}")
            
            if "training" in metrics:
                tr = metrics["training"]
                lines.append(f"  Model training:")
                lines.append(f"    - Total time: {tr['total_time']:.2f}s")
                if "avg_fold_time" in tr:
                    lines.append(f"    - Avg per fold: {tr['avg_fold_time']:.2f}s")
            
            if "prediction" in metrics:
                lines.append(f"  Prediction:")
                for name, pred in metrics["prediction"].items():
                    lines.append(f"    {name}:")
                    lines.append(f"      - Time: {pred['time']:.2f}s")
                    lines.append(f"      - Throughput: {pred['throughput']}")
        
        lines.append("\n" + "="*60)
        return "\n".join(lines)
    
    def print_summary(self):
        """Print timing summary"""
        report = self.get_detailed_report()
        print(self._format_text_report(report))


# Global timing tracker instance
_global_tracker: Optional[TimingTracker] = None


def get_global_tracker() -> TimingTracker:
    """Get global timing tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TimingTracker("global")
    return _global_tracker


def reset_global_tracker():
    """Reset global timing tracker"""
    global _global_tracker
    _global_tracker = TimingTracker("global")
