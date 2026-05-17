#!/usr/bin/env python3
"""Workflow-level progress events and fixed-area monitor."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from phosirdesign.utils.training_progress import (
    ProgressState,
    _format_duration,
    add_training_metric_columns,
    build_training_metric_row,
)


DEFAULT_WORKFLOW_STEPS: List[Tuple[str, str]] = [
    ("data_checks", "Data checks"),
    ("virtual_db_stats", "Virtual DB stats"),
    ("train_models", "Train models"),
    ("model_comparison", "Model comparison"),
    ("best_model_plots", "Best model plots"),
    ("virtual_predictions", "Virtual predictions"),
    ("virtual_plots", "Virtual plots"),
    ("stratified_analysis", "Stratified analysis"),
    ("publication_figures", "Publication figures"),
    ("test_data_prediction", "Test data prediction"),
    ("shap_analysis", "SHAP analysis"),
    ("final_summary", "Final summary"),
]


def write_progress_event(path: str, event: Dict) -> None:
    """Append one JSONL progress event."""
    if not path:
        return
    event = dict(event)
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


@dataclass
class WorkflowProgressState:
    """State for the complete workflow plus nested training progress."""

    steps: List[Tuple[str, str]]
    models: List[str]
    targets: List[str]
    recent_limit: int = 10
    started_at: float = field(default_factory=time.perf_counter)
    workflow_status: Dict[str, str] = field(default_factory=dict)
    current_pipeline_step: str = "starting"
    finished: bool = False
    analysis_context: Dict[str, object] = field(default_factory=dict)
    ours_predictions: Dict[str, Dict[str, object]] = field(default_factory=dict)
    training: ProgressState = field(init=False)
    _recent_logs: Deque[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.workflow_status = {step_id: "pending" for step_id, _ in self.steps}
        self.training = ProgressState(self.models, self.targets, recent_limit=self.recent_limit)
        self._recent_logs = deque(maxlen=self.recent_limit)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def recent_logs(self) -> List[str]:
        logs = list(self._recent_logs)
        for line in self.training.recent_logs:
            if line not in logs:
                logs.append(line)
        return logs[-self.recent_limit :]

    @property
    def completed_steps(self) -> int:
        return sum(1 for status in self.workflow_status.values() if status == "done")

    def label_for(self, step: str) -> str:
        for step_id, label in self.steps:
            if step_id == step:
                return label
        return step

    def log(self, message: str) -> None:
        for line in str(message).splitlines():
            clean = line.rstrip()
            if clean:
                self._recent_logs.append(clean)

    def apply_event(self, event: Dict) -> None:
        event_type = event.get("type")
        if event_type == "workflow_step_started":
            step = event["step"]
            self.workflow_status[step] = "running"
            self.current_pipeline_step = self.label_for(step)
            self.log(f"started: {self.current_pipeline_step}")
        elif event_type == "workflow_step_completed":
            step = event["step"]
            self.workflow_status[step] = "done"
            self.current_pipeline_step = self.label_for(step)
            self.log(f"done: {self.current_pipeline_step}")
        elif event_type == "workflow_step_warning":
            step = event["step"]
            self.workflow_status[step] = "warn"
            self.current_pipeline_step = self.label_for(step)
            self.log(event.get("message", f"warning: {self.current_pipeline_step}"))
        elif event_type == "workflow_step_failed":
            step = event["step"]
            self.workflow_status[step] = "failed"
            self.current_pipeline_step = self.label_for(step)
            self.log(event.get("message", f"failed: {self.current_pipeline_step}"))
        elif event_type == "workflow_finished":
            self.finished = True
            self.current_pipeline_step = "completed"
            self.log(event.get("message", "workflow completed"))
        elif event_type == "log":
            self.log(event.get("message", ""))
        elif event_type == "workflow_context":
            self._apply_workflow_context(event)
        elif event_type == "training_context":
            self._apply_training_context(event)
        elif event_type == "ours_predictions":
            self._apply_ours_predictions(event)
        elif event_type == "model_started":
            self.training.model_started(event["model"], int(event.get("index", 1)), int(event.get("total", 1)))
        elif event_type == "model_completed":
            self.training.model_completed(event["model"], float(event.get("elapsed", 0)))
        elif event_type == "model_failed":
            self.training.model_failed(event["model"], event.get("error", "unknown error"))
        elif event_type == "target_started":
            self.training.target_started(
                event["model"],
                event["target"],
                int(event.get("index", 1)),
                int(event.get("total", 1)),
            )
        elif event_type == "target_completed":
            self.training.target_completed(
                event["model"],
                event["target"],
                float(event.get("elapsed", 0)),
                metrics=event.get("metrics"),
            )
        elif event_type == "fold_started":
            self.training.fold_started(
                event["model"],
                event["target"],
                int(event.get("fold", 1)),
                int(event.get("total", 1)),
            )
        elif event_type == "training_step":
            self.training.step(event.get("message", "training"))

    def _apply_workflow_context(self, event: Dict) -> None:
        for key in (
            "training_data_path",
            "training_data_rows",
            "output_dir",
            "screen_operations",
        ):
            if key in event and event[key] not in (None, ""):
                self.analysis_context[key] = event[key]

    def _apply_ours_predictions(self, event: Dict) -> None:
        model = str(event.get("model", "unknown"))
        rows = event.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        self.ours_predictions[model] = {
            "output_path": event.get("output_path", ""),
            "rows": rows,
        }

    def _apply_training_context(self, event: Dict) -> None:
        for key in (
            "model",
            "target",
            "samples",
            "feature_count",
            "feature_type",
        ):
            if key in event and event[key] not in (None, ""):
                self.analysis_context[key] = event[key]


def render_workflow(state: WorkflowProgressState):
    """Render workflow plus nested training state."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    root = Table.grid(expand=True)
    root.add_column(ratio=1)
    show_compact_final = state.finished and bool(state.ours_predictions)

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    title = "PhosIrDesign Workflow Monitor"
    if state.finished:
        title = "PhosIrDesign Workflow Monitor - completed"
    header.add_row(Text(title, style="bold cyan"), f"Elapsed: {_format_duration(state.elapsed)}")
    header.add_row(
        f"Pipeline: {state.completed_steps}/{len(state.steps)} complete",
        f"Current: {state.current_pipeline_step}",
    )
    root.add_row(Panel(header, box=box.SIMPLE))

    if not show_compact_final:
        pipeline = Table(box=box.SIMPLE, expand=True)
        pipeline.add_column("Pipeline Step", style="bold", no_wrap=True)
        pipeline.add_column("Status", no_wrap=True)
        pipeline.add_column("Progress")
        for step_id, label in state.steps:
            status = state.workflow_status.get(step_id, "pending")
            icon = {
                "pending": "[ ]",
                "running": "[>]",
                "done": "[x]",
                "warn": "[!]",
                "failed": "[X]",
            }.get(status, "[ ]")
            style = {
                "running": "cyan",
                "done": "green",
                "warn": "yellow",
                "failed": "red",
            }.get(status, "dim")
            pipeline.add_row(label, Text(f"{icon} {status}", style=style), "")
        root.add_row(Panel(pipeline, title="Pipeline", box=box.SIMPLE))

    training = Table(box=box.SIMPLE, expand=True)
    add_training_metric_columns(training, state.targets)
    for model in state.models:
        training.add_row(*build_training_metric_row(state.training, model))
    root.add_row(Panel(training, title="Training", box=box.SIMPLE))

    context_lines = _analysis_context_lines(state.analysis_context)
    if context_lines:
        root.add_row(Panel("\n".join(context_lines), title="Analysis Context", box=box.SIMPLE))

    if not show_compact_final:
        current = Table.grid(expand=True)
        current.add_column(ratio=1)
        fold = "-"
        if state.training.current_fold and state.training.total_folds:
            fold = f"{state.training.current_fold}/{state.training.total_folds}"
        current.add_row(f"Pipeline: {state.current_pipeline_step}")
        current.add_row(f"Model: {state.training.current_model or '-'}")
        current.add_row(f"Target: {state.training.current_target or '-'}")
        current.add_row(f"Fold: {fold}")
        current.add_row(f"Step: {state.training.current_step}")
        root.add_row(Panel(current, title="Current Step", box=box.SIMPLE))

        logs = "\n".join(state.recent_logs) or "No events yet"
        root.add_row(Panel(logs, title="Recent Events", box=box.SIMPLE))

    ours_predictions = _build_ours_predictions_table(state.ours_predictions)
    if ours_predictions is not None:
        root.add_row(Panel(ours_predictions, title="Ours Predictions", box=box.SIMPLE))
    return root


def _analysis_context_lines(context: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    if context.get("training_data_path"):
        lines.append(f"Training CSV: {context['training_data_path']}")
    if context.get("training_data_rows") is not None:
        lines.append(f"Training rows: {context['training_data_rows']}")
    if context.get("samples") is not None:
        lines.append(f"Current samples: {context['samples']}")
    if context.get("feature_count") is not None:
        lines.append(f"Features: {context['feature_count']}")
    if context.get("feature_type"):
        lines.append(f"Feature type: {context['feature_type']}")
    if context.get("model") or context.get("target"):
        lines.append(f"Context target: {context.get('model', '-')} / {context.get('target', '-')}")
    if context.get("output_dir"):
        lines.append(f"Output: {context['output_dir']}")

    operations = context.get("screen_operations")
    if isinstance(operations, list) and operations:
        lines.append(f"Screen operations: {' | '.join(str(operation) for operation in operations)}")
    return lines


def _build_ours_predictions_table(predictions: Dict[str, Dict[str, object]]):
    if not predictions:
        return None

    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("Model", style="bold", no_wrap=True)
    sample_labels = _ours_prediction_sample_labels(predictions)
    for sample in sample_labels:
        table.add_column(sample, no_wrap=True)

    for model, payload in predictions.items():
        rows = payload.get("rows", [])
        if not isinstance(rows, list) or not rows:
            table.add_row(model, *(["-"] * len(sample_labels)))
            continue
        values_by_sample = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            sample = row.get("sample", row.get("name", row.get("index", "-")))
            values_by_sample[str(sample)] = (
                f"{_format_prediction_compact(row.get('Predicted_wavelength'), decimals=0)}/"
                f"{_format_plqy_compact(row.get('Predicted_PLQY'))}"
            )
        table.add_row(model, *[values_by_sample.get(sample, "-") for sample in sample_labels])
    return table


def _ours_prediction_sample_labels(predictions: Dict[str, Dict[str, object]]) -> List[str]:
    labels: List[str] = []
    for payload in predictions.values():
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            sample = str(row.get("sample", row.get("name", row.get("index", "-"))))
            if sample not in labels:
                labels.append(sample)
            if len(labels) >= 6:
                return labels
    return labels or ["Prediction"]


def _format_prediction_value(value: object) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 10:
        return f"{numeric:.1f}"
    return f"{numeric:.4f}"


def _format_prediction_compact(value: object, decimals: int) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{decimals}f}"


def _format_plqy_compact(value: object) -> str:
    formatted = _format_prediction_compact(value, decimals=3)
    if formatted.startswith("0."):
        return formatted[1:]
    if formatted.startswith("-0."):
        return "-" + formatted[2:]
    return formatted


def load_events(path: Path, offset: int = 0) -> Tuple[int, List[Dict]]:
    """Read newly appended JSONL events from offset."""
    if not path.exists():
        return offset, []
    events: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return handle.tell(), events
