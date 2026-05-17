#!/usr/bin/env python3
"""Fixed-area training progress display helpers."""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Mapping, Optional, TextIO, Tuple


_ACTIVE_MONITOR: Optional["TrainingMonitor"] = None
METRIC_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("r2", "R2"),
    ("mae", "MAE"),
    ("rmse", "RMSE"),
)


def get_active_monitor() -> Optional["TrainingMonitor"]:
    """Return the currently active monitor, if training is running under TUI."""
    return _ACTIVE_MONITOR


def should_use_tui(env: Optional[Mapping[str, str]] = None, is_tty: Optional[bool] = None) -> bool:
    """Return whether the fixed-area TUI should be enabled."""
    env = env or os.environ
    mode = env.get("TRAIN_TUI", env.get("PHOSIR_TRAIN_TUI", "auto")).lower()
    if mode in {"0", "false", "no", "off"}:
        return False
    if is_tty is None:
        is_tty = sys.stdout.isatty()
    if not is_tty:
        return False
    return mode in {"1", "true", "yes", "on", "auto", ""}


@dataclass
class ProgressState:
    """Mutable state rendered by the training monitor."""

    models: List[str]
    targets: List[str]
    recent_limit: int = 10
    started_at: float = field(default_factory=time.perf_counter)
    current_model: Optional[str] = None
    current_target: Optional[str] = None
    current_fold: Optional[int] = None
    total_folds: Optional[int] = None
    current_step: str = "starting"
    model_status: Dict[str, str] = field(default_factory=dict)
    target_status: Dict[Tuple[str, str], str] = field(default_factory=dict)
    target_metrics: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    _recent_logs: Deque[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._recent_logs = deque(maxlen=self.recent_limit)
        for model in self.models:
            self.model_status.setdefault(model, "pending")
            for target in self.targets:
                self.target_status.setdefault((model, target), "pending")

    @property
    def recent_logs(self) -> List[str]:
        return list(self._recent_logs)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def completed_models(self) -> int:
        return sum(1 for status in self.model_status.values() if status.startswith("complete"))

    def log(self, message: str) -> None:
        for line in message.splitlines():
            clean = line.rstrip()
            if clean:
                self._recent_logs.append(clean)

    def model_started(self, model: str, index: int, total: int) -> None:
        self.current_model = model
        self.current_target = None
        self.current_fold = None
        self.total_folds = None
        self.current_step = f"model {index}/{total} started"
        self.model_status[model] = "running"
        self.log(f"[{index}/{total}] model started: {model}")

    def model_completed(self, model: str, elapsed: float) -> None:
        self.model_status[model] = f"complete {elapsed:.1f}s"
        self.current_step = f"model complete: {model}"
        self.log(f"model complete: {model} ({elapsed:.1f}s)")

    def model_failed(self, model: str, error: str) -> None:
        self.model_status[model] = "failed"
        self.current_step = f"model failed: {model}"
        self.log(f"model failed: {model}: {error}")

    def target_started(self, model: str, target: str, index: int, total: int) -> None:
        self.current_model = model
        self.current_target = target
        self.current_fold = None
        self.total_folds = None
        self.current_step = f"target {index}/{total} started"
        self.target_status[(model, target)] = "running"
        self.log(f"target started: {model} / {target}")

    def target_completed(
        self,
        model: str,
        target: str,
        elapsed: float,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        if metrics:
            self.target_metrics[(model, target)] = {
                key: float(value)
                for key, value in metrics.items()
                if _is_finite_metric(value)
            }
        self.target_status[(model, target)] = _format_target_completion(elapsed, metrics)
        self.current_step = f"target complete: {target}"
        self.log(f"target complete: {model} / {target} ({elapsed:.1f}s)")

    def fold_started(self, model: str, target: str, fold: int, total: int) -> None:
        self.current_model = model
        self.current_target = target
        self.current_fold = fold
        self.total_folds = total
        self.current_step = f"fold {fold}/{total} fitting"
        self.target_status[(model, target)] = f"fold {fold}/{total}"
        self.target_metrics.pop((model, target), None)

    def step(self, message: str) -> None:
        self.current_step = message
        self.log(message)


class ProgressStream:
    """File-like object that mirrors writes to a log and recent-events state."""

    def __init__(
        self,
        state: ProgressState,
        log_file: Optional[TextIO] = None,
        refresh: Optional[Callable[[], None]] = None,
    ):
        self.state = state
        self.log_file = log_file
        self.refresh = refresh

    def write(self, text: str) -> int:
        if self.log_file is not None:
            self.log_file.write(text)
        self.state.log(text)
        if self.refresh is not None and "\n" in text:
            self.refresh()
        return len(text)

    def flush(self) -> None:
        if self.log_file is not None:
            self.log_file.flush()

    def isatty(self) -> bool:
        return False


class TrainingMonitor:
    """Context manager for fixed-area Rich progress output with log capture."""

    def __init__(
        self,
        state: ProgressState,
        log_path: Optional[str] = None,
        enabled: bool = True,
        tick_interval: float = 1.0,
    ):
        self.state = state
        self.log_path = Path(log_path) if log_path else None
        self.enabled = enabled
        self.tick_interval = tick_interval
        self._log_handle: Optional[TextIO] = None
        self._live = None
        self._stdout_cm = None
        self._stderr_cm = None
        self._stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None

    def __enter__(self) -> "TrainingMonitor":
        global _ACTIVE_MONITOR
        _ACTIVE_MONITOR = self
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("a", encoding="utf-8")
        if self.enabled:
            try:
                from rich.live import Live

                self._live = Live(
                    self.render(),
                    refresh_per_second=4,
                    screen=True,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                self._live.__enter__()
                stream = ProgressStream(self.state, self._log_handle, refresh=self.update)
                self._stdout_cm = redirect_stdout(stream)
                self._stderr_cm = redirect_stderr(stream)
                self._stdout_cm.__enter__()
                self._stderr_cm.__enter__()
                self._start_heartbeat()
            except Exception:
                self.enabled = False
                self._live = None
                self._stdout_cm = nullcontext()
                self._stderr_cm = nullcontext()
                self._stdout_cm.__enter__()
                self._stderr_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _ACTIVE_MONITOR
        self._stop_event.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=2.0)
        if self._stderr_cm is not None:
            self._stderr_cm.__exit__(exc_type, exc, tb)
        if self._stdout_cm is not None:
            self._stdout_cm.__exit__(exc_type, exc, tb)
        if self._live is not None:
            self._live.update(self.render(), refresh=True)
            self._live.__exit__(exc_type, exc, tb)
        if self._log_handle is not None:
            self._log_handle.close()
        _ACTIVE_MONITOR = None

    def _start_heartbeat(self) -> None:
        if self.tick_interval <= 0:
            return
        self._refresh_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="training-monitor-heartbeat",
            daemon=True,
        )
        self._refresh_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.tick_interval):
            self.update()

    def update(self) -> None:
        if self._live is not None:
            self._live.update(self.render(), refresh=True)

    def model_started(self, model: str, index: int, total: int) -> None:
        self.state.model_started(model, index, total)
        self.update()

    def model_completed(self, model: str, elapsed: float) -> None:
        self.state.model_completed(model, elapsed)
        self.update()

    def model_failed(self, model: str, error: str) -> None:
        self.state.model_failed(model, error)
        self.update()

    def target_started(self, model: str, target: str, index: int, total: int) -> None:
        self.state.target_started(model, target, index, total)
        self.update()

    def target_completed(
        self,
        model: str,
        target: str,
        elapsed: float,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.state.target_completed(model, target, elapsed, metrics=metrics)
        self.update()

    def fold_started(self, model: str, target: str, fold: int, total: int) -> None:
        self.state.fold_started(model, target, fold, total)
        self.update()

    def step(self, message: str) -> None:
        self.state.step(message)
        self.update()

    def render(self):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        root = Table.grid(expand=True)
        root.add_column(ratio=1)

        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        header.add_row(
            Text("PhosIrDesign Training Monitor", style="bold cyan"),
            f"Elapsed: {_format_duration(self.state.elapsed)}",
        )
        header.add_row(
            f"Models: {self.state.completed_models}/{len(self.state.models)} complete",
            f"Current: {self.state.current_model or '-'}",
        )
        root.add_row(Panel(header, box=box.SIMPLE))

        table = build_training_metric_table(box=box.SIMPLE, expand=True)
        add_training_metric_columns(table, self.state.targets)
        for model in self.state.models:
            table.add_row(*build_training_metric_row(self.state, model))
        root.add_row(table)

        current = Table.grid(expand=True)
        current.add_column(ratio=1)
        fold = "-"
        if self.state.current_fold and self.state.total_folds:
            fold = f"{self.state.current_fold}/{self.state.total_folds}"
        current.add_row(f"Model: {self.state.current_model or '-'}")
        current.add_row(f"Target: {self.state.current_target or '-'}")
        current.add_row(f"Fold: {fold}")
        current.add_row(f"Step: {self.state.current_step}")
        root.add_row(Panel(current, title="Current Step", box=box.SIMPLE))

        logs = "\n".join(self.state.recent_logs[-self.state.recent_limit :]) or "No events yet"
        root.add_row(Panel(logs, title="Recent Events", box=box.SIMPLE))
        return root


def monitor_context(
    models: Iterable[str],
    targets: Iterable[str],
    log_path: Optional[str] = None,
    enabled: bool = True,
) -> TrainingMonitor:
    return TrainingMonitor(
        ProgressState(models=list(models), targets=list(targets)),
        log_path=log_path,
        enabled=enabled,
    )


def _format_duration(seconds: float) -> str:
    seconds_i = int(seconds)
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_training_metric_table(**kwargs):
    from rich.table import Table

    return Table(**kwargs)


def add_training_metric_columns(table, targets: Iterable[str]) -> None:
    table.add_column("Model", style="bold", no_wrap=True)
    for target in targets:
        short_target = _short_target_label(target)
        for _metric_key, metric_label in METRIC_COLUMNS:
            table.add_column(f"{short_target} {metric_label}", no_wrap=True)
    table.add_column("Status", no_wrap=True)


def build_training_metric_row(state: ProgressState, model: str) -> List[str]:
    row = [model]
    for target in state.targets:
        row.extend(_target_metric_cells(state, model, target))
    row.append(state.model_status.get(model, "pending"))
    return row


def _target_metric_cells(state: ProgressState, model: str, target: str) -> List[str]:
    metrics = state.target_metrics.get((model, target))
    if metrics:
        return [_format_metric_range(metrics, metric_key) for metric_key, _ in METRIC_COLUMNS]

    status = state.target_status.get((model, target), "pending")
    return [status] + [""] * (len(METRIC_COLUMNS) - 1)


def _format_metric_range(metrics: Mapping[str, float], metric: str) -> str:
    value = metrics.get(metric)
    if value is None and metric == "mse" and metrics.get("rmse") is not None:
        value = float(metrics["rmse"]) ** 2
    if value is None:
        return ""
    std = metrics.get(f"{metric}_std")
    return _format_value_with_optional_std(value, std)


def _format_value_with_optional_std(value: float, std: Optional[float] = None) -> str:
    formatted = _format_metric_value(value)
    if std is not None:
        formatted += f"+/-{_format_metric_value(std)}"
    return formatted


def _short_target_label(target: str) -> str:
    if target == "Max_wavelength(nm)":
        return "Wavelength"
    return target


def _format_target_completion(elapsed: float, metrics: Optional[Mapping[str, float]] = None) -> str:
    fallback_status = f"done {elapsed:.1f}s"
    if not metrics:
        return fallback_status

    r2 = metrics.get("r2")
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")

    parts = []
    if r2 is not None:
        parts.append(_format_metric_part("R2", r2, metrics.get("r2_std")))
    if mae is not None:
        parts.append(_format_metric_part("MAE", mae, metrics.get("mae_std")))
    if rmse is not None:
        parts.append(_format_metric_part("RMSE", rmse, metrics.get("rmse_std")))

    if not parts:
        return fallback_status
    return " ".join(parts)


def _format_metric_part(label: str, value: float, std: Optional[float] = None) -> str:
    return f"{label} {_format_value_with_optional_std(value, std)}"


def _is_finite_metric(value: float) -> bool:
    try:
        return value == value and value not in (float("inf"), float("-inf"))
    except TypeError:
        return False


def _format_metric_value(value: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_value = abs(numeric)
    if abs_value == 0 or abs_value < 10:
        return f"{numeric:.4f}"
    if abs_value < 100:
        return f"{numeric:.2f}"
    return f"{numeric:.1f}"
