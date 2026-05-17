#!/usr/bin/env python3
"""Fixed-area TUI for the full workflow progress JSONL stream."""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from phosirdesign.utils.workflow_progress import (  # noqa: E402
    DEFAULT_WORKFLOW_STEPS,
    WorkflowProgressState,
    load_events,
    render_workflow,
)


def parse_csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render PhosIrDesign workflow progress.")
    parser.add_argument("--events", required=True, help="Progress JSONL event file")
    parser.add_argument("--models", default="", help="Comma-separated model list")
    parser.add_argument("--targets", default="", help="Comma-separated target list")
    parser.add_argument("--tick", type=float, default=1.0, help="Refresh interval in seconds")
    parser.add_argument(
        "--exit-on-finish",
        action="store_true",
        help="Exit automatically when the workflow_finished event is observed",
    )
    args = parser.parse_args()

    event_path = Path(args.events)
    state = WorkflowProgressState(
        steps=DEFAULT_WORKFLOW_STEPS,
        models=parse_csv(args.models),
        targets=parse_csv(args.targets),
    )
    offset = 0
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    from rich.live import Live

    with Live(render_workflow(state), refresh_per_second=4, screen=True) as live:
        while running:
            offset, events = load_events(event_path, offset)
            for event in events:
                state.apply_event(event)
            live.update(render_workflow(state), refresh=True)
            if args.exit_on_finish and state.finished:
                break
            time.sleep(args.tick)

        offset, events = load_events(event_path, offset)
        for event in events:
            state.apply_event(event)
        live.update(render_workflow(state), refresh=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
