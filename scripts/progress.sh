#!/usr/bin/env bash

set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <experiment_dir>" >&2
    exit 2
fi

EXPERIMENT_DIR="$1"

if [[ ! -d "${EXPERIMENT_DIR}" ]]; then
    echo "Experiment directory not found: ${EXPERIMENT_DIR}" >&2
    exit 1
fi

python3 - "${EXPERIMENT_DIR}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    rounded = int(round(seconds))
    days, remainder = divmod(rounded, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


experiment_dir = Path(sys.argv[1]).resolve()
metadata_path = experiment_dir / "progress.json"
manifest_path = experiment_dir / "merged" / "manifest.json"

progress_metadata: dict = {}
if metadata_path.is_file():
    try:
        progress_metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid progress metadata {metadata_path}: {exc}")

manifest: dict = {}
if manifest_path.is_file():
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        manifest = {}

log_text_by_worker: dict[int, str] = {}
for log_path in sorted((experiment_dir / "logs").glob("worker-*.log")):
    match = re.fullmatch(r"worker-(\d+)\.log", log_path.name)
    if match:
        log_text_by_worker[int(match.group(1))] = log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )


def first_log_integer(pattern: str) -> int | None:
    for worker_id in sorted(log_text_by_worker):
        match = re.search(pattern, log_text_by_worker[worker_id])
        if match:
            return int(match.group(1))
    return None


num_runs = (
    progress_metadata.get("num_runs")
    or manifest.get("num_runs")
    or first_log_integer(r"^Global runs:\s*(\d+)\s*$")
)
num_turns = (
    progress_metadata.get("num_turns")
    or manifest.get("num_turns")
    or first_log_integer(r"^Turns per run:\s*(\d+)\s*$")
    or 10
)
num_workers = (
    progress_metadata.get("num_workers")
    or manifest.get("num_workers")
    or first_log_integer(r"^Worker:\s*\d+/(\d+)\s*$")
    or 7
)

if not isinstance(num_runs, int) or num_runs < 1:
    raise SystemExit(
        "Could not determine total runs from progress.json, manifest, or logs."
    )
if not isinstance(num_workers, int) or num_workers < 1:
    raise SystemExit("Could not determine num_workers.")

completed_by_worker: dict[int, list[dict]] = defaultdict(list)
all_completed: list[dict] = []
completed_run_ids: set[int] = set()
invalid_files: list[str] = []

for path in sorted(
    experiment_dir.glob("workers/worker-*/runs/run_*.json")
):
    try:
        data = read_json(path)
        if data.get("status") != "complete":
            raise ValueError("status is not complete")
        metadata = data["metadata"]
        run_id = int(data["run"])
        worker_id = int(metadata["worker_id"])
        if not 1 <= run_id <= num_runs:
            raise ValueError(f"run ID {run_id} is outside 1..{num_runs}")
        if run_id in completed_run_ids:
            raise ValueError(f"duplicate completed run ID {run_id}")
        expected_worker = (run_id - 1) % num_workers
        if worker_id != expected_worker:
            raise ValueError(
                f"run {run_id} belongs to worker {expected_worker}, "
                f"not worker {worker_id}"
            )
        started_at = datetime.fromisoformat(metadata["started_at"])
        completed_at = datetime.fromisoformat(metadata["completed_at"])
        item = {
            "run": run_id,
            "worker_id": worker_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration": (completed_at - started_at).total_seconds(),
        }
        completed_run_ids.add(run_id)
        completed_by_worker[worker_id].append(item)
        all_completed.append(item)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        invalid_files.append(f"{path}: {exc}")

for items in completed_by_worker.values():
    items.sort(key=lambda item: item["completed_at"])

completed_count = len(all_completed)
percent = min(100.0, completed_count * 100.0 / num_runs)

started_at_epoch = progress_metadata.get("started_at_epoch")
if isinstance(started_at_epoch, (int, float)):
    overall_start = datetime.fromtimestamp(started_at_epoch, timezone.utc)
elif all_completed:
    overall_start = min(item["started_at"] for item in all_completed)
else:
    overall_start = None

if completed_count >= num_runs and all_completed:
    overall_end = max(item["completed_at"] for item in all_completed)
else:
    overall_end = datetime.now(timezone.utc)

elapsed = (
    (overall_end - overall_start).total_seconds()
    if overall_start is not None
    else None
)

if completed_count >= num_runs:
    eta = 0.0
elif elapsed is not None and completed_count > 0:
    eta = elapsed * (num_runs - completed_count) / completed_count
else:
    eta = None

print(f"Experiment: {experiment_dir}")
print(
    f"Overall: {completed_count}/{num_runs} "
    f"({percent:6.2f}%)  "
    f"Elapsed: {format_duration(elapsed)}  "
    f"ETA: {format_duration(eta)}"
)
print()
print(f"{'Worker':<8} {'Current run':<20} {'Completed':<12} {'Last run':>10}")
print(f"{'-' * 8} {'-' * 20} {'-' * 12} {'-' * 10}")

for worker_id in range(num_workers):
    assigned_total = len(range(worker_id + 1, num_runs + 1, num_workers))
    completed_items = completed_by_worker.get(worker_id, [])
    completed = len(completed_items)
    last_duration = (
        completed_items[-1]["duration"] if completed_items else None
    )

    log_text = log_text_by_worker.get(worker_id, "")
    current_matches = re.findall(
        r"--- RUN (\d+) / TURN (\d+) ---",
        log_text,
    )

    if completed >= assigned_total:
        current = "done"
    elif current_matches:
        run_id, turn = current_matches[-1]
        completed_ids = {item["run"] for item in completed_items}
        if int(run_id) in completed_ids:
            current = "starting next"
        else:
            current = f"{run_id} (turn {turn}/{num_turns})"
    elif log_text:
        current = "starting"
    else:
        current = "waiting"

    print(
        f"{worker_id:<8} "
        f"{current:<20} "
        f"{completed:>4}/{assigned_total:<7} "
        f"{format_duration(last_duration):>10}"
    )

if invalid_files:
    print(file=sys.stderr)
    print(
        f"WARNING: {len(invalid_files)} invalid/incomplete run file(s) ignored:",
        file=sys.stderr,
    )
    for message in invalid_files[:10]:
        print(f"  {message}", file=sys.stderr)
    if len(invalid_files) > 10:
        print("  ...", file=sys.stderr)
PY
