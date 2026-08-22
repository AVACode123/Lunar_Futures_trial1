from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_FIELDS = [
    "turn",
    "us_china_tension",
    "shared_infrastructure",
    "scientific_openness",
    "mars_progress",
    "neutral_access",
    "us_power",
    "china_power",
    "us_public_cooperation",
    "china_public_cooperation",
    "third_force_strength",
    "international_trust",
]
CSV_FIELDS = [*STATE_FIELDS, "run", "event"]
EXPECTED_AGENTS = ["usa", "china", "japan"]


def derive_run_seed(master_seed: int, run_id: int) -> int:
    material = f"{master_seed}:{run_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and merge independently produced worker runs."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-runs", type=int, required=True)
    parser.add_argument("--num-turns", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=7)
    args = parser.parse_args()

    if args.num_runs < 1:
        parser.error("--num-runs must be at least 1")
    if args.num_turns < 1:
        parser.error("--num-turns must be at least 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    return args


def load_and_validate_run(
    path: Path,
    expected_run_id: int,
    num_turns: int,
    num_workers: int,
) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc

    metadata = data.get("metadata", {})
    expected_worker = (expected_run_id - 1) % num_workers

    if data.get("schema_version") != 2:
        raise ValueError(f"{path}: unsupported schema_version")
    if data.get("status") != "complete":
        raise ValueError(f"{path}: run is not complete")
    if data.get("run") != expected_run_id:
        raise ValueError(f"{path}: run ID does not match")
    if metadata.get("worker_id") != expected_worker:
        raise ValueError(
            f"{path}: worker ID is {metadata.get('worker_id')}, "
            f"expected {expected_worker}"
        )
    if metadata.get("num_workers") != num_workers:
        raise ValueError(f"{path}: num_workers does not match")
    if not metadata.get("ollama_url"):
        raise ValueError(f"{path}: Ollama URL is missing")
    if not isinstance(metadata.get("run_seed"), int):
        raise ValueError(f"{path}: run seed is missing")
    if not isinstance(metadata.get("master_seed"), int):
        raise ValueError(f"{path}: master seed is missing")
    if metadata["run_seed"] != derive_run_seed(
        metadata["master_seed"], expected_run_id
    ):
        raise ValueError(f"{path}: run seed is invalid")
    if not metadata.get("started_at") or not metadata.get("completed_at"):
        raise ValueError(f"{path}: start/end timestamp is missing")
    if "initial_state" not in data or "final_state" not in data:
        raise ValueError(f"{path}: initial_state or final_state is missing")

    turns = data.get("turns", [])
    if len(turns) != num_turns:
        raise ValueError(
            f"{path}: has {len(turns)} turns, expected {num_turns}"
        )

    for turn_number, turn in enumerate(turns, start=1):
        if turn.get("turn") != turn_number:
            raise ValueError(f"{path}: turn sequence is invalid")
        agents = [
            action.get("agent")
            for action in turn.get("actions", [])
        ]
        if agents != EXPECTED_AGENTS:
            raise ValueError(
                f"{path}: turn {turn_number} agent order is invalid"
            )
        if "ending_state" not in turn:
            raise ValueError(
                f"{path}: turn {turn_number} ending_state is missing"
            )

    if turns[-1]["ending_state"] != data["final_state"]:
        raise ValueError(f"{path}: final_state does not match final turn")

    for state_name, state in (
        ("initial_state", data["initial_state"]),
        ("final_state", data["final_state"]),
    ):
        missing = [field for field in STATE_FIELDS if field not in state]
        if missing:
            raise ValueError(f"{path}: {state_name} misses {missing}")

    return data


def state_row(
    state: dict[str, Any],
    run_id: int,
    event_id: str,
) -> dict[str, Any]:
    return {
        **{field: state[field] for field in STATE_FIELDS},
        "run": run_id,
        "event": event_id,
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main() -> None:
    args = parse_args()
    expected_ids = set(range(1, args.num_runs + 1))
    paths_by_run: dict[int, Path] = {}

    for path in sorted(args.input_root.glob("worker-*/runs/run_*.json")):
        try:
            run_id = int(path.stem.removeprefix("run_"))
        except ValueError as exc:
            raise ValueError(f"unexpected run filename: {path}") from exc
        if run_id in paths_by_run:
            raise ValueError(
                f"duplicate run {run_id}: {paths_by_run[run_id]} and {path}"
            )
        paths_by_run[run_id] = path

    found_ids = set(paths_by_run)
    missing = sorted(expected_ids - found_ids)
    unexpected = sorted(found_ids - expected_ids)
    if missing or unexpected:
        raise ValueError(
            f"run set is incomplete: missing={missing}, unexpected={unexpected}"
        )

    rows: list[dict[str, Any]] = []
    run_index: list[dict[str, Any]] = []
    models: set[str] = set()
    master_seeds: set[int] = set()

    for run_id in sorted(expected_ids):
        path = paths_by_run[run_id]
        data = load_and_validate_run(
            path,
            run_id,
            args.num_turns,
            args.num_workers,
        )
        metadata = data["metadata"]
        models.add(data["model"])
        master_seeds.add(metadata["master_seed"])

        rows.append(state_row(data["initial_state"], run_id, "initial"))
        for turn in data["turns"]:
            rows.append(
                state_row(
                    turn["ending_state"],
                    run_id,
                    turn["event"]["id"],
                )
            )

        run_index.append(
            {
                "run": run_id,
                "worker_id": metadata["worker_id"],
                "ollama_url": metadata["ollama_url"],
                "run_seed": metadata["run_seed"],
                "started_at": metadata["started_at"],
                "completed_at": metadata["completed_at"],
                "path": str(path.relative_to(args.input_root)),
            }
        )

    if len(models) != 1:
        raise ValueError(f"workers used different models: {sorted(models)}")
    if len(master_seeds) != 1:
        raise ValueError(
            f"workers used different master seeds: {sorted(master_seeds)}"
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "num_runs": args.num_runs,
        "num_turns": args.num_turns,
        "num_workers": args.num_workers,
        "state_rows": len(rows),
        "model": next(iter(models)),
        "master_seed": next(iter(master_seeds)),
        "runs": run_index,
    }

    all_states_path = args.output_dir / "all_states.csv"
    manifest_path = args.output_dir / "manifest.json"
    atomic_write_csv(all_states_path, rows)
    atomic_write_json(manifest_path, manifest)

    print("Merge validation passed.")
    print(f"Runs: {args.num_runs}")
    print(f"State rows: {len(rows)}")
    print(f"CSV: {all_states_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
