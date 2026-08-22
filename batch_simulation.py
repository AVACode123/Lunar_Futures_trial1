from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_NUM_RUNS = 100
DEFAULT_NUM_TURNS = 10
DEFAULT_MASTER_SEED = 20260821
DEFAULT_OUTPUT_DIR = "runs"


@dataclass(frozen=True)
class Config:
    ollama_url: str
    model: str
    num_runs: int
    num_turns: int
    master_seed: int
    output_dir: Path
    worker_id: int
    num_workers: int


WORLD_STATE = {
    "turn": 0,
    "us_china_tension": 60,
    "shared_infrastructure": 15,
    "scientific_openness": 30,
    "mars_progress": 20,
    "neutral_access": 20,
    "us_power": 80,
    "china_power": 78,
    "us_public_cooperation": 40,
    "china_public_cooperation": 35,
    "third_force_strength": 10,
    "international_trust": 30,
}


AVAILABLE_ACTIONS = [
    "claim_resource_zone",
    "expand_surveillance",
    "propose_joint_mining",
    "propose_shared_rescue_network",
    "share_scientific_data",
    "invest_in_mars_program",
    "form_coalition",
    "wait_and_observe",
]


EVENTS = [
    {
        "id": "chinese_base_accident",
        "description": (
            "A serious life-support failure occurs "
            "at a Chinese lunar base."
        ),
        "effects": {
            "us_china_tension": -2,
            "international_trust": 1,
        },
    },
    {
        "id": "us_base_accident",
        "description": (
            "A serious accident occurs at a US lunar facility."
        ),
        "effects": {
            "us_china_tension": -2,
            "international_trust": 1,
        },
    },
    {
        "id": "new_ice_discovery",
        "description": (
            "A major new deposit of accessible lunar water ice "
            "is discovered near the south pole."
        ),
        "effects": {
            "neutral_access": 5,
            "us_china_tension": 2,
        },
    },
    {
        "id": "earth_tension",
        "description": (
            "US-China political tensions suddenly rise on Earth."
        ),
        "effects": {
            "us_china_tension": 8,
            "international_trust": -4,
            "us_public_cooperation": -4,
            "china_public_cooperation": -4,
        },
    },
    {
        "id": "japanese_breakthrough",
        "description": (
            "Japan demonstrates a major breakthrough in "
            "closed-loop life-support technology."
        ),
        "effects": {
            "third_force_strength": 8,
            "shared_infrastructure": 3,
            "international_trust": 2,
        },
    },
    {
        "id": "india_transport_success",
        "description": (
            "India successfully demonstrates low-cost "
            "lunar cargo transportation."
        ),
        "effects": {
            "third_force_strength": 5,
            "neutral_access": 4,
        },
    },
    {
        "id": "scientific_discovery",
        "description": (
            "A scientifically extraordinary discovery is made "
            "near the lunar south pole."
        ),
        "effects": {
            "scientific_openness": 6,
            "international_trust": 2,
        },
    },
    {
        "id": "mars_breakthrough",
        "description": (
            "A major technological breakthrough makes "
            "future Mars missions significantly more feasible."
        ),
        "effects": {
            "mars_progress": 10,
        },
    },
]


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {value!r}") from exc


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run an independently restartable simulation worker."
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=env_int("NUM_RUNS", DEFAULT_NUM_RUNS),
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=env_int("NUM_TURNS", DEFAULT_NUM_TURNS),
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=env_int("MASTER_SEED", DEFAULT_MASTER_SEED),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=env_int("WORKER_ID", 0),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=env_int("NUM_WORKERS", 1),
    )
    args = parser.parse_args()

    if args.num_runs < 1:
        parser.error("--num-runs must be at least 1")
    if args.num_turns < 1:
        parser.error("--num-turns must be at least 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if not 0 <= args.worker_id < args.num_workers:
        parser.error("--worker-id must satisfy 0 <= worker-id < num-workers")

    return Config(
        ollama_url=args.ollama_url,
        model=args.model,
        num_runs=args.num_runs,
        num_turns=args.num_turns,
        master_seed=args.master_seed,
        output_dir=args.output_dir,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_run_seed(master_seed: int, run_id: int) -> int:
    material = f"{master_seed}:{run_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def assigned_run_ids(config: Config) -> list[int]:
    return [
        run_id
        for run_id in range(1, config.num_runs + 1)
        if (run_id - 1) % config.num_workers == config.worker_id
    ]


def load_agents() -> list[dict[str, Any]]:
    agents_path = Path(__file__).resolve().parent / "agents.json"
    with agents_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp_state(state: dict[str, Any]) -> None:
    for key, value in state.items():
        if key == "turn":
            continue
        if isinstance(value, (int, float)):
            state[key] = max(0, min(100, value))


def event_agent(rng: random.Random) -> dict[str, Any]:
    roll = rng.random()

    # 80%: no major external event
    if roll < 0.80:
        return {
            "id": "none",
            "description": "No major external event.",
            "effects": {},
        }

    # 18%: one predefined stochastic event
    if roll < 0.98:
        return rng.choice(EVENTS)

    # 2%: rare unknown event
    return {
        "id": "rare_unknown",
        "description": (
            "An unexpected event occurs that was not anticipated "
            "by the simulation designers."
        ),
        "effects": {
            "international_trust": rng.randint(-8, 8),
            "third_force_strength": rng.randint(-5, 10),
            "us_china_tension": rng.randint(-10, 10),
            "scientific_openness": rng.randint(-5, 8),
            "mars_progress": rng.randint(-3, 10),
        },
    }


def apply_event(
    state: dict[str, Any],
    event: dict[str, Any],
) -> None:
    for key, change in event["effects"].items():
        if key in state:
            state[key] += change
    clamp_state(state)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    config: Config,
) -> str:
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = requests.post(
                config.ollama_url,
                json={
                    "model": config.model,
                    "stream": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    "options": {
                        "temperature": 0.9,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(
                f"LLM error "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )

    print(
        "LLM failed after retries. "
        "Using wait_and_observe."
    )
    return json.dumps(
        {
            "public_statement": (
                "No action due to communication failure."
            ),
            "action": "wait_and_observe",
            "target": "none",
            "reason": "LLM request failed.",
        }
    )


def build_prompt(
    agent: dict[str, Any],
    world_state: dict[str, Any],
    event: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = f"""
You are an autonomous policy agent representing {agent["name"]}.

Public goal:
{agent["public_goal"]}

Private strategic goal:
{agent["hidden_goal"]}

Behavioral style:
{agent["style"]}

You must act consistently with these goals.

You should consider both lunar conditions and Earth-side political factors.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside JSON.
"""

    user_prompt = f"""
Current world state:

{json.dumps(world_state, indent=2)}

Current external event:

{event["description"]}

You may choose exactly one action from this list:

{json.dumps(AVAILABLE_ACTIONS, indent=2)}

Choose one action for this turn.

Return JSON in exactly this structure:

{{
  "public_statement": "short public statement",
  "action": "one action from the allowed list",
  "target": "usa, china, japan, all, or none",
  "reason": "brief strategic reason"
}}
"""
    return system_prompt, user_prompt


def parse_agent_response(
    raw_text: str,
) -> dict[str, Any]:
    try:
        result = json.loads(raw_text)
        if result.get("action") not in AVAILABLE_ACTIONS:
            raise ValueError("Invalid action")
        return result
    except Exception:
        print("\nWARNING: Invalid model response:")
        print(raw_text)
        return {
            "public_statement": "No valid statement generated.",
            "action": "wait_and_observe",
            "target": "none",
            "reason": "Invalid JSON response.",
        }


def apply_action(
    state: dict[str, Any],
    agent_id: str,
    action: dict[str, Any],
) -> None:
    name = action["action"]

    if name == "claim_resource_zone":
        state["us_china_tension"] += 5
        state["neutral_access"] -= 3
        state["international_trust"] -= 2
    elif name == "expand_surveillance":
        state["us_china_tension"] += 4
        state["shared_infrastructure"] -= 1
        state["international_trust"] -= 1
    elif name == "propose_joint_mining":
        state["us_china_tension"] -= 3
        state["shared_infrastructure"] += 3
        state["neutral_access"] += 2
        state["international_trust"] += 2
    elif name == "propose_shared_rescue_network":
        state["us_china_tension"] -= 2
        state["shared_infrastructure"] += 5
        state["international_trust"] += 2
    elif name == "share_scientific_data":
        state["scientific_openness"] += 5
        state["us_china_tension"] -= 1
        state["international_trust"] += 3
    elif name == "invest_in_mars_program":
        state["mars_progress"] += 5
    elif name == "form_coalition":
        state["shared_infrastructure"] += 2
        state["neutral_access"] += 3
        state["third_force_strength"] += 4

        if agent_id == "japan":
            state["us_china_tension"] -= 1
            state["international_trust"] += 2
    elif name == "wait_and_observe":
        pass

    clamp_state(state)


def update_earth_factors(
    state: dict[str, Any],
    rng: random.Random,
) -> None:
    # High lunar tension slowly reduces willingness to cooperate
    if state["us_china_tension"] > 70:
        state["us_public_cooperation"] -= 1
        state["china_public_cooperation"] -= 1

    # Strong shared infrastructure can slowly increase cooperation
    if state["shared_infrastructure"] > 40:
        state["us_public_cooperation"] += 1
        state["china_public_cooperation"] += 1

    # Strong international trust promotes cooperation
    if state["international_trust"] > 50:
        state["us_public_cooperation"] += 1
        state["china_public_cooperation"] += 1

    # Small random drift in national power
    state["us_power"] += rng.choice([-1, 0, 0, 0, 1])
    state["china_power"] += rng.choice([-1, 0, 0, 0, 1])
    clamp_state(state)


def run_single_simulation(
    run_id: int,
    agents: list[dict[str, Any]],
    config: Config,
) -> dict[str, Any]:
    run_seed = derive_run_seed(config.master_seed, run_id)
    rng = random.Random(run_seed)
    world_state = WORLD_STATE.copy()

    run_log = {
        "schema_version": 2,
        "status": "running",
        "run": run_id,
        "model": config.model,
        "metadata": {
            "worker_id": config.worker_id,
            "num_workers": config.num_workers,
            "ollama_url": config.ollama_url,
            "master_seed": config.master_seed,
            "run_seed": run_seed,
            "started_at": utc_now(),
            "completed_at": None,
        },
        "initial_state": world_state.copy(),
        "turns": [],
    }

    print(
        f"\n"
        f"==============================\n"
        f"RUN {run_id} (seed={run_seed})\n"
        f"=============================="
    )

    for turn in range(1, config.num_turns + 1):
        world_state["turn"] = turn
        print(f"\n--- RUN {run_id} / TURN {turn} ---")

        event = event_agent(rng)
        apply_event(world_state, event)

        if event["id"] != "none":
            print(
                f"EVENT: {event['id']} "
                f"- {event['description']}"
            )

        turn_log = {
            "turn": turn,
            "event": event,
            "starting_state": world_state.copy(),
            "actions": [],
        }

        # Preserve the existing USA -> China -> Japan order.
        for agent in agents:
            print(f"{agent['name']} is thinking...")
            system_prompt, user_prompt = build_prompt(
                agent,
                world_state,
                event,
            )
            raw_response = call_llm(
                system_prompt,
                user_prompt,
                config,
            )
            action = parse_agent_response(raw_response)
            print(f"{agent['name']} -> {action['action']}")
            apply_action(world_state, agent["id"], action)
            turn_log["actions"].append(
                {
                    "agent": agent["id"],
                    "decision": action,
                    "state_after_action": world_state.copy(),
                }
            )

        update_earth_factors(world_state, rng)
        turn_log["ending_state"] = world_state.copy()
        run_log["turns"].append(turn_log)

        print(
            "State:",
            json.dumps(world_state, ensure_ascii=False),
        )

    run_log["final_state"] = world_state.copy()
    run_log["metadata"]["completed_at"] = utc_now()
    run_log["status"] = "complete"
    return run_log


def run_path(config: Config, run_id: int) -> Path:
    return config.output_dir / f"run_{run_id:06d}.json"


def validate_completed_run(
    path: Path,
    run_id: int,
    config: Config,
) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read completed run: {exc}"

    expected_agents = ["usa", "china", "japan"]
    metadata = data.get("metadata", {})

    if data.get("status") != "complete":
        return False, "status is not complete"
    if data.get("run") != run_id:
        return False, "run ID does not match filename"
    if data.get("model") != config.model:
        return False, "model differs from current configuration"
    if metadata.get("master_seed") != config.master_seed:
        return False, "master seed differs from current configuration"
    if metadata.get("run_seed") != derive_run_seed(config.master_seed, run_id):
        return False, "run seed is invalid"
    if metadata.get("worker_id") != config.worker_id:
        return False, "worker ID differs from current configuration"
    if metadata.get("num_workers") != config.num_workers:
        return False, "num_workers differs from current configuration"
    if metadata.get("ollama_url") != config.ollama_url:
        return False, "Ollama URL differs from current configuration"
    if not metadata.get("started_at") or not metadata.get("completed_at"):
        return False, "start/end timestamp is missing"
    if len(data.get("turns", [])) != config.num_turns:
        return False, "turn count differs from current configuration"
    if "final_state" not in data:
        return False, "final_state is missing"

    for expected_turn, turn in enumerate(data["turns"], start=1):
        if turn.get("turn") != expected_turn:
            return False, f"turn {expected_turn} is missing or out of order"
        actual_agents = [
            action.get("agent")
            for action in turn.get("actions", [])
        ]
        if actual_agents != expected_agents:
            return False, f"turn {expected_turn} has an invalid agent order"

    return True, "complete"


def atomic_save_run(run_log: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary_path = Path(f.name)
            json.dump(
                run_log,
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        with temporary_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("status") != "complete":
            raise ValueError("temporary run file is not complete")

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    agents = load_agents()
    run_ids = assigned_run_ids(config)

    print("\n=== Lunar Futures Batch Simulation Worker ===")
    print(f"Model: {config.model}")
    print(f"Ollama URL: {config.ollama_url}")
    print(f"Global runs: {config.num_runs}")
    print(f"Turns per run: {config.num_turns}")
    print(f"Master seed: {config.master_seed}")
    print(f"Worker: {config.worker_id}/{config.num_workers}")
    print(f"Assigned runs: {run_ids}")
    print(f"Output directory: {config.output_dir}")

    completed = 0
    skipped = 0

    for run_id in run_ids:
        path = run_path(config, run_id)

        if path.exists():
            valid, reason = validate_completed_run(
                path,
                run_id,
                config,
            )
            if valid:
                print(f"Skipping completed run {run_id}: {path}")
                skipped += 1
                continue
            raise RuntimeError(
                f"Refusing to overwrite invalid run file {path}: {reason}"
            )

        run_log = run_single_simulation(
            run_id,
            agents,
            config,
        )
        atomic_save_run(run_log, path)
        completed += 1
        print(f"Saved run {run_id}: {path}")

    print("\n==============================")
    print("Worker complete.")
    print(f"New runs: {completed}")
    print(f"Skipped runs: {skipped}")
    print("==============================")


if __name__ == "__main__":
    main()
