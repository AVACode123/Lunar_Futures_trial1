from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Any

import pandas as pd
import requests


# ============================================================
# Basic settings
# ============================================================

MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

NUM_RUNS = 100
NUM_TURNS = 10


# ============================================================
# Initial world state
# ============================================================

WORLD_STATE = {
    "turn": 0,

    # Lunar geopolitical state
    "us_china_tension": 60,
    "shared_infrastructure": 15,
    "scientific_openness": 30,
    "mars_progress": 20,
    "neutral_access": 20,

    # Earth-side factors
    "us_power": 80,
    "china_power": 78,
    "us_public_cooperation": 40,
    "china_public_cooperation": 35,

    # Potential for a third force
    "third_force_strength": 10,

    # General international trust
    "international_trust": 30,
}


# ============================================================
# Actions available to the agents
# ============================================================

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


# ============================================================
# Random events
# ============================================================

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


# ============================================================
# Utility functions
# ============================================================

def load_agents() -> list[dict[str, Any]]:
    with open("agents.json", "r", encoding="utf-8") as f:
        return json.load(f)


def clamp_state(state: dict[str, Any]) -> None:
    for key, value in state.items():
        if key == "turn":
            continue

        if isinstance(value, (int, float)):
            state[key] = max(0, min(100, value))


# ============================================================
# Event agent
# ============================================================

def event_agent() -> dict[str, Any]:
    roll = random.random()

    # 80%: no major external event
    if roll < 0.80:
        return {
            "id": "none",
            "description": "No major external event.",
            "effects": {},
        }

    # 18%: one predefined stochastic event
    if roll < 0.98:
        return random.choice(EVENTS)

    # 2%: rare unknown event
    return {
        "id": "rare_unknown",
        "description": (
            "An unexpected event occurs that was not anticipated "
            "by the simulation designers."
        ),
        "effects": {
            "international_trust": random.randint(-8, 8),
            "third_force_strength": random.randint(-5, 10),
            "us_china_tension": random.randint(-10, 10),
            "scientific_openness": random.randint(-5, 8),
            "mars_progress": random.randint(-3, 10),
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


# ============================================================
# LLM call
# ============================================================

def call_llm(
    system_prompt: str,
    user_prompt: str,
) -> str:

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
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

# ============================================================
# Prompt builder
# ============================================================

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


# ============================================================
# Parse LLM response
# ============================================================

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


# ============================================================
# Apply actions to the world
# ============================================================

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


# ============================================================
# Earth-side slow dynamics
# ============================================================

def update_earth_factors(
    state: dict[str, Any],
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
    state["us_power"] += random.choice([-1, 0, 0, 0, 1])
    state["china_power"] += random.choice([-1, 0, 0, 0, 1])

    clamp_state(state)


# ============================================================
# One simulation run
# ============================================================

def run_single_simulation(
    run_id: int,
    agents: list[dict[str, Any]],
    all_states: list[dict[str, Any]],
) -> dict[str, Any]:

    world_state = WORLD_STATE.copy()

    run_log = {
        "run": run_id,
        "model": MODEL,
        "initial_state": world_state.copy(),
        "turns": [],
    }

    # Save initial state
    initial_record = world_state.copy()
    initial_record["run"] = run_id
    initial_record["event"] = "initial"
    all_states.append(initial_record)

    print(
        f"\n"
        f"==============================\n"
        f"RUN {run_id}\n"
        f"=============================="
    )

    for turn in range(1, NUM_TURNS + 1):

        world_state["turn"] = turn

        print(f"\n--- RUN {run_id} / TURN {turn} ---")

        # ----------------------------------------------------
        # Event occurs
        # ----------------------------------------------------

        event = event_agent()

        apply_event(
            world_state,
            event,
        )

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

        # ----------------------------------------------------
        # Agents act
        # ----------------------------------------------------

        for agent in agents:

            print(
                f"{agent['name']} is thinking..."
            )

            system_prompt, user_prompt = build_prompt(
                agent,
                world_state,
                event,
            )

            raw_response = call_llm(
                system_prompt,
                user_prompt,
            )

            action = parse_agent_response(
                raw_response,
            )

            print(
                f"{agent['name']} -> "
                f"{action['action']}"
            )

            apply_action(
                world_state,
                agent["id"],
                action,
            )

            turn_log["actions"].append(
                {
                    "agent": agent["id"],
                    "decision": action,
                    "state_after_action": world_state.copy(),
                }
            )

        # ----------------------------------------------------
        # Earth-side slow changes
        # ----------------------------------------------------

        update_earth_factors(
            world_state,
        )

        # ----------------------------------------------------
        # Save the state for PCA
        # ----------------------------------------------------

        state_record = world_state.copy()

        state_record["run"] = run_id
        state_record["event"] = event["id"]

        all_states.append(
            state_record
        )

        turn_log["ending_state"] = world_state.copy()

        run_log["turns"].append(
            turn_log
        )

        print(
            "State:",
            json.dumps(
                world_state,
                ensure_ascii=False,
            ),
        )

    run_log["final_state"] = world_state.copy()

    return run_log


# ============================================================
# Save results
# ============================================================

def save_run(
    run_log: dict[str, Any],
) -> str:

    os.makedirs(
        "runs",
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"runs/run_{run_log['run']}_{timestamp}.json"
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_log,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return filename


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        "runs",
        exist_ok=True,
    )

    agents = load_agents()

    all_states: list[dict[str, Any]] = []

    print("\n=== Lunar Futures Batch Simulation ===")
    print(f"Model: {MODEL}")
    print(f"Runs: {NUM_RUNS}")
    print(f"Turns per run: {NUM_TURNS}")

    for run_id in range(
        1,
        NUM_RUNS + 1,
    ):

        run_log = run_single_simulation(
            run_id,
            agents,
            all_states,
        )

        filename = save_run(
            run_log,
        )

        print(
            f"Saved run {run_id}: {filename}"
        )

    # --------------------------------------------------------
    # Save all state vectors for PCA / UMAP
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_states
    )

    csv_filename = (
        "runs/all_states.csv"
    )

    df.to_csv(
        csv_filename,
        index=False,
    )

    print(
        "\n=============================="
    )
    print(
        "Batch simulation complete."
    )
    print(
        f"State data saved to: {csv_filename}"
    )
    print(
        "=============================="
    )


if __name__ == "__main__":
    main()