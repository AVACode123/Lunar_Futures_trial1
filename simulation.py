from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import requests


MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

NUM_TURNS = 5


WORLD_STATE = {
    "turn": 0,
    "us_china_tension": 60,
    "shared_infrastructure": 15,
    "scientific_openness": 30,
    "mars_progress": 20,
    "neutral_access": 20
}


AVAILABLE_ACTIONS = [
    "claim_resource_zone",
    "expand_surveillance",
    "propose_joint_mining",
    "propose_shared_rescue_network",
    "share_scientific_data",
    "invest_in_mars_program",
    "form_coalition",
    "wait_and_observe"
]


def load_agents() -> list[dict[str, Any]]:
    with open("agents.json", "r", encoding="utf-8") as f:
        return json.load(f)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "options": {
                "temperature": 0.8
            }
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["message"]["content"]


def build_prompt(
    agent: dict[str, Any],
    world_state: dict[str, Any]
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

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside JSON.
"""

    user_prompt = f"""
Current lunar world state:

{json.dumps(world_state, indent=2)}

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


def parse_agent_response(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        print("\nWARNING: Model returned invalid JSON:")
        print(raw_text)

        return {
            "public_statement": "No valid statement generated.",
            "action": "wait_and_observe",
            "target": "none",
            "reason": "Invalid JSON response."
        }


def apply_action(
    state: dict[str, Any],
    agent_id: str,
    action: dict[str, Any]
) -> None:

    name = action["action"]

    if name == "claim_resource_zone":
        state["us_china_tension"] += 5
        state["neutral_access"] -= 3

    elif name == "expand_surveillance":
        state["us_china_tension"] += 4
        state["shared_infrastructure"] -= 1

    elif name == "propose_joint_mining":
        state["us_china_tension"] -= 3
        state["shared_infrastructure"] += 3
        state["neutral_access"] += 2

    elif name == "propose_shared_rescue_network":
        state["us_china_tension"] -= 2
        state["shared_infrastructure"] += 5

    elif name == "share_scientific_data":
        state["scientific_openness"] += 5
        state["us_china_tension"] -= 1

    elif name == "invest_in_mars_program":
        state["mars_progress"] += 5

    elif name == "form_coalition":
        state["shared_infrastructure"] += 2
        state["neutral_access"] += 3

        if agent_id == "japan":
            state["us_china_tension"] -= 1

    elif name == "wait_and_observe":
        pass

    clamp_state(state)


def clamp_state(state: dict[str, Any]) -> None:
    for key, value in state.items():
        if key == "turn":
            continue

        if isinstance(value, (int, float)):
            state[key] = max(0, min(100, value))


def save_run(log: dict[str, Any]) -> str:
    os.makedirs("runs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"runs/run_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return filename


def main():
    agents = load_agents()

    world_state = WORLD_STATE.copy()

    run_log = {
        "model": MODEL,
        "initial_state": world_state.copy(),
        "turns": []
    }

    print("\n=== Lunar Futures Simulation ===\n")

    for turn in range(1, NUM_TURNS + 1):
        print(f"\n--- TURN {turn} ---")

        world_state["turn"] = turn

        turn_log = {
            "turn": turn,
            "starting_state": world_state.copy(),
            "actions": []
        }

        for agent in agents:
            print(f"\n{agent['name']} is thinking...")

            system_prompt, user_prompt = build_prompt(
                agent,
                world_state
            )

            raw_response = call_llm(
                system_prompt,
                user_prompt
            )

            action = parse_agent_response(raw_response)

            print(
                f"{agent['name']} -> "
                f"{action['action']} "
                f"(target: {action['target']})"
            )

            print(
                f'Statement: {action["public_statement"]}'
            )

            apply_action(
                world_state,
                agent["id"],
                action
            )

            turn_log["actions"].append({
                "agent": agent["id"],
                "decision": action,
                "state_after_action": world_state.copy()
            })

        turn_log["ending_state"] = world_state.copy()
        run_log["turns"].append(turn_log)

        print("\nWorld state:")
        print(json.dumps(
            world_state,
            indent=2,
            ensure_ascii=False
        ))

    run_log["final_state"] = world_state.copy()

    filename = save_run(run_log)

    print("\n=== Simulation complete ===")
    print(f"Saved to: {filename}")


if __name__ == "__main__":
    main()