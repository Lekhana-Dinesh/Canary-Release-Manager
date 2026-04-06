"""
Deterministic baseline runner for Canary Release Manager.

Usage:
  python -m canary_release_env.baseline --url http://localhost:7860
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canary_release_env import CanaryEnv
from canary_release_env.server.policies import baseline_action


async def run_task(env: CanaryEnv, task_id: str) -> dict:
    observation = await env.reset(task=task_id)
    actions: list[dict] = []

    while not observation.is_done:
        action = baseline_action(observation)
        actions.append(
            {
                "step_number": observation.step_number,
                "traffic_pct": observation.traffic_pct,
                "action_type": action.action_type,
            }
        )

        result = await env.step(action)
        observation = result.observation

    return {
        "task_id": task_id,
        "score": observation.cumulative_reward,
        "steps": observation.step_number,
        "outcome": "success" if observation.traffic_pct >= 1.0 else "rollback",
        "actions": actions,
    }


async def main(base_url: str) -> None:
    all_scores: list[float] = []
    async with CanaryEnv(base_url=base_url) as env:
        for task_id in ["easy", "medium", "hard"]:
            result = await run_task(env, task_id)
            all_scores.append(result["score"])
            print(f"{task_id}: score={result['score']:.4f} steps={result['steps']} outcome={result['outcome']}")

    print(f"average={sum(all_scores) / len(all_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://localhost:7860",
        help="Base URL for the running environment.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.url))
