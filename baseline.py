"""
Deterministic baseline runner for Canary Release Manager.

Usage:
  python -m canary_release_env.baseline --url http://localhost:7860
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent
    parent = repo_root.parent
    for path in (repo_root, parent):
        candidate = str(path)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
        from client import CanaryEnv
        from server.canary_environment import CanaryEnvironment
        from server.policies import baseline_action
        from server.scenarios import PUBLIC_TASK_IDS
    except ModuleNotFoundError:
        from canary_release_env import CanaryEnv
        from canary_release_env.server.canary_environment import CanaryEnvironment
        from canary_release_env.server.policies import baseline_action
        from canary_release_env.server.scenarios import PUBLIC_TASK_IDS
else:
    from canary_release_env import CanaryEnv
    from canary_release_env.server.canary_environment import CanaryEnvironment
    from canary_release_env.server.policies import baseline_action
    from canary_release_env.server.scenarios import PUBLIC_TASK_IDS


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "").strip() or os.getenv("IMAGE_NAME", "").strip()


class _LocalEnvRunner:
    def __init__(self) -> None:
        self._env = CanaryEnvironment()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def reset(self, task: str, seed: int = 0):
        return self._env.reset(task=task, seed=seed)

    async def step(self, action):
        observation = self._env.step(action)
        return SimpleNamespace(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )


def _make_remote_runner(base_url: str):
    class _RemoteEnvRunner:
        def __init__(self, url: str) -> None:
            self._url = url
            self._env: CanaryEnv | None = None

        async def __aenter__(self):
            self._env = CanaryEnv(base_url=self._url)
            await self._env.connect()
            return self._env

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            if self._env is not None:
                await self._env.close()
            return False

    return _RemoteEnvRunner(base_url)


def _make_image_runner(local_image_name: str):
    class _ImageEnvRunner:
        def __init__(self, image: str) -> None:
            self._image = image
            self._env: CanaryEnv | None = None

        async def __aenter__(self):
            self._env = await CanaryEnv.from_docker_image(self._image)
            return self._env

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            if self._env is not None:
                await self._env.close()
            return False

    return _ImageEnvRunner(local_image_name)


async def run_task(env: CanaryEnv, task_id: str, seed: int = 0) -> dict:
    observation = await env.reset(task=task_id, seed=seed)
    actions: list[dict] = []
    rewards: list[float] = []

    while not observation.done:
        action = baseline_action(observation)
        actions.append(
            {
                "step_number": observation.step_number,
                "traffic_pct": observation.traffic_pct,
                "action_type": action.action_type,
            }
        )

        result = await env.step(action)
        rewards.append(float(result.reward or 0.0))
        observation = result.observation

    return {
        "task_id": task_id,
        "seed": seed,
        "score": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "steps": observation.step_number,
        "outcome": "success" if observation.traffic_pct >= 1.0 else "rollback",
        "actions": actions,
    }


async def main(base_url: str | None) -> None:
    all_scores: list[float] = []
    if base_url:
        env_runner = _make_remote_runner(base_url)
    elif LOCAL_IMAGE_NAME:
        env_runner = _make_image_runner(LOCAL_IMAGE_NAME)
    else:
        env_runner = _LocalEnvRunner()

    async with env_runner as env:
        for task_id in PUBLIC_TASK_IDS:
            result = await run_task(env, task_id)
            all_scores.append(result["score"])
            print(f"{task_id}: score={result['score']:.4f} steps={result['steps']} outcome={result['outcome']}")

    print(f"average={sum(all_scores) / len(all_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=None,
        help="Base URL for a running environment. If omitted, LOCAL_IMAGE_NAME is used when set; otherwise the script falls back to an in-process environment.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.url))
