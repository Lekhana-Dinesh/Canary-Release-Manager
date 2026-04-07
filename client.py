"""
Typed OpenEnv client for Canary Release Manager.
"""
from __future__ import annotations

import sys
from pathlib import Path

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent
    parent = repo_root.parent
    for path in (repo_root, parent):
        candidate = str(path)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
        from models import CanaryAction, CanaryObservation
    except ModuleNotFoundError:
        from canary_release_env.models import CanaryAction, CanaryObservation
else:
    from .models import CanaryAction, CanaryObservation


class CanaryEnv(EnvClient[CanaryAction, CanaryObservation, State]):
    action_type = CanaryAction
    observation_type = CanaryObservation

    def _step_payload(self, action: CanaryAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[CanaryObservation]:
        observation = CanaryObservation.model_validate(payload.get("observation", {}))
        observation.reward = payload.get("reward")
        observation.done = payload.get("done", False)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State.model_validate(payload)

    async def reset(self, task: str = "easy", seed: int | None = None) -> CanaryObservation:
        if seed is None:
            result = await super().reset(task=task)
        else:
            result = await super().reset(task=task, seed=seed)
        return result.observation
