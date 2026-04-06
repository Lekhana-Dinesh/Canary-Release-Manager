"""
Typed OpenEnv client for Canary Release Manager.
"""
from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

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

    async def reset(self, task: str = "easy") -> CanaryObservation:
        result = await super().reset(task=task)
        return result.observation
