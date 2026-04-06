"""
Canary Release Manager - FastAPI application.

OpenEnv provides the standard reset/step/state/ws endpoints. WebSocket sessions
are the canonical persistent interface. The /episodes endpoints are additional
stateful HTTP helpers for plain REST integrations.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Any
from pathlib import Path

import uvicorn
from fastapi import HTTPException
from openenv.core.env_server import create_app
from pydantic import BaseModel, Field

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from server._compat import bootstrap_import_paths
else:
    from ._compat import bootstrap_import_paths

bootstrap_import_paths()

USE_PACKAGE_IMPORTS = (__package__ or "").startswith("canary_release_env.server")

if USE_PACKAGE_IMPORTS:
    from canary_release_env.models import CanaryAction, CanaryObservation
    from canary_release_env.server.canary_environment import CanaryEnvironment
    from canary_release_env.server.grader import grade
    from canary_release_env.server.policies import baseline_action
    from canary_release_env.server.scenarios import (
        GLOBAL_NOISE_STEP,
        SCENARIOS,
        SLO_DIFFERENTIAL_P99_THRESHOLD,
        SLO_ERROR_THRESHOLD,
        SLO_P99_THRESHOLD,
    )
else:
    from models import CanaryAction, CanaryObservation
    from server.canary_environment import CanaryEnvironment
    from server.grader import grade
    from server.policies import baseline_action
    from server.scenarios import (
        GLOBAL_NOISE_STEP,
        SCENARIOS,
        SLO_DIFFERENTIAL_P99_THRESHOLD,
        SLO_ERROR_THRESHOLD,
        SLO_P99_THRESHOLD,
    )

app = create_app(
    CanaryEnvironment,
    CanaryAction,
    CanaryObservation,
    env_name="canary_release_env",
)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


MAX_COMPLETED_EPISODES = _env_int("MAX_COMPLETED_EPISODES", 128)

_episodes: dict[str, CanaryEnvironment] = {}
_completed_episode_results: OrderedDict[str, dict[str, Any]] = OrderedDict()


class CreateEpisodeRequest(BaseModel):
    task: str = "easy"


class EpisodeStepRequest(BaseModel):
    action: dict


class GraderRequest(BaseModel):
    task_id: str
    action: dict
    traffic_pct: float = Field(..., ge=0.0, le=1.0)
    canary_error_rate: float = Field(..., ge=0.0, le=1.0)
    canary_p99_ms: float = Field(..., ge=0.0)
    stable_error_rate: float = Field(..., ge=0.0, le=1.0)
    stable_p99_ms: float = Field(..., ge=0.0)
    alert_count: int = Field(default=0, ge=0)
    step_number: int = Field(default=0, ge=0)
    consecutive_holds: int = Field(default=0, ge=0)


def _get_episode(episode_id: str) -> CanaryEnvironment:
    env = _episodes.get(episode_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found. Call POST /episodes to start one.",
        )
    return env


def _store_completed_episode(env: CanaryEnvironment) -> dict[str, Any]:
    result = env.episode_result()
    _completed_episode_results[result["episode_id"]] = result
    while len(_completed_episode_results) > MAX_COMPLETED_EPISODES:
        _completed_episode_results.popitem(last=False)
    return result


@app.post("/episodes", tags=["Episodes"])
def create_episode(request: CreateEpisodeRequest = CreateEpisodeRequest()) -> dict[str, Any]:
    if request.task not in SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{request.task}'. Valid task_ids: {', '.join(SCENARIOS)}",
        )

    env = CanaryEnvironment()
    observation = env.reset(task=request.task)
    episode_id = env.state.episode_id
    _episodes[episode_id] = env
    return {
        "episode_id": episode_id,
        "observation": observation.model_dump(),
        "done": observation.is_done,
    }


@app.post("/episodes/{episode_id}/step", tags=["Episodes"])
def step_episode(episode_id: str, request: EpisodeStepRequest) -> dict[str, Any]:
    env = _get_episode(episode_id)
    try:
        action = CanaryAction(**request.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}") from exc

    observation = env.step(action)
    response: dict[str, Any] = {
        "observation": observation.model_dump(),
        "reward": observation.step_reward,
        "done": observation.is_done,
    }
    if observation.is_done:
        response["episode_result"] = _store_completed_episode(env)
        del _episodes[episode_id]
    return response


@app.get("/episodes/{episode_id}/state", tags=["Episodes"])
def get_episode_state(episode_id: str) -> dict[str, Any]:
    env = _get_episode(episode_id)
    return env.state.model_dump()


@app.get("/episodes/{episode_id}/transcript", tags=["Episodes"])
def get_episode_transcript(episode_id: str) -> dict[str, Any]:
    if episode_id in _episodes:
        env = _episodes[episode_id]
        return env.episode_result()
    if episode_id in _completed_episode_results:
        return _completed_episode_results[episode_id]
    raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.")


@app.get("/tasks", tags=["Environment Info"])
def list_tasks() -> dict[str, Any]:
    return {
        "environment": "Canary Release Manager",
        "description": (
            "OpenEnv environment for AI agents that manage canary releases for a new model version."
        ),
        "tasks": [
            {
                "id": scenario.id,
                "name": scenario.name,
                "failure_pattern": scenario.failure_pattern,
                "difficulty": scenario.difficulty,
                "expected_baseline_score": list(scenario.expected_baseline_score),
                "expected_trained_score": list(scenario.expected_trained_score),
                "correct_rollback_at": scenario.correct_rollback_at,
            }
            for scenario in SCENARIOS.values()
        ],
        "slo_thresholds": {
            "error_rate": SLO_ERROR_THRESHOLD,
            "p99_ms": SLO_P99_THRESHOLD,
            "differential_p99_ms": SLO_DIFFERENTIAL_P99_THRESHOLD,
        },
        "action_schema": CanaryAction.model_json_schema(),
        "observation_schema": CanaryObservation.model_json_schema(),
        "scoring_weights": {
            "breach_detection_score": 0.35,
            "rollback_timing_score": 0.25,
            "promotion_safety_score": 0.30,
            "reasoning_score": 0.10,
        },
        "notes": {
            "canonical_stateful_interface": "WebSocket /ws via OpenEnv clients",
            "extra_stateful_http_interface": "/episodes",
            "global_noise_step": GLOBAL_NOISE_STEP,
            "episode_score_definition": "running average of normalized step scores",
            "standard_http_reset_step_state": "validator-safe but stateless across separate HTTP requests",
            "completed_episode_retention": f"{MAX_COMPLETED_EPISODES} recent completed episode results kept in memory",
            "invalid_task_behavior": "/episodes rejects unknown task ids with 422; standard /reset defaults unknown tasks to easy and says so in feedback",
        },
    }


@app.post("/grader", tags=["Evaluation"])
def run_grader(request: GraderRequest) -> dict[str, Any]:
    scenario = SCENARIOS.get(request.task_id)
    if scenario is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{request.task_id}'. Valid task_ids: {', '.join(SCENARIOS)}",
        )

    try:
        action = CanaryAction(**request.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action format: {exc}") from exc

    result = grade(
        action=action,
        traffic_pct=request.traffic_pct,
        canary_error=request.canary_error_rate,
        canary_p99=request.canary_p99_ms,
        stable_error=request.stable_error_rate,
        stable_p99=request.stable_p99_ms,
        alert_count=request.alert_count,
        scenario=scenario,
        step_number=request.step_number,
        consecutive_holds=request.consecutive_holds,
    )

    return {
        "task_id": request.task_id,
        "total_score": result.total_score,
        "max_score": 1.0,
        "actual_breach": result.actual_breach,
        "policy_assessment": result.policy_assessment,
        "reward_breakdown": result.reward_breakdown,
        "breach_detection_score": result.reward_breakdown["breach_detection_score"],
        "rollback_timing_score": result.reward_breakdown["rollback_timing_score"],
        "promotion_safety_score": result.reward_breakdown["promotion_safety_score"],
        "reasoning_score": result.reward_breakdown["reasoning_score"],
        "explanation": result.explanation,
        "feedback": result.feedback_parts,
        "deterministic": True,
        "normalized": True,
    }


@app.post("/baseline", tags=["Evaluation"])
def run_baseline() -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}

    for task_id in SCENARIOS:
        env = CanaryEnvironment()
        observation = env.reset(task=task_id)

        while not observation.is_done:
            action = baseline_action(observation)
            observation = env.step(action)

        episode_result = env.episode_result()
        results[task_id] = {
            "score": round(episode_result["episode_score"], 4),
            "steps": episode_result["steps"],
            "outcome": episode_result["outcome"],
            "score_breakdown": episode_result["score_breakdown"],
            "rollback_assessment": episode_result["rollback_assessment"],
            "first_breach_point": episode_result["first_breach_point"],
            "episode_summary": episode_result["episode_summary"],
            "transcript": episode_result["transcript"],
        }

    scores = {task_id: result["score"] for task_id, result in results.items()}
    average = round(sum(scores.values()) / len(scores), 4)
    return {
        "agent_type": "rule-based-rollout-v3",
        "description": (
            "Deterministic rule baseline: hold on explicit shared noise, rollback on confirmed breaches, otherwise increase by 10%. "
            "It intentionally ignores warning windows so medium and hard remain discriminative."
        ),
        "scores": scores,
        "average": average,
        "details": results,
        "score_contract": {
            "per_step_score_range": [0.0, 1.0],
            "episode_score_range": [0.0, 1.0],
            "episode_score_definition": "running average of normalized step scores",
        },
        "normalized": True,
    }


def main() -> None:
    port = _env_int("PORT", 7860)
    uvicorn.run(
        "canary_release_env.server.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
