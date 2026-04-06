"""
Canary Release Manager - Data Models

The models are the public contract for agents, validators, and reviewers.
"""
from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CanaryAction(Action):
    """
    A rollout decision for the current canary observation.
    """

    action_type: Literal[
        "increase_5",
        "increase_10",
        "increase_25",
        "hold",
        "rollback",
    ] = Field(
        ...,
        description=(
            "Traffic action for the canary deployment. Increase actions are only "
            "appropriate when the observed metrics are healthy."
        ),
    )
    reasoning: str = Field(
        ...,
        min_length=15,
        description=(
            "Short explanation referencing the metrics that motivated the decision."
        ),
    )
    slo_breach_detected: bool = Field(
        ...,
        description=(
            "Whether the agent believes the current observation reflects a real canary breach."
        ),
    )


class CanaryObservation(Observation):
    """
    Full metric snapshot plus scoring feedback for the last action.
    """

    traffic_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of total traffic currently routed to the canary.",
    )
    canary_error_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Observed error rate on the canary model.",
    )
    canary_p99_ms: float = Field(
        ...,
        ge=0.0,
        description="Observed p99 latency on the canary model in milliseconds.",
    )
    stable_error_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Observed error rate on the stable model baseline.",
    )
    stable_p99_ms: float = Field(
        ...,
        ge=0.0,
        description="Observed p99 latency on the stable model baseline.",
    )
    differential_error: float = Field(
        ...,
        description="canary_error_rate - stable_error_rate.",
    )
    differential_p99_ms: float = Field(
        ...,
        description="canary_p99_ms - stable_p99_ms.",
    )
    alert_count: int = Field(
        ...,
        ge=0,
        description="Number of active alerts for the currently observed state.",
    )

    step_number: int = Field(default=0, description="Steps taken in this episode.")
    step_reward: float = Field(
        default=0.0,
        description="Normalized score from the last action, always in [0.0, 1.0].",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Running average episode score so far, always in [0.0, 1.0].",
    )
    is_done: bool = Field(default=False, description="True when the episode has ended.")
    consecutive_holds: int = Field(
        default=0,
        description="How many hold actions have been taken in a row.",
    )
    actual_breach: bool = Field(
        default=False,
        description="Whether the pre-action observation represented a real canary breach.",
    )
    policy_assessment: str = Field(
        default="",
        description="Short label describing how the previous action was assessed.",
    )
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-component score breakdown for the previous action.",
    )
    step_explanation: str = Field(
        default="",
        description="Plain-English explanation of the previous action's score.",
    )

    task_id: str = Field(..., description="Task identifier, such as easy, medium, or hard.")
    task_description: str = Field(
        ...,
        description="Task instructions shown to the agent.",
    )
    feedback: str = Field(
        default="",
        description="Detailed feedback about what happened after the previous action.",
    )
