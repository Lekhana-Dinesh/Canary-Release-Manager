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
            "Short explanation kept for transcript clarity and reviewer inspection. "
            "The public grader is driven by the structured fields, not by prose quality."
        ),
    )
    slo_breach_detected: bool = Field(
        ...,
        description=(
            "Whether the agent believes the current observation reflects a real canary breach."
        ),
    )
    state_assessment: Literal[
        "healthy",
        "warning",
        "noise",
        "phantom_alert",
        "breach",
    ] = Field(
        ...,
        description=(
            "Structured coarse-grained classification of the observed rollout state. "
            "Use 'healthy' for safe promotion, 'warning' for ambiguous pre-breach drift or watch states, "
            "'noise' for shared infrastructure distortion, 'phantom_alert' for false-positive monitoring noise, "
            "and 'breach' for a confirmed rollback condition. The public contract intentionally stays coarse; "
            "the grader may use richer internal subtypes without exposing them in the action schema."
        ),
    )


class CanaryObservation(Observation):
    """
    Agent-facing rollout telemetry plus operational guardrails.
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
    consecutive_holds: int = Field(
        default=0,
        description="How many hold actions have been taken in a row.",
    )
    step_number: int = Field(
        default=0,
        description="Current rollout decision index for this episode.",
    )
    rollback_on_error_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rollback guardrail for canary_error_rate.",
    )
    rollback_on_canary_p99_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Rollback guardrail for absolute canary p99 latency when applicable.",
    )
    rollback_on_differential_p99_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Rollback guardrail for canary-vs-stable p99 latency differential when applicable.",
    )
