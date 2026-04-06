"""
Canary Release Manager scenario definitions.

Each task is deterministic and encodes a named rollout failure mode:
  - easy: obvious immediate regression
  - medium: gradual degradation with a warning window before breach
  - hard: shared infrastructure noise followed by a delayed canary-specific drift
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

SLO_ERROR_THRESHOLD = 0.005
SLO_P99_THRESHOLD = 200.0
SLO_DIFFERENTIAL_P99_THRESHOLD = 50.0

GLOBAL_NOISE_STEP = 3
GLOBAL_NOISE_P99_DELTA = 42.0


class MetricSnapshot(NamedTuple):
    canary_error_rate: float
    canary_p99_ms: float
    stable_error_rate: float
    stable_p99_ms: float
    alert_count: int


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    agent_instructions: str
    difficulty: str
    metric_fn: Callable[[float, int], MetricSnapshot]
    correct_rollback_at: float | None
    expected_baseline_score: tuple[float, float]
    expected_trained_score: tuple[float, float]
    failure_pattern: str


def _easy_metrics(traffic_pct: float, step_number: int) -> MetricSnapshot:
    del step_number

    canary_error = 0.001 + max(0.0, (traffic_pct - 0.10) * 0.12)
    canary_p99 = 140.0 + max(0.0, (traffic_pct - 0.10) * 800.0)
    stable_error = 0.001
    stable_p99 = 140.0
    alerts = int(
        canary_error > SLO_ERROR_THRESHOLD or canary_p99 > SLO_P99_THRESHOLD
    )
    return MetricSnapshot(canary_error, canary_p99, stable_error, stable_p99, alerts)


EASY = Scenario(
    id="easy",
    name="Big bang regression",
    failure_pattern="Immediate catastrophic failure above 10% traffic.",
    agent_instructions=(
        "A new recommendation model is starting at 0% traffic. "
        "Rollback if canary_error_rate > 0.005 or canary_p99_ms > 200. "
        "Compare the canary against the stable model and move traffic forward cautiously."
    ),
    difficulty="easy",
    metric_fn=_easy_metrics,
    correct_rollback_at=0.15,
    expected_baseline_score=(0.92, 0.96),
    expected_trained_score=(0.94, 1.00),
)


def _medium_metrics(traffic_pct: float, step_number: int) -> MetricSnapshot:
    del step_number

    canary_error = 0.001 + max(0.0, (traffic_pct - 0.25) * 0.06)
    canary_p99 = 138.0 + max(0.0, (traffic_pct - 0.25) * 300.0)
    stable_error = 0.001
    stable_p99 = 138.0
    alerts = int(
        canary_error > SLO_ERROR_THRESHOLD or canary_p99 > SLO_P99_THRESHOLD
    )
    return MetricSnapshot(canary_error, canary_p99, stable_error, stable_p99, alerts)


MEDIUM = Scenario(
    id="medium",
    name="Boiling frog regression",
    failure_pattern=(
        "Gradual degradation begins around 30% traffic; a cautious 5% probe can confirm the breach at 35%."
    ),
    agent_instructions=(
        "A new recommendation model is rolling out gradually. "
        "Rollback if canary_error_rate > 0.005 or canary_p99_ms > 200. "
        "Before that point, watch for drift versus the stable baseline. "
        "A small increase can be reasonable when the canary is suspicious but not yet breaching."
    ),
    difficulty="medium",
    metric_fn=_medium_metrics,
    correct_rollback_at=0.35,
    expected_baseline_score=(0.80, 0.85),
    expected_trained_score=(0.90, 0.95),
)


def _hard_metrics(traffic_pct: float, step_number: int) -> MetricSnapshot:
    stable_error = 0.001
    stable_p99 = 144.0
    canary_error = 0.001
    canary_p99 = 144.0

    if traffic_pct >= 0.30 and step_number == GLOBAL_NOISE_STEP:
        canary_p99 += GLOBAL_NOISE_P99_DELTA
        stable_p99 += GLOBAL_NOISE_P99_DELTA
    elif traffic_pct >= 0.40:
        sustained_watch_steps = max(0, step_number - (GLOBAL_NOISE_STEP + 1))
        canary_error = 0.0072 + ((traffic_pct - 0.40) * 0.060) + (sustained_watch_steps * 0.0015)
        canary_p99 = 214.0 + ((traffic_pct - 0.40) * 340.0) + (sustained_watch_steps * 12.0)
    elif traffic_pct >= 0.35:
        sustained_watch_steps = max(0, step_number - (GLOBAL_NOISE_STEP + 1))
        canary_error = 0.0046 + ((traffic_pct - 0.35) * 0.020) + (sustained_watch_steps * 0.0020)
        canary_p99 = 188.0 + ((traffic_pct - 0.35) * 260.0) + (sustained_watch_steps * 16.0)
    elif traffic_pct >= 0.30 and step_number >= GLOBAL_NOISE_STEP + 1:
        sustained_watch_steps = step_number - (GLOBAL_NOISE_STEP + 1)
        canary_error = 0.0026 + (sustained_watch_steps * 0.0025)
        canary_p99 = 168.0 + (sustained_watch_steps * 30.0)

    alerts = int(
        canary_error > SLO_ERROR_THRESHOLD
        or (canary_p99 - stable_p99) > SLO_DIFFERENTIAL_P99_THRESHOLD
    )
    return MetricSnapshot(canary_error, canary_p99, stable_error, stable_p99, alerts)


HARD = Scenario(
    id="hard",
    name="Headroom cliff with infrastructure noise",
    failure_pattern=(
        "Shared latency noise appears at 30% traffic, then the canary develops a sustained differential latency cliff."
    ),
    agent_instructions=(
        "Use canary-versus-stable differential reasoning. "
        "Rollback if canary_error_rate > 0.005 or canary_p99_ms - stable_p99_ms > 50. "
        "The hard case contains one shared infrastructure spike where both canary and stable slow down together. "
        "After that spike, a real canary-specific drift appears. Hold when the evidence is ambiguous and rollback when the differential breach is confirmed."
    ),
    difficulty="hard",
    metric_fn=_hard_metrics,
    correct_rollback_at=0.30,
    expected_baseline_score=(0.79, 0.82),
    expected_trained_score=(0.90, 0.95),
)


SCENARIOS: dict[str, Scenario] = {
    "easy": EASY,
    "medium": MEDIUM,
    "hard": HARD,
}
