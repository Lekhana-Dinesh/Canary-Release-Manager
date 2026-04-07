"""
Canary Release Manager scenario definitions.

Each task is deterministic and encodes a named rollout failure mode:
  - easy:   obvious immediate regression
  - medium: gradual degradation with a warning window before breach
  - hard:   shared infrastructure noise followed by a delayed canary-specific drift
  - expert: phantom monitoring alerts (false positives) followed by real differential drift
  - recovery: transient canary-specific degradation that recovers after cautious waiting
  - silent: canary-specific differential drift with weak alert support
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

SLO_ERROR_THRESHOLD = 0.005
SLO_P99_THRESHOLD = 200.0
SLO_DIFFERENTIAL_P99_THRESHOLD = 50.0

GLOBAL_NOISE_STEP = 3
GLOBAL_NOISE_P99_DELTA = 42.0

EXPERT_PHANTOM_ALERT_STEP = 2


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
    metric_fn: Callable[[float, int, int], MetricSnapshot]
    correct_rollback_at: float | None
    expected_baseline_score: tuple[float, float]
    expected_trained_score: tuple[float, float]
    failure_pattern: str
    rollback_on_error_rate: float
    rollback_on_canary_p99_ms: float | None
    rollback_on_differential_p99_ms: float | None


class HardProfile(NamedTuple):
    noise_step: int
    watch_traffic_pct: float
    breach_traffic_pct: float
    severe_breach_traffic_pct: float


class ExpertProfile(NamedTuple):
    phantom_step: int
    breach_traffic_pct: float
    phantom_alert_count: int


class RecoveryProfile(NamedTuple):
    anomaly_step: int
    anomaly_traffic_pct: float
    anomaly_duration_steps: int


class SilentProfile(NamedTuple):
    warning_traffic_pct: float
    breach_traffic_pct: float
    severe_breach_traffic_pct: float
    breach_after_warning_steps: int
    mixed_error_signal: bool


def _seed_choice(seed: int, values: tuple[float, ...], *, salt: int = 0) -> float:
    if seed == 0:
        return values[0]
    return values[(seed + salt) % len(values)]


def _seed_choice_int(seed: int, values: tuple[int, ...], *, salt: int = 0) -> int:
    if seed == 0:
        return values[0]
    return values[(seed + salt) % len(values)]


def hard_profile(seed: int) -> HardProfile:
    if seed == 0:
        return HardProfile(
            noise_step=GLOBAL_NOISE_STEP,
            watch_traffic_pct=0.30,
            breach_traffic_pct=0.35,
            severe_breach_traffic_pct=0.40,
        )

    profile_index = (seed + 47) % 3
    noise_step = (2, 3, 4)[profile_index]
    watch_traffic = (0.20, 0.30, 0.40)[profile_index]
    breach_traffic = (0.30, 0.35, 0.45)[profile_index]
    severe_breach_traffic = (0.35, 0.40, 0.50)[profile_index]
    return HardProfile(
        noise_step=noise_step,
        watch_traffic_pct=watch_traffic,
        breach_traffic_pct=breach_traffic,
        severe_breach_traffic_pct=severe_breach_traffic,
    )


def expert_profile(seed: int) -> ExpertProfile:
    if seed == 0:
        return ExpertProfile(
            phantom_step=EXPERT_PHANTOM_ALERT_STEP,
            breach_traffic_pct=0.35,
            phantom_alert_count=4,
        )

    return ExpertProfile(
        phantom_step=_seed_choice_int(seed, (1, 2, 3), salt=51),
        breach_traffic_pct=_seed_choice(seed, (0.30, 0.35, 0.40), salt=52),
        phantom_alert_count=_seed_choice_int(seed, (3, 4, 5), salt=53),
    )


def recovery_profile(seed: int) -> RecoveryProfile:
    if seed == 0:
        return RecoveryProfile(
            anomaly_step=3,
            anomaly_traffic_pct=0.30,
            anomaly_duration_steps=1,
        )

    profile_index = (seed + 61) % 3
    return RecoveryProfile(
        anomaly_step=(2, 3, 4)[profile_index],
        anomaly_traffic_pct=(0.20, 0.30, 0.40)[profile_index],
        anomaly_duration_steps=(1, 2, 1)[profile_index],
    )


def silent_profile(seed: int) -> SilentProfile:
    if seed == 0:
        return SilentProfile(
            warning_traffic_pct=0.30,
            breach_traffic_pct=0.40,
            severe_breach_traffic_pct=0.50,
            breach_after_warning_steps=1,
            mixed_error_signal=False,
        )

    profiles = (
        SilentProfile(0.30, 0.40, 0.50, 1, True),
        SilentProfile(0.30, 0.45, 0.55, 1, False),
        SilentProfile(0.30, 0.50, 0.60, 2, True),
        SilentProfile(0.30, 0.40, 0.50, 2, True),
    )
    return profiles[(seed + 79) % len(profiles)]


def _easy_metrics(traffic_pct: float, step_number: int, seed: int = 0) -> MetricSnapshot:
    del step_number

    base_error = 0.001 + _seed_choice(seed, (0.0, -0.00008, 0.00008), salt=1)
    error_slope = 0.12 + _seed_choice(seed, (0.0, -0.01, 0.01), salt=2)
    base_p99 = 140.0 + _seed_choice(seed, (0.0, -3.0, 4.0), salt=3)
    p99_slope = 800.0 + _seed_choice(seed, (0.0, -60.0, 60.0), salt=4)
    breach_anchor = _seed_choice(seed, (0.10, 0.15, 0.20), salt=5)

    canary_error = base_error + max(0.0, (traffic_pct - breach_anchor) * error_slope)
    canary_p99 = base_p99 + max(0.0, (traffic_pct - breach_anchor) * p99_slope)
    stable_error = 0.001
    stable_p99 = base_p99
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
    rollback_on_error_rate=SLO_ERROR_THRESHOLD,
    rollback_on_canary_p99_ms=SLO_P99_THRESHOLD,
    rollback_on_differential_p99_ms=None,
)


def _medium_metrics(traffic_pct: float, step_number: int, seed: int = 0) -> MetricSnapshot:
    del step_number

    base_error = 0.001 + _seed_choice(seed, (0.0, -0.00008, 0.00008), salt=11)
    error_slope = 0.06 + _seed_choice(seed, (0.0, -0.005, 0.005), salt=12)
    base_p99 = 138.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=13)
    p99_slope = 300.0 + _seed_choice(seed, (0.0, -25.0, 25.0), salt=14)
    drift_anchor = _seed_choice(seed, (0.25, 0.20, 0.30), salt=15)

    canary_error = base_error + max(0.0, (traffic_pct - drift_anchor) * error_slope)
    canary_p99 = base_p99 + max(0.0, (traffic_pct - drift_anchor) * p99_slope)
    stable_error = 0.001
    stable_p99 = base_p99
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
    rollback_on_error_rate=SLO_ERROR_THRESHOLD,
    rollback_on_canary_p99_ms=SLO_P99_THRESHOLD,
    rollback_on_differential_p99_ms=None,
)


def _hard_metrics(traffic_pct: float, step_number: int, seed: int = 0) -> MetricSnapshot:
    profile = hard_profile(seed)
    stable_error = 0.001
    stable_p99 = 144.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=21)
    canary_error = 0.001
    canary_p99 = stable_p99

    noise_delta = GLOBAL_NOISE_P99_DELTA + _seed_choice(seed, (0.0, -4.0, 4.0), salt=22)
    watch_error_base = 0.0026 + _seed_choice(seed, (0.0, -0.0002, 0.0002), salt=23)
    watch_error_step = 0.0025 + _seed_choice(seed, (0.0, -0.0003, 0.0003), salt=24)
    watch_p99_base = 168.0 + _seed_choice(seed, (0.0, -4.0, 4.0), salt=25)
    watch_p99_step = 30.0 + _seed_choice(seed, (0.0, -4.0, 4.0), salt=26)
    breach_error_base = 0.0046 + _seed_choice(seed, (0.0, -0.0002, 0.0002), salt=27)
    breach_error_traffic = 0.020 + _seed_choice(seed, (0.0, -0.004, 0.004), salt=28)
    breach_error_step = 0.0020 + _seed_choice(seed, (0.0, -0.0003, 0.0003), salt=29)
    breach_p99_base = 188.0 + _seed_choice(seed, (0.0, -4.0, 4.0), salt=30)
    breach_p99_traffic = 260.0 + _seed_choice(seed, (0.0, -30.0, 30.0), salt=31)
    breach_p99_step = 16.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=32)
    hard_breach_error_base = 0.0072 + _seed_choice(seed, (0.0, -0.0003, 0.0003), salt=33)
    hard_breach_error_traffic = 0.060 + _seed_choice(seed, (0.0, -0.008, 0.008), salt=34)
    hard_breach_error_step = 0.0015 + _seed_choice(seed, (0.0, -0.0002, 0.0002), salt=35)
    hard_breach_p99_base = 214.0 + _seed_choice(seed, (0.0, -6.0, 6.0), salt=36)
    hard_breach_p99_traffic = 340.0 + _seed_choice(seed, (0.0, -40.0, 40.0), salt=37)
    hard_breach_p99_step = 12.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=38)

    if traffic_pct >= profile.watch_traffic_pct and step_number == profile.noise_step:
        canary_p99 += noise_delta
        stable_p99 += noise_delta
    elif traffic_pct >= profile.severe_breach_traffic_pct:
        sustained_watch_steps = max(0, step_number - (profile.noise_step + 1))
        canary_error = (
            hard_breach_error_base
            + ((traffic_pct - profile.severe_breach_traffic_pct) * hard_breach_error_traffic)
            + (sustained_watch_steps * hard_breach_error_step)
        )
        canary_p99 = (
            hard_breach_p99_base
            + ((traffic_pct - profile.severe_breach_traffic_pct) * hard_breach_p99_traffic)
            + (sustained_watch_steps * hard_breach_p99_step)
        )
    elif traffic_pct >= profile.breach_traffic_pct:
        sustained_watch_steps = max(0, step_number - (profile.noise_step + 1))
        canary_error = (
            breach_error_base
            + ((traffic_pct - profile.breach_traffic_pct) * breach_error_traffic)
            + (sustained_watch_steps * breach_error_step)
        )
        canary_p99 = (
            breach_p99_base
            + ((traffic_pct - profile.breach_traffic_pct) * breach_p99_traffic)
            + (sustained_watch_steps * breach_p99_step)
        )
    elif traffic_pct >= profile.watch_traffic_pct and step_number >= profile.noise_step + 1:
        sustained_watch_steps = step_number - (profile.noise_step + 1)
        canary_error = watch_error_base + (sustained_watch_steps * watch_error_step)
        canary_p99 = watch_p99_base + (sustained_watch_steps * watch_p99_step)

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
    rollback_on_error_rate=SLO_ERROR_THRESHOLD,
    rollback_on_canary_p99_ms=None,
    rollback_on_differential_p99_ms=SLO_DIFFERENTIAL_P99_THRESHOLD,
)


def _expert_metrics(traffic_pct: float, step_number: int, seed: int = 0) -> MetricSnapshot:
    profile = expert_profile(seed)
    canary_error = 0.0018 + _seed_choice(seed, (0.0, -0.0002, 0.0002), salt=41)
    canary_p99 = 146.0 + _seed_choice(seed, (0.0, -3.0, 3.0), salt=42)
    stable_error = 0.001
    stable_p99 = 140.0 + _seed_choice(seed, (0.0, -1.0, 1.0), salt=43)
    alert_count = 0

    if traffic_pct >= 0.10 and step_number == profile.phantom_step:
        # Phantom alert storm: monitoring fires several alerts while all metrics remain healthy
        alert_count = profile.phantom_alert_count
    elif traffic_pct >= profile.breach_traffic_pct:
        # Real differential drift begins after phantom window clears
        error_slope = 0.032 + _seed_choice(seed, (0.0, -0.004, 0.004), salt=45)
        p99_slope = 560.0 + _seed_choice(seed, (0.0, -60.0, 60.0), salt=46)
        drift_anchor = max(0.10, profile.breach_traffic_pct - 0.10)
        canary_error = 0.001 + max(0.0, (traffic_pct - drift_anchor) * error_slope)
        canary_p99 = 140.0 + max(0.0, (traffic_pct - drift_anchor) * p99_slope)
        alert_count = int(
            canary_error > SLO_ERROR_THRESHOLD
            or (canary_p99 - stable_p99) > SLO_DIFFERENTIAL_P99_THRESHOLD
        )

    return MetricSnapshot(canary_error, canary_p99, stable_error, stable_p99, alert_count)


def _recovery_metrics(traffic_pct: float, step_number: int, seed: int = 0) -> MetricSnapshot:
    profile = recovery_profile(seed)
    stable_error = 0.001
    stable_p99 = 143.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=71)
    canary_error = 0.001
    canary_p99 = stable_p99
    alert_count = 0

    anomaly_step_end = profile.anomaly_step + profile.anomaly_duration_steps - 1
    if (
        traffic_pct >= profile.anomaly_traffic_pct
        and profile.anomaly_step <= step_number <= anomaly_step_end
    ):
        canary_error = 0.0027 + _seed_choice(seed, (0.0, -0.0002, 0.0002), salt=72)
        canary_p99 = 176.0 + _seed_choice(seed, (0.0, -6.0, 6.0), salt=73)
    elif traffic_pct >= profile.anomaly_traffic_pct and step_number > anomaly_step_end:
        canary_error = 0.0012 + _seed_choice(seed, (0.0, -0.0001, 0.0001), salt=74)
        canary_p99 = stable_p99 + _seed_choice(seed, (6.0, 8.0, 10.0), salt=75)

    return MetricSnapshot(canary_error, canary_p99, stable_error, stable_p99, alert_count)


def _silent_metrics(traffic_pct: float, step_number: int, seed: int = 0) -> MetricSnapshot:
    profile = silent_profile(seed)
    stable_error = 0.001
    stable_p99 = 145.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=81)
    canary_error = 0.001
    canary_p99 = stable_p99
    alert_count = 0

    warning_diff_base = 24.0 + _seed_choice(seed, (0.0, -4.0, 4.0), salt=82)
    warning_diff_step = 8.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=83)
    warning_error_base = (0.0026 if profile.mixed_error_signal else 0.0018) + _seed_choice(
        seed,
        (0.0, -0.0002, 0.0002),
        salt=84,
    )
    warning_error_step = 0.0004 if profile.mixed_error_signal else 0.00015

    breach_diff_base = 58.0 + _seed_choice(seed, (0.0, -6.0, 6.0), salt=85)
    breach_diff_traffic = 36.0 + _seed_choice(seed, (0.0, -6.0, 6.0), salt=86)
    breach_diff_step = 10.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=87)
    breach_error_base = (0.0046 if profile.mixed_error_signal else 0.0030) + _seed_choice(
        seed,
        (0.0, -0.0002, 0.0002),
        salt=88,
    )
    breach_error_traffic = 0.020 if profile.mixed_error_signal else 0.006
    breach_error_step = 0.0004 if profile.mixed_error_signal else 0.00015

    severe_diff_base = 88.0 + _seed_choice(seed, (0.0, -8.0, 8.0), salt=89)
    severe_diff_traffic = 48.0 + _seed_choice(seed, (0.0, -8.0, 8.0), salt=90)
    severe_diff_step = 12.0 + _seed_choice(seed, (0.0, -2.0, 2.0), salt=91)
    severe_error_base = (0.0068 if profile.mixed_error_signal else 0.0040) + _seed_choice(
        seed,
        (0.0, -0.0003, 0.0003),
        salt=92,
    )
    severe_error_traffic = 0.030 if profile.mixed_error_signal else 0.010
    severe_error_step = 0.0005 if profile.mixed_error_signal else 0.0002

    warning_steps = max(0, step_number - 3)
    severe_breach = traffic_pct >= profile.severe_breach_traffic_pct or (
        traffic_pct >= profile.warning_traffic_pct
        and warning_steps >= profile.breach_after_warning_steps + 2
    )
    confirmed_breach = traffic_pct >= profile.breach_traffic_pct or (
        traffic_pct >= profile.warning_traffic_pct
        and warning_steps >= profile.breach_after_warning_steps
    )

    if severe_breach:
        canary_error = (
            severe_error_base
            + (max(0.0, traffic_pct - profile.severe_breach_traffic_pct) * severe_error_traffic)
            + (warning_steps * severe_error_step)
        )
        canary_p99 = stable_p99 + (
            severe_diff_base
            + (max(0.0, traffic_pct - profile.severe_breach_traffic_pct) * severe_diff_traffic)
            + (warning_steps * severe_diff_step)
        )
    elif confirmed_breach:
        canary_error = (
            breach_error_base
            + (max(0.0, traffic_pct - profile.breach_traffic_pct) * breach_error_traffic)
            + (warning_steps * breach_error_step)
        )
        canary_p99 = stable_p99 + (
            breach_diff_base
            + (max(0.0, traffic_pct - profile.breach_traffic_pct) * breach_diff_traffic)
            + (warning_steps * breach_diff_step)
        )
    elif traffic_pct >= profile.warning_traffic_pct:
        canary_error = warning_error_base + (warning_steps * warning_error_step)
        canary_p99 = stable_p99 + warning_diff_base + (warning_steps * warning_diff_step)

    return MetricSnapshot(canary_error, canary_p99, stable_error, stable_p99, alert_count)


EXPERT = Scenario(
    id="expert",
    name="Phantom Alert Storm",
    failure_pattern=(
        "A monitoring false-positive fires at step 2 (alert_count=4) while all metrics remain healthy. "
        "After the phantom clears, a real canary-specific differential drift builds and breaches at 35% traffic."
    ),
    agent_instructions=(
        "A new recommendation model is rolling out. "
        "Rollback if canary_error_rate > 0.005 or canary_p99_ms - stable_p99_ms > 50. "
        "Early in the rollout a burst of monitoring alerts will fire even though the canary metrics are healthy. "
        "Do not rollback on alerts alone — verify the raw metrics before acting. "
        "After the alert storm passes, watch for real canary-versus-stable differential drift."
    ),
    difficulty="expert",
    metric_fn=_expert_metrics,
    correct_rollback_at=0.35,
    expected_baseline_score=(0.69, 0.71),
    expected_trained_score=(0.85, 0.92),
    rollback_on_error_rate=SLO_ERROR_THRESHOLD,
    rollback_on_canary_p99_ms=None,
    rollback_on_differential_p99_ms=SLO_DIFFERENTIAL_P99_THRESHOLD,
)


RECOVERY = Scenario(
    id="recovery",
    name="Transient hot shard recovery",
    failure_pattern=(
        "A canary-specific latency spike appears during rollout, stays below rollback guardrails, and then clears after a short watch period."
    ),
    agent_instructions=(
        "A new recommendation model is rolling out. "
        "Rollback if canary_error_rate > 0.005 or canary_p99_ms - stable_p99_ms > 50. "
        "This task includes a transient canary-specific degradation that should trigger a watchful hold, not an immediate rollback. "
        "If the differential recovers while staying below rollback guardrails, continue the rollout."
    ),
    difficulty="hard",
    metric_fn=_recovery_metrics,
    correct_rollback_at=None,
    expected_baseline_score=(0.90, 0.92),
    expected_trained_score=(0.93, 0.98),
    rollback_on_error_rate=SLO_ERROR_THRESHOLD,
    rollback_on_canary_p99_ms=None,
    rollback_on_differential_p99_ms=SLO_DIFFERENTIAL_P99_THRESHOLD,
)


SILENT = Scenario(
    id="silent",
    name="Silent differential burn",
    failure_pattern=(
        "Canary-specific differential latency drift builds with weak alert support, forcing rollback from raw telemetry rather than alert_count."
    ),
    agent_instructions=(
        "A new recommendation model is rolling out. "
        "Rollback if canary_error_rate > 0.005 or canary_p99_ms - stable_p99_ms > 50. "
        "This task intentionally provides weak alert support, so do not wait for alert_count to confirm a canary-specific regression. "
        "Use raw canary-versus-stable telemetry to decide when to hold and when to rollback."
    ),
    difficulty="hard",
    metric_fn=_silent_metrics,
    correct_rollback_at=0.40,
    expected_baseline_score=(0.80, 0.86),
    expected_trained_score=(0.91, 0.97),
    rollback_on_error_rate=SLO_ERROR_THRESHOLD,
    rollback_on_canary_p99_ms=None,
    rollback_on_differential_p99_ms=SLO_DIFFERENTIAL_P99_THRESHOLD,
)


SCENARIOS: dict[str, Scenario] = {
    "easy": EASY,
    "medium": MEDIUM,
    "hard": HARD,
    "expert": EXPERT,
    "recovery": RECOVERY,
    "silent": SILENT,
}

PUBLIC_TASK_IDS: tuple[str, ...] = tuple(SCENARIOS)
