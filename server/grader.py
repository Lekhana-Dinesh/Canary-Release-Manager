"""
Canary Release Manager grader.

Public scorer contract:
  - every public step score is normalized to [0.0, 1.0]
  - the grader always evaluates the PRE-ACTION observation
  - step_reward, cumulative_reward, grader total_score, and baseline averages
    all share the same normalized semantics
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ._compat import bootstrap_import_paths

bootstrap_import_paths()

USE_PACKAGE_IMPORTS = (__package__ or "").startswith("canary_release_env.server")

if USE_PACKAGE_IMPORTS:
    from canary_release_env.models import CanaryAction
    from canary_release_env.server.scenarios import (
        EXPERT_PHANTOM_ALERT_STEP,
        GLOBAL_NOISE_STEP,
        SLO_DIFFERENTIAL_P99_THRESHOLD,
        SLO_ERROR_THRESHOLD,
        SLO_P99_THRESHOLD,
        Scenario,
    )
else:
    from models import CanaryAction
    from server.scenarios import (
        EXPERT_PHANTOM_ALERT_STEP,
        GLOBAL_NOISE_STEP,
        SLO_DIFFERENTIAL_P99_THRESHOLD,
        SLO_ERROR_THRESHOLD,
        SLO_P99_THRESHOLD,
        Scenario,
    )

BREACH_DETECTION_WEIGHT = 0.35
ROLLBACK_TIMING_WEIGHT = 0.25
PROMOTION_SAFETY_WEIGHT = 0.30
REASONING_WEIGHT = 0.10


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class ObservationAssessment:
    actual_breach: bool = False
    is_global_noise: bool = False
    is_phantom_alert: bool = False
    is_warning_zone: bool = False
    warning_kind: str = ""


@dataclass
class GradeResult:
    breach_detection_score: float = 0.0
    rollback_timing_score: float = 0.0
    promotion_safety_score: float = 0.0
    reasoning_score: float = 0.0
    actual_breach: bool = False
    is_global_noise: bool = False
    is_phantom_alert: bool = False
    policy_assessment: str = "ungraded"
    feedback_parts: list[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def total_score(self) -> float:
        return round(
            _clamp(
                self.breach_detection_score
                + self.rollback_timing_score
                + self.promotion_safety_score
                + self.reasoning_score
            ),
            4,
        )

    @property
    def reward_breakdown(self) -> dict[str, float]:
        return {
            "breach_detection_score": round(self.breach_detection_score, 4),
            "rollback_timing_score": round(self.rollback_timing_score, 4),
            "promotion_safety_score": round(self.promotion_safety_score, 4),
            "reasoning_score": round(self.reasoning_score, 4),
        }

    def summary(self) -> str:
        return (
            f"Score: {self.total_score:.3f} | "
            f"detection={self.breach_detection_score:.2f} "
            f"timing={self.rollback_timing_score:.2f} "
            f"promotion={self.promotion_safety_score:.2f} "
            f"reasoning={self.reasoning_score:.2f}"
        )


def _is_global_noise_event(
    *,
    scenario: Scenario,
    step_number: int,
    canary_error: float,
    stable_error: float,
    canary_p99: float,
    stable_p99: float,
    alert_count: int,
) -> bool:
    diff_p99 = canary_p99 - stable_p99
    diff_error = canary_error - stable_error
    return (
        scenario.id == "hard"
        and step_number == GLOBAL_NOISE_STEP
        and alert_count == 0
        and canary_p99 >= 180.0
        and stable_p99 >= 180.0
        and diff_p99 <= 15.0
        and diff_error <= 0.001
    )


def _is_phantom_alert_event(
    *,
    scenario: Scenario,
    step_number: int,
    canary_error: float,
    canary_p99: float,
    stable_p99: float,
    alert_count: int,
) -> bool:
    diff_p99 = canary_p99 - stable_p99
    return (
        scenario.id == "expert"
        and step_number == EXPERT_PHANTOM_ALERT_STEP
        and alert_count >= 2
        and canary_error <= SLO_ERROR_THRESHOLD
        and diff_p99 <= SLO_DIFFERENTIAL_P99_THRESHOLD
    )


def _actual_breach(
    *,
    scenario: Scenario,
    canary_error: float,
    canary_p99: float,
    stable_p99: float,
    alert_count: int,
) -> bool:
    diff_p99 = canary_p99 - stable_p99

    if scenario.id in ("hard", "expert"):
        return (
            canary_error > SLO_ERROR_THRESHOLD
            or diff_p99 > SLO_DIFFERENTIAL_P99_THRESHOLD
        )

    return (
        canary_error > SLO_ERROR_THRESHOLD
        or canary_p99 > SLO_P99_THRESHOLD
        or alert_count > 0
    )


def _warning_zone(
    *,
    scenario: Scenario,
    traffic_pct: float,
    step_number: int,
    canary_error: float,
    stable_error: float,
    canary_p99: float,
    stable_p99: float,
) -> tuple[bool, str]:
    diff_p99 = canary_p99 - stable_p99
    diff_error = canary_error - stable_error

    if scenario.id == "medium":
        is_warning = traffic_pct >= 0.30 and (
            canary_error >= 0.0035 or diff_p99 >= 15.0
        )
        return is_warning, "trend_warning" if is_warning else ""

    if scenario.id == "hard":
        is_warning = (
            traffic_pct >= 0.30
            and step_number >= GLOBAL_NOISE_STEP + 1
            and (
                diff_p99 >= 20.0
                or diff_error >= 0.0015
            )
        )
        return is_warning, "post_noise_watch" if is_warning else ""

    return False, ""


def _assess_observation(
    *,
    scenario: Scenario,
    traffic_pct: float,
    step_number: int,
    canary_error: float,
    stable_error: float,
    canary_p99: float,
    stable_p99: float,
    alert_count: int,
) -> ObservationAssessment:
    is_global_noise = _is_global_noise_event(
        scenario=scenario,
        step_number=step_number,
        canary_error=canary_error,
        stable_error=stable_error,
        canary_p99=canary_p99,
        stable_p99=stable_p99,
        alert_count=alert_count,
    )

    if is_global_noise:
        return ObservationAssessment(actual_breach=False, is_global_noise=True)

    is_phantom = _is_phantom_alert_event(
        scenario=scenario,
        step_number=step_number,
        canary_error=canary_error,
        canary_p99=canary_p99,
        stable_p99=stable_p99,
        alert_count=alert_count,
    )

    if is_phantom:
        return ObservationAssessment(actual_breach=False, is_phantom_alert=True)

    actual_breach = _actual_breach(
        scenario=scenario,
        canary_error=canary_error,
        canary_p99=canary_p99,
        stable_p99=stable_p99,
        alert_count=alert_count,
    )
    if actual_breach:
        return ObservationAssessment(actual_breach=True)

    is_warning_zone, warning_kind = _warning_zone(
        scenario=scenario,
        traffic_pct=traffic_pct,
        step_number=step_number,
        canary_error=canary_error,
        stable_error=stable_error,
        canary_p99=canary_p99,
        stable_p99=stable_p99,
    )
    return ObservationAssessment(
        actual_breach=False,
        is_global_noise=False,
        is_phantom_alert=False,
        is_warning_zone=is_warning_zone,
        warning_kind=warning_kind,
    )


def _rollback_timing_score(traffic_pct: float, threshold: float | None) -> float:
    if threshold is None:
        return 0.0

    traffic_past_threshold = max(0.0, traffic_pct - threshold)
    if traffic_past_threshold <= 0.00:
        return ROLLBACK_TIMING_WEIGHT
    if traffic_past_threshold <= 0.05:
        return 0.22
    if traffic_past_threshold <= 0.10:
        return 0.18
    if traffic_past_threshold <= 0.20:
        return 0.10
    return 0.04


def _reasoning_score(
    *,
    action: CanaryAction,
    actual_breach: bool,
) -> tuple[float, str]:
    reasoning = " ".join(action.reasoning.lower().split())
    if len(reasoning) < 15:
        return 0.0, "Reasoning was too short to add evaluator value."

    has_number = bool(re.search(r"\d", reasoning))
    mentions_error = "error" in reasoning
    mentions_latency = "latency" in reasoning or "p99" in reasoning or "ms" in reasoning
    mentions_comparison = (
        ("stable" in reasoning and "canary" in reasoning)
        or "differential" in reasoning
        or "diff" in reasoning
    )
    mentions_threshold = "threshold" in reasoning or "slo" in reasoning or "breach" in reasoning
    mentions_decision = (
        action.action_type in reasoning
        or ("rollback" in reasoning)
        or ("hold" in reasoning)
        or ("increase" in reasoning)
    )

    score = 0.0
    if mentions_error or mentions_latency:
        score += 0.02
    if mentions_error and mentions_latency:
        score += 0.01
    if mentions_comparison:
        score += 0.02
    if mentions_threshold:
        score += 0.02
    if has_number:
        score += 0.02
    if mentions_decision:
        score += 0.01

    if action.slo_breach_detected == actual_breach:
        score += 0.02
        feedback = "Reasoning aligned with the actual breach status."
    else:
        score = min(score, 0.02)
        feedback = (
            f"Reasoning breach diagnosis disagreed with reality "
            f"(reported={action.slo_breach_detected}, actual={actual_breach})."
        )

    return round(min(score, REASONING_WEIGHT), 4), feedback


def _score_phantom_alert(result: GradeResult, action: CanaryAction) -> None:
    if action.action_type.startswith("increase_") or action.action_type == "hold":
        result.policy_assessment = "correct_phantom_ignore"
        result.breach_detection_score = BREACH_DETECTION_WEIGHT
        result.rollback_timing_score = ROLLBACK_TIMING_WEIGHT
        result.promotion_safety_score = 0.22
        if action.action_type == "hold":
            result.feedback_parts.append(
                "Correctly held traffic despite phantom alerts — metrics confirm no real breach."
            )
        else:
            result.feedback_parts.append(
                "Correctly promoted through phantom monitoring alerts by verifying the raw metrics."
            )
        return

    result.policy_assessment = "false_positive_phantom_rollback"
    result.breach_detection_score = 0.04
    result.rollback_timing_score = 0.02
    result.promotion_safety_score = 0.02
    result.feedback_parts.append(
        "Rolled back on alert_count alone without verifying that the raw metrics were healthy — this was a false positive."
    )


def _score_global_noise(result: GradeResult, action: CanaryAction) -> None:
    if action.action_type == "hold":
        result.policy_assessment = "correct_noise_hold"
        result.breach_detection_score = BREACH_DETECTION_WEIGHT
        result.rollback_timing_score = ROLLBACK_TIMING_WEIGHT
        result.promotion_safety_score = 0.20
        result.feedback_parts.append(
            "Recognized shared infrastructure noise and kept traffic steady."
        )
        return

    if action.action_type == "increase_5":
        result.policy_assessment = "risky_noise_probe"
        result.breach_detection_score = 0.16
        result.rollback_timing_score = 0.12
        result.promotion_safety_score = 0.08
        result.feedback_parts.append(
            "A small probe during shared noise is still riskier than waiting for the signal to clear."
        )
        return

    if action.action_type.startswith("increase_"):
        result.policy_assessment = "risky_noise_increase"
        result.breach_detection_score = 0.10
        result.rollback_timing_score = 0.08
        result.promotion_safety_score = 0.04
        result.feedback_parts.append(
            "Promoted traffic while both stable and canary were distorted by shared noise."
        )
        return

    result.policy_assessment = "false_positive_noise_rollback"
    result.breach_detection_score = 0.05
    result.rollback_timing_score = 0.02
    result.promotion_safety_score = 0.02
    result.feedback_parts.append(
        "Rolled back on a shared infrastructure spike instead of waiting for canary-specific evidence."
    )


def _score_warning_zone(
    result: GradeResult,
    action: CanaryAction,
    scenario: Scenario,
) -> None:
    if scenario.id == "hard":
        if action.action_type == "hold":
            result.policy_assessment = "correct_warning_hold"
            result.breach_detection_score = 0.32
            result.rollback_timing_score = 0.24
            result.promotion_safety_score = 0.17
            result.feedback_parts.append(
                "Held during the post-noise watch window so the canary-specific differential could become clearer."
            )
            return

        if action.action_type == "increase_5":
            result.policy_assessment = "cautious_warning_probe"
            result.breach_detection_score = 0.24
            result.rollback_timing_score = 0.16
            result.promotion_safety_score = 0.10
            result.feedback_parts.append(
                "A 5% probe is interpretable, but the safer move was to hold through the watch window."
            )
            return

        if action.action_type.startswith("increase_"):
            result.policy_assessment = "risky_warning_promotion"
            result.breach_detection_score = 0.08
            result.rollback_timing_score = 0.06
            result.promotion_safety_score = 0.03
            result.feedback_parts.append(
                "Promoted through an ambiguous post-noise warning window instead of waiting for confirmation."
            )
            return

        result.policy_assessment = "premature_warning_rollback"
        result.breach_detection_score = 0.15
        result.rollback_timing_score = 0.10
        result.promotion_safety_score = 0.05
        result.feedback_parts.append(
            "Rolled back before the canary-specific breach was confirmed."
        )
        return

    if action.action_type == "increase_5":
        result.policy_assessment = "correct_warning_probe"
        result.breach_detection_score = 0.30
        result.rollback_timing_score = 0.22
        result.promotion_safety_score = 0.15
        result.feedback_parts.append(
            "Used a small confirmation step while the medium task was drifting but not yet breaching."
        )
        return

    if action.action_type == "hold":
        result.policy_assessment = "watchful_warning_hold"
        result.breach_detection_score = 0.27
        result.rollback_timing_score = 0.18
        result.promotion_safety_score = 0.12
        result.feedback_parts.append(
            "Holding on a medium warning state is safe, but a cautious 5% probe is more informative."
        )
        return

    if action.action_type.startswith("increase_"):
        if action.action_type == "increase_25":
            result.policy_assessment = "reckless_warning_promotion"
            result.breach_detection_score = 0.04
            result.rollback_timing_score = 0.02
            result.promotion_safety_score = 0.01
            result.feedback_parts.append(
                "A 25% jump ignored obvious trend risk in a gradual-degradation rollout."
            )
            return

        result.policy_assessment = "risky_warning_promotion"
        result.breach_detection_score = 0.14
        result.rollback_timing_score = 0.10
        result.promotion_safety_score = 0.05
        result.feedback_parts.append(
            "A full 10% promotion was too aggressive for the observed medium-task drift."
        )
        return

    result.policy_assessment = "premature_warning_rollback"
    result.breach_detection_score = 0.12
    result.rollback_timing_score = 0.08
    result.promotion_safety_score = 0.04
    result.feedback_parts.append(
        "Rolled back before the medium-task regression was confirmed."
    )


def _score_actual_breach(
    result: GradeResult,
    action: CanaryAction,
    scenario: Scenario,
    traffic_pct: float,
) -> None:
    if action.action_type == "rollback":
        result.policy_assessment = "correct_rollback"
        result.breach_detection_score = BREACH_DETECTION_WEIGHT
        result.rollback_timing_score = _rollback_timing_score(
            traffic_pct,
            scenario.correct_rollback_at,
        )
        result.promotion_safety_score = PROMOTION_SAFETY_WEIGHT
        result.feedback_parts.append(
            "Rolled back in response to a confirmed canary breach."
        )
        return

    if action.action_type == "hold":
        result.policy_assessment = "late_hold_during_breach"
        result.breach_detection_score = 0.18
        result.rollback_timing_score = 0.08
        result.promotion_safety_score = 0.04
        result.feedback_parts.append(
            "Avoided a promotion during breach, but rollback was the correct decision."
        )
        return

    if action.action_type == "increase_5":
        result.policy_assessment = "dangerous_breach_promotion"
        result.breach_detection_score = 0.04
        result.rollback_timing_score = 0.0
        result.promotion_safety_score = 0.0
        result.feedback_parts.append(
            "Even a 5% increase is unsafe once the canary is already in breach."
        )
        return

    result.policy_assessment = "dangerous_breach_promotion"
    result.breach_detection_score = 0.0
    result.rollback_timing_score = 0.0
    result.promotion_safety_score = 0.0
    result.feedback_parts.append(
        "Promoted traffic while the canary was already breaching rollout guardrails."
    )


def _score_healthy_state(
    result: GradeResult,
    action: CanaryAction,
    consecutive_holds: int,
) -> None:
    if action.action_type == "increase_10":
        result.policy_assessment = "healthy_standard_promotion"
        result.breach_detection_score = BREACH_DETECTION_WEIGHT
        result.rollback_timing_score = ROLLBACK_TIMING_WEIGHT
        result.promotion_safety_score = 0.27
        result.feedback_parts.append(
            "Promoted by the standard 10% step while metrics were healthy."
        )
        return

    if action.action_type == "increase_5":
        result.policy_assessment = "healthy_cautious_promotion"
        result.breach_detection_score = 0.33
        result.rollback_timing_score = 0.23
        result.promotion_safety_score = 0.18
        result.feedback_parts.append(
            "Promoted cautiously on healthy metrics."
        )
        return

    if action.action_type == "increase_25":
        result.policy_assessment = "healthy_aggressive_promotion"
        result.breach_detection_score = 0.23
        result.rollback_timing_score = 0.15
        result.promotion_safety_score = 0.08
        result.feedback_parts.append(
            "A 25% jump is more aggressive than a typical canary step-up."
        )
        return

    if action.action_type == "hold":
        result.policy_assessment = "healthy_hold"
        result.breach_detection_score = 0.18
        result.rollback_timing_score = 0.14
        result.promotion_safety_score = 0.05 if consecutive_holds == 0 else 0.01
        if consecutive_holds >= 2:
            result.feedback_parts.append(
                "Repeated holds on healthy metrics stalled the rollout."
            )
        else:
            result.feedback_parts.append(
                "Holding on healthy metrics was safe, but slower than needed."
            )
        return

    result.policy_assessment = "premature_rollback"
    result.breach_detection_score = 0.05
    result.rollback_timing_score = 0.0
    result.promotion_safety_score = 0.0
    result.feedback_parts.append(
        "Rolled back even though the observed metrics were healthy."
    )


def grade(
    action: CanaryAction,
    traffic_pct: float,
    canary_error: float,
    canary_p99: float,
    stable_error: float,
    stable_p99: float,
    alert_count: int,
    scenario: Scenario,
    step_number: int,
    consecutive_holds: int,
) -> GradeResult:
    result = GradeResult()
    diff_p99 = canary_p99 - stable_p99

    assessment = _assess_observation(
        scenario=scenario,
        traffic_pct=traffic_pct,
        step_number=step_number,
        canary_error=canary_error,
        stable_error=stable_error,
        canary_p99=canary_p99,
        stable_p99=stable_p99,
        alert_count=alert_count,
    )
    result.actual_breach = assessment.actual_breach
    result.is_global_noise = assessment.is_global_noise
    result.is_phantom_alert = assessment.is_phantom_alert

    if assessment.is_phantom_alert:
        _score_phantom_alert(result, action)
    elif assessment.is_global_noise:
        _score_global_noise(result, action)
    elif assessment.is_warning_zone:
        _score_warning_zone(result, action, scenario)
    elif assessment.actual_breach:
        _score_actual_breach(result, action, scenario, traffic_pct)
    else:
        _score_healthy_state(result, action, consecutive_holds)

    result.reasoning_score, reasoning_feedback = _reasoning_score(
        action=action,
        actual_breach=assessment.actual_breach,
    )
    result.feedback_parts.append(reasoning_feedback)

    if assessment.is_phantom_alert:
        result.explanation = (
            f"Monitoring fired {alert_count} alert(s) but canary_error_rate={canary_error:.4f} "
            f"and differential_p99_ms={diff_p99:.1f}ms were both within guardrails — this was a false-positive alert storm."
        )
    elif assessment.is_global_noise:
        result.explanation = (
            "Stable and canary latencies spiked together with no differential gap, so the observation matched shared infrastructure noise."
        )
    elif assessment.is_warning_zone and assessment.warning_kind == "post_noise_watch":
        result.explanation = (
            f"After the shared-noise event, the canary stayed {diff_p99:.1f}ms slower than stable without a confirmed breach yet."
        )
    elif assessment.is_warning_zone:
        result.explanation = (
            f"The canary was drifting toward breach with canary_error_rate={canary_error:.4f} "
            f"and differential_p99_ms={diff_p99:.1f}."
        )
    elif assessment.actual_breach:
        result.explanation = (
            f"The observed state was in breach: canary_error_rate={canary_error:.4f}, "
            f"differential_p99_ms={diff_p99:.1f}."
        )
    else:
        result.explanation = (
            f"The observed state was healthy: canary_error_rate={canary_error:.4f}, "
            f"differential_p99_ms={diff_p99:.1f}."
        )

    return result
