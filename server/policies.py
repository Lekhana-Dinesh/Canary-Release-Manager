"""
Shared observation-side policies used by the baseline runner and inference fallback.
"""
from __future__ import annotations

from ._compat import bootstrap_import_paths

bootstrap_import_paths()

USE_PACKAGE_IMPORTS = (__package__ or "").startswith("canary_release_env.server")

if USE_PACKAGE_IMPORTS:
    from canary_release_env.models import CanaryAction
    from canary_release_env.server.scenarios import (
        GLOBAL_NOISE_STEP,
        SLO_DIFFERENTIAL_P99_THRESHOLD,
        SLO_ERROR_THRESHOLD,
        SLO_P99_THRESHOLD,
    )
else:
    from models import CanaryAction
    from server.scenarios import (
        GLOBAL_NOISE_STEP,
        SLO_DIFFERENTIAL_P99_THRESHOLD,
        SLO_ERROR_THRESHOLD,
        SLO_P99_THRESHOLD,
    )


def observation_is_shared_noise(observation) -> bool:
    return (
        observation.task_id == "hard"
        and observation.step_number == GLOBAL_NOISE_STEP
        and observation.canary_p99_ms >= 180.0
        and observation.stable_p99_ms >= 180.0
        and observation.differential_p99_ms <= 15.0
        and observation.differential_error <= 0.001
    )


def observation_has_actual_breach(observation) -> bool:
    if observation.task_id == "hard":
        return (
            observation.alert_count > 0
            or observation.canary_error_rate > SLO_ERROR_THRESHOLD
            or observation.differential_p99_ms > SLO_DIFFERENTIAL_P99_THRESHOLD
        )

    return (
        observation.alert_count > 0
        or observation.canary_error_rate > SLO_ERROR_THRESHOLD
        or observation.canary_p99_ms > SLO_P99_THRESHOLD
    )


def observation_is_warning_window(observation) -> bool:
    if observation_has_actual_breach(observation) or observation_is_shared_noise(observation):
        return False

    if observation.task_id == "medium":
        return observation.traffic_pct >= 0.30 and (
            observation.canary_error_rate >= 0.0035
            or observation.differential_p99_ms >= 15.0
        )

    if observation.task_id == "hard":
        return (
            observation.traffic_pct >= 0.30
            and observation.step_number >= GLOBAL_NOISE_STEP + 1
            and (
                observation.differential_p99_ms >= 20.0
                or observation.differential_error >= 0.0015
            )
        )

    return False


def baseline_action(observation) -> CanaryAction:
    if observation_is_shared_noise(observation):
        return CanaryAction(
            action_type="hold",
            reasoning=(
                "Stable and canary latencies spiked together with no differential gap, so hold for one more observation."
            ),
            slo_breach_detected=False,
        )

    if observation_has_actual_breach(observation):
        return CanaryAction(
            action_type="rollback",
            reasoning=(
                f"The canary is breaching guardrails with error={observation.canary_error_rate:.4f} "
                f"and differential_p99={observation.differential_p99_ms:.1f}ms, so rollback now."
            ),
            slo_breach_detected=True,
        )

    return CanaryAction(
        action_type="increase_10",
        reasoning=(
            f"Metrics remain healthy with error={observation.canary_error_rate:.4f} "
            f"and differential_p99={observation.differential_p99_ms:.1f}ms, so take the standard 10% promotion step."
        ),
        slo_breach_detected=False,
    )


def safe_fallback_action(observation) -> CanaryAction:
    if observation_is_shared_noise(observation):
        return CanaryAction(
            action_type="hold",
            reasoning=(
                "Stable and canary latencies moved together with no differential gap, which looks like shared infrastructure noise."
            ),
            slo_breach_detected=False,
        )

    if observation_has_actual_breach(observation):
        return CanaryAction(
            action_type="rollback",
            reasoning=(
                f"The canary is in breach with error={observation.canary_error_rate:.4f} "
                f"and differential_p99={observation.differential_p99_ms:.1f}ms, so rollback immediately."
            ),
            slo_breach_detected=True,
        )

    if observation_is_warning_window(observation):
        if observation.task_id == "medium":
            return CanaryAction(
                action_type="increase_5",
                reasoning=(
                    f"The canary is drifting with error={observation.canary_error_rate:.4f} "
                    f"and differential_p99={observation.differential_p99_ms:.1f}ms, so use a 5% confirmation step."
                ),
                slo_breach_detected=False,
            )

        return CanaryAction(
            action_type="hold",
            reasoning=(
                f"Metrics are drifting toward the guardrail with error={observation.canary_error_rate:.4f} "
                f"and differential_p99={observation.differential_p99_ms:.1f}ms, so hold for confirmation."
            ),
            slo_breach_detected=False,
        )

    return CanaryAction(
        action_type="increase_10",
        reasoning=(
            f"Metrics are healthy with error={observation.canary_error_rate:.4f} "
            f"and differential_p99={observation.differential_p99_ms:.1f}ms, so increase traffic by 10%."
        ),
        slo_breach_detected=False,
    )
