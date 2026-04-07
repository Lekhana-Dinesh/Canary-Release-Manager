"""
Shared observation-side policies used by the baseline runner and inference fallback.
"""
from __future__ import annotations

from ._compat import bootstrap_import_paths

bootstrap_import_paths()

USE_PACKAGE_IMPORTS = (__package__ or "").startswith("canary_release_env.server")

if USE_PACKAGE_IMPORTS:
    from canary_release_env.models import CanaryAction
    from canary_release_env.server.scenarios import SLO_DIFFERENTIAL_P99_THRESHOLD
else:
    from models import CanaryAction
    from server.scenarios import SLO_DIFFERENTIAL_P99_THRESHOLD


def observation_uses_differential_guardrail(observation) -> bool:
    return observation.rollback_on_differential_p99_ms is not None


def observation_uses_absolute_latency_guardrail(observation) -> bool:
    return observation.rollback_on_canary_p99_ms is not None


def observation_is_phantom_alert(observation) -> bool:
    return (
        observation.alert_count >= 2
        and not observation_has_actual_breach(observation)
        and observation.canary_error_rate <= observation.rollback_on_error_rate
        and observation.differential_p99_ms <= min(
            15.0,
            (
                observation.rollback_on_differential_p99_ms
                if observation.rollback_on_differential_p99_ms is not None
                else SLO_DIFFERENTIAL_P99_THRESHOLD
            ),
        )
    )


def observation_is_shared_noise(observation) -> bool:
    return (
        observation.canary_p99_ms >= 180.0
        and observation.stable_p99_ms >= 180.0
        and observation.differential_p99_ms <= 15.0
        and observation.differential_error <= 0.001
        and not observation_has_actual_breach(observation)
    )


def observation_has_actual_breach(observation) -> bool:
    return (
        observation.canary_error_rate > observation.rollback_on_error_rate
        or (
            observation.rollback_on_canary_p99_ms is not None
            and observation.canary_p99_ms > observation.rollback_on_canary_p99_ms
        )
        or (
            observation.rollback_on_differential_p99_ms is not None
            and observation.differential_p99_ms > observation.rollback_on_differential_p99_ms
        )
    )


def observation_is_warning_window(observation) -> bool:
    if (
        observation_has_actual_breach(observation)
        or observation_is_shared_noise(observation)
        or observation_is_phantom_alert(observation)
    ):
        return False

    if observation.traffic_pct < 0.30:
        return False

    if observation_uses_absolute_latency_guardrail(observation):
        return observation.traffic_pct >= 0.30 and (
            observation.canary_error_rate
            >= max(0.0035, observation.rollback_on_error_rate * 0.70)
            or (
                observation.rollback_on_canary_p99_ms is not None
                and observation.canary_p99_ms
                >= observation.rollback_on_canary_p99_ms * 0.90
            )
            or observation.differential_p99_ms >= 15.0
        )

    if observation_uses_differential_guardrail(observation):
        return (
            observation.differential_p99_ms >= 20.0
            or observation.differential_error >= 0.0015
        )

    return False


def state_assessment_for_observation(observation) -> str:
    if observation_is_phantom_alert(observation):
        return "phantom_alert"
    if observation_is_shared_noise(observation):
        return "noise"
    if observation_has_actual_breach(observation):
        return "breach"
    if observation_is_warning_window(observation):
        return "warning"
    return "healthy"


def baseline_action(observation) -> CanaryAction:
    if observation_is_shared_noise(observation):
        return CanaryAction(
            action_type="hold",
            reasoning=(
                "Stable and canary latencies spiked together with no differential gap, so hold for one more observation."
            ),
            slo_breach_detected=False,
            state_assessment="noise",
        )

    # Naive alert check: roll back on any alert_count > 0, including phantom alerts.
    # This intentionally gets tricked by the expert phantom alert step.
    if observation.alert_count > 0:
        return CanaryAction(
            action_type="rollback",
            reasoning=(
                f"Alert count is {observation.alert_count}, treating as a breach and rolling back."
            ),
            slo_breach_detected=True,
            state_assessment="phantom_alert"
            if observation_is_phantom_alert(observation)
            else "breach",
        )

    if observation_has_actual_breach(observation):
        return CanaryAction(
            action_type="rollback",
            reasoning=(
                f"The canary is breaching guardrails with error={observation.canary_error_rate:.4f} "
                f"and differential_p99={observation.differential_p99_ms:.1f}ms, so rollback now."
            ),
            slo_breach_detected=True,
            state_assessment="breach",
        )

    return CanaryAction(
        action_type="increase_10",
        reasoning=(
            f"Metrics remain healthy with error={observation.canary_error_rate:.4f} "
            f"and differential_p99={observation.differential_p99_ms:.1f}ms, so take the standard 10% promotion step."
        ),
        slo_breach_detected=False,
        state_assessment=state_assessment_for_observation(observation),
    )


def safe_fallback_action(observation) -> CanaryAction:
    if observation_is_phantom_alert(observation):
        return CanaryAction(
            action_type="hold",
            reasoning=(
                f"Monitoring fired {observation.alert_count} alert(s) but canary_error_rate={observation.canary_error_rate:.4f} "
                f"and differential_p99={observation.differential_p99_ms:.1f}ms are within guardrails; "
                "hold to confirm before acting on alerts alone."
            ),
            slo_breach_detected=False,
            state_assessment="phantom_alert",
        )

    if observation_is_shared_noise(observation):
        return CanaryAction(
            action_type="hold",
            reasoning=(
                "Stable and canary latencies moved together with no differential gap, which looks like shared infrastructure noise."
            ),
            slo_breach_detected=False,
            state_assessment="noise",
        )

    if observation_has_actual_breach(observation):
        return CanaryAction(
            action_type="rollback",
            reasoning=(
                f"The canary is in breach with error={observation.canary_error_rate:.4f} "
                f"and differential_p99={observation.differential_p99_ms:.1f}ms, so rollback immediately."
            ),
            slo_breach_detected=True,
            state_assessment="breach",
        )

    if observation_is_warning_window(observation):
        if observation_uses_absolute_latency_guardrail(observation):
            return CanaryAction(
                action_type="increase_5",
                reasoning=(
                    f"The canary is drifting with error={observation.canary_error_rate:.4f} "
                    f"and differential_p99={observation.differential_p99_ms:.1f}ms, so use a 5% confirmation step."
                ),
                slo_breach_detected=False,
                state_assessment="warning",
            )

        return CanaryAction(
            action_type="hold",
            reasoning=(
                f"Metrics are drifting toward the guardrail with error={observation.canary_error_rate:.4f} "
                f"and differential_p99={observation.differential_p99_ms:.1f}ms, so hold for confirmation."
            ),
            slo_breach_detected=False,
            state_assessment="warning",
        )

    return CanaryAction(
        action_type="increase_10",
        reasoning=(
            f"Metrics are healthy with error={observation.canary_error_rate:.4f} "
            f"and differential_p99={observation.differential_p99_ms:.1f}ms, so increase traffic by 10%."
        ),
        slo_breach_detected=False,
        state_assessment="healthy",
    )
