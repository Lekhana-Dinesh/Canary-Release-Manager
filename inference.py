from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent
    parent = repo_root.parent
    for path in (repo_root, parent):
        candidate = str(path)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
        from client import CanaryEnv
        from models import CanaryAction
        from server.canary_environment import CanaryEnvironment
        from server.policies import safe_fallback_action
        from server.scenarios import PUBLIC_TASK_IDS
    except ModuleNotFoundError:
        from canary_release_env import CanaryAction, CanaryEnv
        from canary_release_env.server.canary_environment import CanaryEnvironment
        from canary_release_env.server.policies import safe_fallback_action
        from canary_release_env.server.scenarios import PUBLIC_TASK_IDS
else:
    from canary_release_env import CanaryAction, CanaryEnv
    from canary_release_env.server.canary_environment import CanaryEnvironment
    from canary_release_env.server.policies import safe_fallback_action
    from canary_release_env.server.scenarios import PUBLIC_TASK_IDS

TASK_IDS = PUBLIC_TASK_IDS
ENV_BENCHMARK = "canary-release-env"
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
MODEL_CALL_TIMEOUT_SECS = 45.0
ENV_OPERATION_TIMEOUT_SECS = 30.0
ALLOWED_ACTION_TYPES = {
    "increase_5",
    "increase_10",
    "increase_25",
    "hold",
    "rollback",
}
ALLOWED_STATE_ASSESSMENTS = {
    "healthy",
    "warning",
    "noise",
    "phantom_alert",
    "breach",
}


@dataclass
class DecisionEnvelope:
    action: CanaryAction
    error: str | None = None
    degraded: bool = False
    attempted_model_call: bool = False




def _build_messages(observation) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You manage a production-style model canary rollout benchmark. "
                "Return only JSON with keys action_type, reasoning, slo_breach_detected, state_assessment. "
                "Allowed action_type values: increase_5, increase_10, increase_25, hold, rollback. "
                "Allowed state_assessment values: healthy, warning, noise, phantom_alert, breach. "
                "state_assessment is a coarse public family label, not a benchmark-internal subtype. "
                "Use warning for ambiguous pre-breach drift, post-noise watch windows, or transient recovery windows. "
                "Use noise only for shared infrastructure distortion that affects stable and canary together. "
                "The structured fields drive evaluation; reasoning is retained mainly for transcript clarity, so keep it short and metric-grounded. "
                "Use only the current telemetry and the guardrail fields in the observation. "
                "Treat alerts as supporting evidence and verify them against the raw metrics before rollback. "
                "Prefer a standard 10% promotion on healthy metrics, hold during shared-noise or watch-window states, "
                "and rollback on confirmed canary breaches."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "traffic_pct": observation.traffic_pct,
                    "canary_error_rate": observation.canary_error_rate,
                    "canary_p99_ms": observation.canary_p99_ms,
                    "stable_error_rate": observation.stable_error_rate,
                    "stable_p99_ms": observation.stable_p99_ms,
                    "differential_error": observation.differential_error,
                    "differential_p99_ms": observation.differential_p99_ms,
                    "alert_count": observation.alert_count,
                    "step_number": observation.step_number,
                    "consecutive_holds": observation.consecutive_holds,
                    "rollback_on_error_rate": observation.rollback_on_error_rate,
                    "rollback_on_canary_p99_ms": observation.rollback_on_canary_p99_ms,
                    "rollback_on_differential_p99_ms": observation.rollback_on_differential_p99_ms,
                },
                separators=(",", ":"),
            ),
        },
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Model output was not a JSON object.")
    return parsed


def _normalized_reasoning(reasoning: Any, fallback_reasoning: str) -> str:
    normalized = " ".join(str(reasoning).split())
    if len(normalized) < 15:
        return fallback_reasoning
    return normalized[:280]


def _parse_model_action(raw_content: str, observation) -> CanaryAction:
    fallback = safe_fallback_action(observation)

    data = _extract_json_object(raw_content)
    action_type = str(data.get("action_type", "")).strip()
    state_assessment = str(data.get("state_assessment", "")).strip()
    slo_breach_detected = data.get("slo_breach_detected")
    reasoning = _normalized_reasoning(data.get("reasoning", ""), fallback.reasoning)

    if action_type not in ALLOWED_ACTION_TYPES:
        raise ValueError(f"invalid_action_type:{action_type or 'missing'}")
    if state_assessment not in ALLOWED_STATE_ASSESSMENTS:
        raise ValueError(f"invalid_state_assessment:{state_assessment or 'missing'}")
    if not isinstance(slo_breach_detected, bool):
        raise ValueError("invalid_slo_breach_detected")

    return CanaryAction(
        action_type=action_type,
        reasoning=reasoning,
        slo_breach_detected=slo_breach_detected,
        state_assessment=state_assessment,
    )




def _sanitized_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()[:120] or exc.__class__.__name__


def _model_action(client: OpenAI, model_name: str, observation) -> CanaryAction:
    response = client.chat.completions.create(
        model=model_name,
        messages=_build_messages(observation),
        temperature=0,
        timeout=MODEL_CALL_TIMEOUT_SECS,
    )
    content = response.choices[0].message.content or "{}"
    return _parse_model_action(content, observation)


def _decide_action(
    client: OpenAI | None,
    model_name: str,
    observation,
) -> DecisionEnvelope:
    if client is None or not model_name:
        return DecisionEnvelope(action=safe_fallback_action(observation))

    try:
        return DecisionEnvelope(
            action=_model_action(client, model_name, observation),
            attempted_model_call=True,
        )
    except Exception as exc:
        return DecisionEnvelope(
            action=safe_fallback_action(observation),
            error=f"model_call_failed:{_sanitized_error(exc)}",
            degraded=True,
            attempted_model_call=True,
        )


class _LocalEnvRunner:
    def __init__(self) -> None:
        self._env = CanaryEnvironment()

    async def __aenter__(self) -> "_LocalEnvRunner":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def reset(self, task: str, seed: int = 0):
        return self._env.reset(task=task, seed=seed)

    async def step(self, action: CanaryAction):
        observation = self._env.step(action)
        return SimpleNamespace(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    async def close(self) -> None:
        return None


def _make_remote_runner(env_base_url: str):
    class _RemoteEnvRunner:
        def __init__(self, base_url: str) -> None:
            self._base_url = base_url
            self._env: CanaryEnv | None = None

        async def __aenter__(self):
            self._env = CanaryEnv(base_url=self._base_url)
            await self._env.connect()
            return self._env

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            if self._env is not None:
                await self._env.close()
            return False

    return _RemoteEnvRunner(env_base_url)


def _make_image_runner(local_image_name: str):
    class _ImageEnvRunner:
        def __init__(self, image: str) -> None:
            self._image = image
            self._env: CanaryEnv | None = None

        async def __aenter__(self):
            self._env = await CanaryEnv.from_docker_image(self._image)
            return self._env

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            if self._env is not None:
                await self._env.close()
            return False

    return _ImageEnvRunner(local_image_name)


async def _await_with_timeout(awaitable, *, label: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=ENV_OPERATION_TIMEOUT_SECS)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{label}_timeout") from exc


async def _run_task(
    client: OpenAI | None,
    model_name: str,
    env_base_url: str | None,
    local_image_name: str,
    task_id: str,
    *,
    proxy_required: bool = False,
    startup_error: str | None = None,
) -> dict[str, Any]:
    display_model = model_name if proxy_required else "fallback"
    print(f"[START] task={task_id} env={ENV_BENCHMARK} model={display_model}", flush=True)

    step_rewards: list[float] = []
    success = False
    degraded = False
    steps_taken = 0
    final_traffic_pct = 0.0
    step_failed = False
    model_call_attempted = False
    step_num = 0
    env_runner = None
    runner_entered = False

    try:
        if env_base_url:
            env_runner = _make_remote_runner(env_base_url)
        elif local_image_name:
            env_runner = _make_image_runner(local_image_name)
        else:
            env_runner = _LocalEnvRunner()

        env = await _await_with_timeout(env_runner.__aenter__(), label="runner_enter")
        runner_entered = True
        observation = await _await_with_timeout(
            env.reset(task=task_id, seed=0),
            label="reset",
        )

        while not observation.done:
            step_num += 1
            decision = _decide_action(client, model_name, observation)
            action = decision.action
            error_val = decision.error or "null"
            reward = 0.0
            done = False
            model_call_attempted = model_call_attempted or decision.attempted_model_call

            try:
                result = await _await_with_timeout(
                    env.step(action),
                    label="step",
                )
                next_obs = result.observation
                reward = round(float(result.reward or 0.0), 2)
                done = bool(result.done)
                observation = next_obs
            except Exception as exc:
                done = True
                step_failed = True
                error_val = _sanitized_error(exc)

            degraded = degraded or decision.degraded
            step_rewards.append(reward)
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_num} action={action.action_type} "
                f"reward={reward:.2f} done={done_str} error={error_val}",
                flush=True,
            )
            if done:
                break

        success = not step_failed
        steps_taken = step_num
        final_traffic_pct = getattr(observation, "traffic_pct", 0.0)

    except Exception as exc:
        success = False
        steps_taken = step_num if step_num else len(step_rewards)
        print(f"task={task_id} env_error={_sanitized_error(exc)}", file=sys.stderr, flush=True)
    finally:
        if runner_entered and env_runner is not None:
            try:
                await _await_with_timeout(
                    env_runner.__aexit__(None, None, None),
                    label="close",
                )
            except Exception as exc:
                print(
                    f"task={task_id} close_error={_sanitized_error(exc)}",
                    file=sys.stderr,
                    flush=True,
                )

    if proxy_required and startup_error:
        degraded = True
        print(
            f"task={task_id} proxy_error={startup_error}",
            file=sys.stderr,
            flush=True,
        )

    if proxy_required and not model_call_attempted:
        success = False
        degraded = True
        print(
            f"task={task_id} proxy_error=no_model_call_attempted",
            file=sys.stderr,
            flush=True,
        )

    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
    avg_score = round(sum(step_rewards) / len(step_rewards), 4) if step_rewards else 0.0
    success_str = "true" if success and not degraded else "false"
    print(
        f"[END] success={success_str} steps={steps_taken} score={avg_score:.4f} rewards={rewards_str}",
        flush=True,
    )

    if not success:
        outcome = "error"
    elif final_traffic_pct >= 1.0:
        outcome = "success"
    else:
        outcome = "rollback"

    return {
        "task_id": task_id,
        "score": avg_score,
        "steps": steps_taken,
        "outcome": outcome,
        "degraded": degraded,
        "attempted_model_call": model_call_attempted,
    }


async def main() -> list[dict[str, Any]]:
    return await run(None)


async def run(env_base_url: str | None) -> list[dict[str, Any]]:
    # Log all env var names present (not values) for debugging
    relevant = ["API_BASE_URL", "API_KEY", "HF_TOKEN", "MODEL_NAME",
                "LOCAL_IMAGE_NAME", "IMAGE_NAME", "OPENAI_API_KEY", "OPENAI_BASE_URL"]
    present = [k for k in relevant if os.environ.get(k)]
    print(f"[ENV] present={present}", file=sys.stderr, flush=True)

    # Build client exactly as the validator instructs
    client: OpenAI | None = None
    startup_error: str | None = None
    proxy_required = False
    try:
        api_base_url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        client = OpenAI(base_url=api_base_url, api_key=api_key)
        proxy_required = True
        print(f"[ENV] client=proxy base_url_set=True", file=sys.stderr, flush=True)
    except KeyError:
        # Validator vars not present — try HF_TOKEN fallback
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            try:
                fallback_base = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)
                client = OpenAI(base_url=fallback_base, api_key=hf_token)
                proxy_required = True
                print(f"[ENV] client=hf_token_fallback", file=sys.stderr, flush=True)
            except Exception as exc:
                startup_error = f"client_init_failed:{_sanitized_error(exc)}"
                print(f"[ENV] {startup_error}", file=sys.stderr, flush=True)
        else:
            print("[ENV] client=fallback no_credentials", file=sys.stderr, flush=True)
    except Exception as exc:
        startup_error = f"client_init_failed:{_sanitized_error(exc)}"
        print(f"[ENV] {startup_error}", file=sys.stderr, flush=True)

    model_name = os.environ.get("MODEL_NAME", "").strip() or DEFAULT_MODEL_NAME
    local_image_name = (
        os.environ.get("LOCAL_IMAGE_NAME", "").strip()
        or os.environ.get("IMAGE_NAME", "").strip()
    )
    results = []

    for task_id in TASK_IDS:
        results.append(
            await _run_task(
                client=client,
                model_name=model_name,
                env_base_url=env_base_url,
                local_image_name=local_image_name,
                task_id=task_id,
                proxy_required=proxy_required,
                startup_error=startup_error,
            )
        )

    return results


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(exit_on_error=False)
        parser.add_argument(
            "--env-url",
            default=None,
            help="Base URL for a running environment. If omitted, LOCAL_IMAGE_NAME is used when set; otherwise the script falls back to an in-process environment.",
        )
        args, _unknown = parser.parse_known_args()
        asyncio.run(run(args.env_url))
    except Exception as _top_exc:
        print(f"top-level error: {_top_exc}", file=sys.stderr, flush=True)
