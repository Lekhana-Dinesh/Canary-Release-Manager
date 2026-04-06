from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canary_release_env import CanaryAction, CanaryEnv
from canary_release_env.server.policies import safe_fallback_action

TASK_IDS = ("easy", "medium", "hard")
DEFAULT_ENV_BASE_URL = "http://127.0.0.1:7860"
ALLOWED_TAGS = {"START", "STEP", "END"}
ALLOWED_ACTION_TYPES = {
    "increase_5",
    "increase_10",
    "increase_25",
    "hold",
    "rollback",
}


def _env_settings() -> dict[str, str]:
    return {
        "api_base_url": os.getenv("API_BASE_URL", "").strip(),
        "model_name": os.getenv("MODEL_NAME", "").strip(),
        "hf_token": os.getenv("HF_TOKEN", "").strip(),
    }


def _emit_event(tag: str, **fields: Any) -> None:
    if tag not in ALLOWED_TAGS:
        raise ValueError(f"Unsupported log tag: {tag}")

    ordered = " ".join(
        f"{key}={json.dumps(value, separators=(',', ':'))}"
        for key, value in fields.items()
    )
    line = f"[{tag}]"
    if ordered:
        line = f"{line} {ordered}"
    print(line, flush=True)


def _build_messages(observation) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You manage a production-style model canary rollout benchmark. "
                "Return only JSON with keys action_type, reasoning, slo_breach_detected. "
                "Allowed action_type values: increase_5, increase_10, increase_25, hold, rollback. "
                "Use the current observation only. Prefer a standard 10% promotion on healthy metrics, "
                "hold during shared-noise or watch-window states, and rollback on confirmed canary breaches."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": observation.task_id,
                    "task_description": observation.task_description,
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

    try:
        data = _extract_json_object(raw_content)
        action_type = str(data.get("action_type", "")).strip()
        slo_breach_detected = data.get("slo_breach_detected")
        reasoning = _normalized_reasoning(data.get("reasoning", ""), fallback.reasoning)

        if action_type not in ALLOWED_ACTION_TYPES:
            return fallback
        if not isinstance(slo_breach_detected, bool):
            return fallback

        return CanaryAction(
            action_type=action_type,
            reasoning=reasoning,
            slo_breach_detected=slo_breach_detected,
        )
    except Exception:
        return fallback


def _build_client(settings: dict[str, str]) -> OpenAI | None:
    if not all(settings.values()):
        return None
    return OpenAI(
        base_url=settings["api_base_url"],
        api_key=settings["hf_token"],
    )


def _model_action(client: OpenAI, model_name: str, observation) -> CanaryAction:
    response = client.chat.completions.create(
        model=model_name,
        messages=_build_messages(observation),
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return _parse_model_action(content, observation)


def _decide_action(
    client: OpenAI | None,
    model_name: str,
    observation,
) -> tuple[CanaryAction, str]:
    if client is None or not model_name:
        return safe_fallback_action(observation), "fallback"

    try:
        return _model_action(client, model_name, observation), "model"
    except Exception:
        return safe_fallback_action(observation), "fallback"


async def _run_task(
    client: OpenAI | None,
    model_name: str,
    env_base_url: str,
    task_id: str,
) -> dict[str, Any]:
    _emit_event("START", task=task_id, model=model_name or "fallback")

    try:
        async with CanaryEnv(base_url=env_base_url) as env:
            observation = await env.reset(task=task_id)

            while not observation.is_done:
                action, source = _decide_action(client, model_name, observation)
                result = await env.step(action)
                next_observation = result.observation
                _emit_event(
                    "STEP",
                    task=task_id,
                    observed_step=observation.step_number,
                    traffic_pct=observation.traffic_pct,
                    action=action.action_type,
                    source=source,
                    reward=round(float(result.reward or 0.0), 4),
                    done=bool(result.done),
                    assessment=next_observation.policy_assessment,
                )
                observation = next_observation

        outcome = "success" if observation.traffic_pct >= 1.0 else "rollback"
        score = round(observation.cumulative_reward, 4)
        _emit_event(
            "END",
            task=task_id,
            score=score,
            steps=observation.step_number,
            outcome=outcome,
        )
        return {
            "task_id": task_id,
            "score": score,
            "steps": observation.step_number,
            "outcome": outcome,
        }
    except Exception:
        _emit_event(
            "END",
            task=task_id,
            score=0.0,
            steps=0,
            outcome="error",
        )
        return {
            "task_id": task_id,
            "score": 0.0,
            "steps": 0,
            "outcome": "error",
        }


async def main() -> list[dict[str, Any]]:
    return await run(DEFAULT_ENV_BASE_URL)


async def run(env_base_url: str) -> list[dict[str, Any]]:
    settings = _env_settings()
    client = _build_client(settings)
    results = []

    for task_id in TASK_IDS:
        results.append(
            await _run_task(
                client=client,
                model_name=settings["model_name"],
                env_base_url=env_base_url,
                task_id=task_id,
            )
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-url",
        default=DEFAULT_ENV_BASE_URL,
        help="Base URL for the environment. Defaults to the validator-safe local port.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.env_url))
