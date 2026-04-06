import asyncio
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import (
    _env_settings,
    _parse_model_action,
    _run_task,
)


def make_observation(**overrides):
    base = {
        "task_id": "hard",
        "task_description": "Test task",
        "traffic_pct": 0.0,
        "canary_error_rate": 0.001,
        "canary_p99_ms": 142.0,
        "stable_error_rate": 0.001,
        "stable_p99_ms": 142.0,
        "differential_error": 0.0,
        "differential_p99_ms": 0.0,
        "alert_count": 0,
        "step_number": 0,
        "consecutive_holds": 0,
        "is_done": False,
        "cumulative_reward": 0.0,
        "policy_assessment": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeEnv:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._observation = make_observation()

    async def connect(self):
        return None

    async def close(self):
        return None

    async def reset(self, task: str):
        self._observation = make_observation(task_id=task)
        return self._observation

    async def step(self, action):
        self._observation = make_observation(
            task_id=self._observation.task_id,
            traffic_pct=0.1,
            step_number=1,
            is_done=True,
            cumulative_reward=0.87,
            policy_assessment="healthy_standard_promotion",
        )
        return SimpleNamespace(
            observation=self._observation,
            reward=0.87,
            done=True,
        )


class InferenceTests(unittest.TestCase):
    def test_env_settings_reads_only_required_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "API_BASE_URL": "https://example.invalid/v1",
                "MODEL_NAME": "test-model",
                "HF_TOKEN": "secret",
                "LOCAL_IMAGE_NAME": "local-image",
                "ENV_BASE_URL": "http://should-be-ignored",
            },
            clear=True,
        ):
            settings = _env_settings()

        self.assertEqual(
            settings,
            {
                "api_base_url": "https://example.invalid/v1",
                "model_name": "test-model",
                "hf_token": "secret",
                "local_image_name": "local-image",
            },
        )

    def test_parse_model_action_falls_back_on_malformed_output(self) -> None:
        observation = make_observation(
            traffic_pct=0.30,
            step_number=4,
            differential_error=0.0016,
            differential_p99_ms=24.0,
            canary_p99_ms=168.0,
            stable_p99_ms=144.0,
            canary_error_rate=0.0026,
        )

        action = _parse_model_action("not-json", observation)
        self.assertEqual(action.action_type, "hold")
        self.assertFalse(action.slo_breach_detected)

    def test_run_task_emits_only_start_step_end_lines(self) -> None:
        stream = io.StringIO()
        with patch("inference.CanaryEnv", FakeEnv):
            with redirect_stdout(stream):
                result = asyncio.run(
                    _run_task(
                        client=None,
                        model_name="",
                        env_base_url="http://127.0.0.1:7860",
                        local_image_name="",
                        task_id="easy",
                    )
                )

        lines = [line.strip() for line in stream.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "[START] task=easy env=canary-release-env model=fallback")
        self.assertEqual(
            lines[1],
            "[STEP] step=1 action=increase_10 reward=0.87 done=true error=null",
        )
        self.assertEqual(lines[2], "[END] success=true steps=1 score=0.8700 rewards=0.87")
        self.assertEqual(result["score"], 0.87)
        self.assertEqual(result["outcome"], "rollback")


if __name__ == "__main__":
    unittest.main()
