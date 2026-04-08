import asyncio
import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import (
    DEFAULT_API_BASE_URL,
    DEFAULT_MODEL_NAME,
    _build_client,
    _env_settings,
    _parse_model_action,
    _run_task,
    _use_model,
    run,
)


def make_observation(**overrides):
    base = {
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
        "rollback_on_error_rate": 0.005,
        "rollback_on_canary_p99_ms": None,
        "rollback_on_differential_p99_ms": 50.0,
        "done": False,
        "reward": 0.0,
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

    async def reset(self, task: str, seed: int = 0):
        self._observation = make_observation()
        return self._observation

    async def step(self, action):
        self._observation = make_observation(
            traffic_pct=0.1,
            step_number=1,
            done=True,
            reward=0.87,
        )
        return SimpleNamespace(
            observation=self._observation,
            reward=0.87,
            done=True,
        )


class DoneEnv(FakeEnv):
    async def reset(self, task: str, seed: int = 0):
        self._observation = make_observation(done=True)
        return self._observation


class CloseFailEnv(FakeEnv):
    async def close(self):
        raise RuntimeError("close failed")


class FakeChatCompletions:
    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"action_type":"increase_10","reasoning":"Metrics remain healthy '
                            'with low error and latency, so continue the standard rollout step.",'
                            '"slo_breach_detected":false,"state_assessment":"healthy"}'
                        )
                    )
                )
            ]
        )


class FakeOpenAIClient:
    def __init__(self, init_kwargs: dict, calls: list[dict]) -> None:
        self.init_kwargs = init_kwargs
        self.chat = SimpleNamespace(completions=FakeChatCompletions(calls))


class OpenAIFactory:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.request_calls: list[dict] = []

    def __call__(self, **kwargs):
        self.init_calls.append(kwargs)
        return FakeOpenAIClient(kwargs, self.request_calls)


class InferenceTests(unittest.TestCase):
    def test_env_settings_prefers_api_key_and_applies_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HF_TOKEN": "hf-secret",
                "API_KEY": "plain-api-key",
                "API_BASE_URL": "",
            },
            clear=True,
        ):
            settings = _env_settings()

        self.assertEqual(settings["api_base_url"], DEFAULT_API_BASE_URL)
        self.assertEqual(settings["model_name"], DEFAULT_MODEL_NAME)
        self.assertEqual(settings["api_key"], "plain-api-key")
        self.assertEqual(settings["credential_source"], "API_KEY")
        self.assertEqual(settings["local_image_name"], "")
        self.assertTrue(settings["proxy_config_present"])

    def test_env_settings_uses_hf_token_when_api_key_absent(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HF_TOKEN": "hf-secret",
            },
            clear=True,
        ):
            settings = _env_settings()

        self.assertEqual(settings["api_key"], "hf-secret")
        self.assertEqual(settings["credential_source"], "HF_TOKEN")
        self.assertEqual(settings["api_base_url"], DEFAULT_API_BASE_URL)
        self.assertEqual(settings["model_name"], DEFAULT_MODEL_NAME)

    def test_use_model_only_requires_proxy_credentials_once_defaults_exist(self) -> None:
        settings = {
            "api_base_url": DEFAULT_API_BASE_URL,
            "model_name": DEFAULT_MODEL_NAME,
            "api_key": "validator-key",
            "local_image_name": "",
            "proxy_config_present": True,
            "credential_source": "API_KEY",
        }
        self.assertTrue(_use_model(settings))

    def test_build_client_uses_api_key_before_hf_token(self) -> None:
        factory = OpenAIFactory()
        settings = {
            "api_base_url": DEFAULT_API_BASE_URL,
            "model_name": DEFAULT_MODEL_NAME,
            "api_key": "validator-key",
            "local_image_name": "",
            "proxy_config_present": True,
            "credential_source": "API_KEY",
        }

        with patch("inference.OpenAI", factory):
            client = _build_client(settings)

        self.assertIsNotNone(client)
        self.assertEqual(
            factory.init_calls,
            [{"base_url": DEFAULT_API_BASE_URL, "api_key": "validator-key"}],
        )

    def test_parse_model_action_rejects_malformed_output(self) -> None:
        observation = make_observation(
            traffic_pct=0.30,
            step_number=4,
            differential_error=0.0016,
            differential_p99_ms=24.0,
            canary_p99_ms=168.0,
            stable_p99_ms=144.0,
            canary_error_rate=0.0026,
        )

        with self.assertRaises(Exception):
            _parse_model_action("not-json", observation)

    def test_parse_model_action_requires_state_assessment(self) -> None:
        observation = make_observation()
        with self.assertRaises(ValueError):
            _parse_model_action(
                '{"action_type":"increase_10","reasoning":"Metrics look healthy with low error and low latency, so continue the rollout.","slo_breach_detected":false}',
                observation,
            )

    def test_run_task_emits_only_start_step_end_lines_in_fallback_mode(self) -> None:
        stream = io.StringIO()
        with patch("inference.CanaryEnv", FakeEnv):
            with redirect_stdout(stream):
                result = asyncio.run(
                    _run_task(
                        client=None,
                        model_name=DEFAULT_MODEL_NAME,
                        env_base_url="http://127.0.0.1:7860",
                        local_image_name="",
                        task_id="easy",
                        proxy_required=False,
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
        self.assertFalse(result["attempted_model_call"])

    def test_model_failure_is_disclosed_and_does_not_look_like_model_success(self) -> None:
        stream = io.StringIO()
        with patch("inference.CanaryEnv", FakeEnv):
            with patch("inference._model_action", side_effect=RuntimeError("upstream timeout")):
                with redirect_stdout(stream):
                    result = asyncio.run(
                        _run_task(
                            client=object(),
                            model_name="test-model",
                            env_base_url="http://127.0.0.1:7860",
                            local_image_name="",
                            task_id="easy",
                            proxy_required=True,
                        )
                    )

        lines = [line.strip() for line in stream.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines[0], "[START] task=easy env=canary-release-env model=test-model")
        self.assertIn("error=model_call_failed:upstream timeout", lines[1])
        self.assertEqual(lines[2], "[END] success=false steps=1 score=0.8700 rewards=0.87")
        self.assertTrue(result["degraded"])
        self.assertTrue(result["attempted_model_call"])

    def test_proxy_required_zero_call_task_cannot_look_successful(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("inference.CanaryEnv", DoneEnv):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = asyncio.run(
                    _run_task(
                        client=object(),
                        model_name="test-model",
                        env_base_url="http://127.0.0.1:7860",
                        local_image_name="",
                        task_id="easy",
                        proxy_required=True,
                    )
                )

        lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines[0], "[START] task=easy env=canary-release-env model=test-model")
        self.assertEqual(lines[1], "[END] success=false steps=0 score=0.0000 rewards=")
        self.assertIn("proxy_error=no_model_call_attempted", stderr.getvalue())
        self.assertFalse(result["attempted_model_call"])
        self.assertEqual(result["outcome"], "error")

    def test_close_failure_does_not_break_stdout_contract(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("inference.CanaryEnv", CloseFailEnv):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = asyncio.run(
                    _run_task(
                        client=None,
                        model_name=DEFAULT_MODEL_NAME,
                        env_base_url="http://127.0.0.1:7860",
                        local_image_name="",
                        task_id="easy",
                        proxy_required=False,
                    )
                )

        lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(line.startswith(("[START]", "[STEP]", "[END]")) for line in lines))
        self.assertIn("close_error=close failed", stderr.getvalue())
        self.assertEqual(result["score"], 0.87)

    def test_validator_style_env_config_attempts_proxy_calls(self) -> None:
        factory = OpenAIFactory()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.dict(
            os.environ,
            {
                "API_BASE_URL": "https://proxy.example.invalid/v1",
                "MODEL_NAME": "validator-model",
                "API_KEY": "validator-key",
                "HF_TOKEN": "stale-hf-token",
            },
            clear=True,
        ):
            with patch("inference.OpenAI", factory), patch("inference.CanaryEnv", FakeEnv), patch(
                "inference.TASK_IDS",
                ["easy"],
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    results = asyncio.run(run("http://127.0.0.1:7860"))

        self.assertEqual(
            factory.init_calls,
            [{"base_url": "https://proxy.example.invalid/v1", "api_key": "validator-key"}],
        )
        self.assertEqual(len(factory.request_calls), 1)
        self.assertEqual(factory.request_calls[0]["model"], "validator-model")
        self.assertEqual(results[0]["attempted_model_call"], True)
        lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines[0], "[START] task=easy env=canary-release-env model=validator-model")
        self.assertEqual(lines[-1], "[END] success=true steps=1 score=0.8700 rewards=0.87")
        self.assertEqual(stderr.getvalue().strip(), "")

    def test_hf_token_only_config_still_attempts_proxy_calls(self) -> None:
        factory = OpenAIFactory()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.dict(
            os.environ,
            {
                "HF_TOKEN": "hf-validator-key",
            },
            clear=True,
        ):
            with patch("inference.OpenAI", factory), patch("inference.CanaryEnv", FakeEnv), patch(
                "inference.TASK_IDS",
                ["easy"],
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    results = asyncio.run(run("http://127.0.0.1:7860"))

        self.assertEqual(
            factory.init_calls,
            [{"base_url": DEFAULT_API_BASE_URL, "api_key": "hf-validator-key"}],
        )
        self.assertEqual(len(factory.request_calls), 1)
        self.assertEqual(factory.request_calls[0]["model"], DEFAULT_MODEL_NAME)
        self.assertTrue(results[0]["attempted_model_call"])
        self.assertEqual(stderr.getvalue().strip(), "")

    def test_partial_proxy_config_without_credentials_fails_honestly(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.dict(
            os.environ,
            {
                "MODEL_NAME": "validator-model",
            },
            clear=True,
        ):
            with patch("inference.CanaryEnv", FakeEnv), patch("inference.TASK_IDS", ["easy"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    results = asyncio.run(run("http://127.0.0.1:7860"))

        lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines[0], "[START] task=easy env=canary-release-env model=validator-model")
        self.assertEqual(lines[1], "[STEP] step=1 action=increase_10 reward=0.87 done=true error=null")
        self.assertEqual(lines[2], "[END] success=false steps=1 score=0.8700 rewards=0.87")
        self.assertIn("proxy_error=missing_proxy_credentials", stderr.getvalue())
        self.assertIn("proxy_error=no_model_call_attempted", stderr.getvalue())
        self.assertFalse(results[0]["attempted_model_call"])
        self.assertTrue(results[0]["degraded"])

    def test_client_init_failure_prevents_silent_proxy_bypass(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.dict(
            os.environ,
            {
                "HF_TOKEN": "validator-key",
            },
            clear=True,
        ):
            with patch("inference.OpenAI", side_effect=RuntimeError("client init failed")), patch(
                "inference.CanaryEnv",
                FakeEnv,
            ), patch("inference.TASK_IDS", ["easy"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    results = asyncio.run(run("http://127.0.0.1:7860"))

        lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines[0], f"[START] task=easy env=canary-release-env model={DEFAULT_MODEL_NAME}")
        self.assertEqual(lines[-1], "[END] success=false steps=1 score=0.8700 rewards=0.87")
        self.assertIn("proxy_error=client_init_failed:client init failed", stderr.getvalue())
        self.assertIn("proxy_error=no_model_call_attempted", stderr.getvalue())
        self.assertFalse(results[0]["attempted_model_call"])
        self.assertTrue(results[0]["degraded"])

    def test_no_proxy_config_keeps_honest_fallback_behavior(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.dict(os.environ, {}, clear=True):
            with patch("inference.CanaryEnv", FakeEnv), patch("inference.TASK_IDS", ["easy"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    results = asyncio.run(run("http://127.0.0.1:7860"))

        lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(lines[0], "[START] task=easy env=canary-release-env model=fallback")
        self.assertEqual(lines[-1], "[END] success=true steps=1 score=0.8700 rewards=0.87")
        self.assertEqual(stderr.getvalue().strip(), "")
        self.assertFalse(results[0]["attempted_model_call"])


if __name__ == "__main__":
    unittest.main()
