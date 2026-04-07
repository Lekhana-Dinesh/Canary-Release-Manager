import subprocess
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.app import app


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_validator_safe_http_endpoints_are_available(self) -> None:
        health = self.client.get("/health")
        tasks = self.client.get("/tasks")
        state = self.client.get("/state")
        reset = self.client.post("/reset", json={"task": "medium", "seed": 0})

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")
        self.assertEqual(tasks.status_code, 200)
        self.assertIn("tasks", tasks.json())
        self.assertEqual(
            tasks.json()["notes"]["recommended_plain_http_interface"],
            "/episodes",
        )
        self.assertEqual(
            tasks.json()["notes"]["recommended_plain_http_flow"],
            "POST /episodes -> POST /episodes/{episode_id}/step -> GET /episodes/{episode_id}/transcript",
        )
        self.assertIn(
            "coarse labels",
            tasks.json()["notes"]["state_assessment_contract"],
        )
        self.assertIn(
            "does not affect the score",
            tasks.json()["notes"]["reasoning_contract"],
        )
        self.assertEqual(state.status_code, 200)
        self.assertIn("episode_id", state.json())
        self.assertEqual(reset.status_code, 200)
        observation = reset.json()["observation"]
        self.assertNotIn("task_id", observation)
        self.assertNotIn("policy_assessment", observation)
        self.assertEqual(observation["rollback_on_error_rate"], 0.005)
        self.assertEqual(observation["rollback_on_canary_p99_ms"], 200.0)
        self.assertIsNone(observation["rollback_on_differential_p99_ms"])

    def test_server_app_import_path_works_from_repo_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", "import server.app; print('ok')"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok", completed.stdout)

    def test_standard_http_step_does_not_silently_default_to_easy(self) -> None:
        reset_response = self.client.post("/reset", json={"task": "hard"})
        self.assertEqual(reset_response.status_code, 200)

        step_response = self.client.post(
            "/step",
            json={
                "action": {
                    "action_type": "increase_10",
                    "reasoning": "Metrics are healthy and traffic can increase safely.",
                    "slo_breach_detected": False,
                    "state_assessment": "healthy",
                }
            },
        )
        self.assertEqual(step_response.status_code, 200)
        body = step_response.json()

        self.assertTrue(body["done"])
        self.assertEqual(body["observation"]["step_number"], 0)
        self.assertNotIn("policy_assessment", body["observation"])
        self.assertIsNone(body["observation"]["rollback_on_canary_p99_ms"])

    def test_invalid_task_is_explicit_in_reset_and_rejected_by_episode_helper(self) -> None:
        reset_response = self.client.post("/reset", json={"task": "unknown-task"})
        self.assertEqual(reset_response.status_code, 200)
        observation = reset_response.json()["observation"]
        self.assertEqual(observation["rollback_on_canary_p99_ms"], 200.0)
        self.assertIsNone(observation["rollback_on_differential_p99_ms"])

        episode_response = self.client.post("/episodes", json={"task": "unknown-task"})
        self.assertEqual(episode_response.status_code, 422)
        self.assertIn("Valid task_ids", episode_response.json()["detail"])

    def test_seeded_reset_is_reproducible_and_changes_metrics(self) -> None:
        hard_seed_a = self.client.post("/reset", json={"task": "hard", "seed": 17})
        hard_seed_b = self.client.post("/reset", json={"task": "hard", "seed": 17})
        hard_seed_c = self.client.post("/reset", json={"task": "hard", "seed": 18})

        self.assertEqual(hard_seed_a.status_code, 200)
        self.assertEqual(hard_seed_b.status_code, 200)
        self.assertEqual(hard_seed_c.status_code, 200)

        observation_a = hard_seed_a.json()["observation"]
        observation_b = hard_seed_b.json()["observation"]
        observation_c = hard_seed_c.json()["observation"]

        self.assertEqual(observation_a, observation_b)
        self.assertNotEqual(
            (
                observation_a["canary_p99_ms"],
                observation_a["stable_p99_ms"],
                observation_a["canary_error_rate"],
            ),
            (
                observation_c["canary_p99_ms"],
                observation_c["stable_p99_ms"],
                observation_c["canary_error_rate"],
            ),
        )

    def test_episodes_endpoints_preserve_state_and_return_episode_summary(self) -> None:
        create_response = self.client.post("/episodes", json={"task": "hard", "seed": 0})
        self.assertEqual(create_response.status_code, 200)
        episode_id = create_response.json()["episode_id"]
        self.assertEqual(create_response.json()["episode_context"]["task_id"], "hard")

        actions = [
            {
                "action_type": "increase_10",
                "reasoning": "Metrics are healthy with low error and no differential latency, so increase traffic by 10%.",
                "slo_breach_detected": False,
                "state_assessment": "healthy",
            },
            {
                "action_type": "increase_10",
                "reasoning": "Metrics remain healthy with stable canary performance, so continue the 10% rollout plan.",
                "slo_breach_detected": False,
                "state_assessment": "healthy",
            },
            {
                "action_type": "increase_10",
                "reasoning": "Canary and stable remain aligned, so the rollout can move to the next checkpoint.",
                "slo_breach_detected": False,
                "state_assessment": "healthy",
            },
            {
                "action_type": "hold",
                "reasoning": "Stable and canary latencies spiked together with no differential gap, so hold for confirmation.",
                "slo_breach_detected": False,
                "state_assessment": "noise",
            },
            {
                "action_type": "hold",
                "reasoning": "The canary still has elevated differential latency after the noise event, so hold for one more observation.",
                "slo_breach_detected": False,
                "state_assessment": "warning",
            },
            {
                "action_type": "rollback",
                "reasoning": "The canary now breaches the differential latency threshold, so rollback immediately.",
                "slo_breach_detected": True,
                "state_assessment": "breach",
            },
        ]

        final_step = None
        for action in actions:
            final_step = self.client.post(
                f"/episodes/{episode_id}/step",
                json={"action": action},
            )
            self.assertEqual(final_step.status_code, 200)

        assert final_step is not None
        self.assertTrue(final_step.json()["done"])
        self.assertIn("evaluation", final_step.json())
        self.assertEqual(final_step.json()["evaluation"]["policy_assessment"], "correct_rollback")
        episode_result = final_step.json()["episode_result"]

        self.assertIn("score_breakdown", episode_result)
        self.assertIn("episode_summary", episode_result)
        self.assertEqual(
            episode_result["rollback_assessment"]["relative_to_benchmark_point"],
            "timely",
        )
        self.assertEqual(
            episode_result["rollback_assessment"]["relative_to_first_breach"],
            "on_first_breach",
        )
        self.assertIsNotNone(episode_result["first_breach_point"])

        transcript = episode_result["transcript"]
        self.assertGreaterEqual(len(transcript), 1)
        self.assertNotIn("policy_assessment", final_step.json()["observation"])
        first_entry = transcript[0]
        self.assertEqual(
            set(first_entry),
            {
                "decision_number",
                "pre_observation",
                "action",
                "step_score",
                "actual_breach",
                "policy_assessment",
                "reward_breakdown",
                "explanation",
                "post_observation",
            },
        )
        self.assertNotIn("metadata", first_entry["action"])

    def test_root_and_server_dockerfiles_stay_in_sync(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root_dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8").splitlines()
        server_dockerfile = (repo_root / "server" / "Dockerfile").read_text(encoding="utf-8").splitlines()
        self.assertEqual(root_dockerfile, server_dockerfile)

    def test_grader_endpoint_returns_normalized_score_and_breakdown(self) -> None:
        response = self.client.post(
            "/grader",
            json={
                "task_id": "easy",
                "action": {
                    "action_type": "increase_10",
                    "reasoning": "Increasing traffic during a breach to gather more data.",
                    "slo_breach_detected": False,
                    "state_assessment": "healthy",
                },
                "traffic_pct": 0.20,
                "canary_error_rate": 0.009,
                "canary_p99_ms": 220.0,
                "stable_error_rate": 0.001,
                "stable_p99_ms": 140.0,
                "alert_count": 1,
                "step_number": 2,
                "consecutive_holds": 0,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertGreaterEqual(body["total_score"], 0.0)
        self.assertLessEqual(body["total_score"], 1.0)
        self.assertEqual(
            set(body["reward_breakdown"]),
            {
                "breach_detection_score",
                "rollback_timing_score",
                "promotion_safety_score",
                "reasoning_score",
            },
        )

    def test_baseline_endpoint_keeps_scores_normalized_and_hard_is_lower(self) -> None:
        response = self.client.post("/baseline")
        self.assertEqual(response.status_code, 200)
        body = response.json()

        for score in body["scores"].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        self.assertIn("silent", body["scores"])
        self.assertGreaterEqual(body["average"], 0.0)
        self.assertLessEqual(body["average"], 1.0)
        self.assertGreater(body["scores"]["easy"], body["scores"]["medium"])
        self.assertGreater(body["scores"]["medium"], body["scores"]["hard"])
        self.assertGreater(body["scores"]["recovery"], body["scores"]["expert"])
        self.assertEqual(
            body["details"]["medium"]["rollback_assessment"]["relative_to_first_breach"],
            "on_first_breach",
        )
        self.assertEqual(
            body["details"]["medium"]["rollback_assessment"]["relative_to_benchmark_point"],
            "late",
        )
        self.assertEqual(
            body["details"]["hard"]["rollback_assessment"]["relative_to_first_breach"],
            "on_first_breach",
        )
        self.assertEqual(
            body["details"]["hard"]["rollback_assessment"]["relative_to_benchmark_point"],
            "late",
        )


if __name__ == "__main__":
    unittest.main()
