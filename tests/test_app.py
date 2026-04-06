import subprocess
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from canary_release_env.server.app import app


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_validator_safe_http_endpoints_are_available(self) -> None:
        health = self.client.get("/health")
        tasks = self.client.get("/tasks")
        state = self.client.get("/state")
        reset = self.client.post("/reset", json={"task": "medium"})

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")
        self.assertEqual(tasks.status_code, 200)
        self.assertIn("tasks", tasks.json())
        self.assertEqual(state.status_code, 200)
        self.assertIn("episode_id", state.json())
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["observation"]["task_id"], "medium")

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
                }
            },
        )
        self.assertEqual(step_response.status_code, 200)
        body = step_response.json()

        self.assertEqual(body["observation"]["task_id"], "uninitialized")
        self.assertTrue(body["done"])
        self.assertEqual(body["observation"]["policy_assessment"], "reset_required")

    def test_invalid_task_is_explicit_in_reset_and_rejected_by_episode_helper(self) -> None:
        reset_response = self.client.post("/reset", json={"task": "unknown-task"})
        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(reset_response.json()["observation"]["task_id"], "easy")
        self.assertIn("Unknown task 'unknown-task' requested", reset_response.json()["observation"]["feedback"])

        episode_response = self.client.post("/episodes", json={"task": "unknown-task"})
        self.assertEqual(episode_response.status_code, 422)
        self.assertIn("Valid task_ids", episode_response.json()["detail"])

    def test_episodes_endpoints_preserve_state_and_return_episode_summary(self) -> None:
        create_response = self.client.post("/episodes", json={"task": "hard"})
        self.assertEqual(create_response.status_code, 200)
        episode_id = create_response.json()["episode_id"]

        actions = [
            {
                "action_type": "increase_10",
                "reasoning": "Metrics are healthy with low error and no differential latency, so increase traffic by 10%.",
                "slo_breach_detected": False,
            },
            {
                "action_type": "increase_10",
                "reasoning": "Metrics remain healthy with stable canary performance, so continue the 10% rollout plan.",
                "slo_breach_detected": False,
            },
            {
                "action_type": "increase_10",
                "reasoning": "Canary and stable remain aligned, so the rollout can move to the next checkpoint.",
                "slo_breach_detected": False,
            },
            {
                "action_type": "hold",
                "reasoning": "Stable and canary latencies spiked together with no differential gap, so hold for confirmation.",
                "slo_breach_detected": False,
            },
            {
                "action_type": "hold",
                "reasoning": "The canary still has elevated differential latency after the noise event, so hold for one more observation.",
                "slo_breach_detected": False,
            },
            {
                "action_type": "rollback",
                "reasoning": "The canary now breaches the differential latency threshold, so rollback immediately.",
                "slo_breach_detected": True,
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

    def test_grader_endpoint_returns_normalized_score_and_breakdown(self) -> None:
        response = self.client.post(
            "/grader",
            json={
                "task_id": "easy",
                "action": {
                    "action_type": "increase_10",
                    "reasoning": "Increasing traffic during a breach to gather more data.",
                    "slo_breach_detected": False,
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
        self.assertGreaterEqual(body["average"], 0.0)
        self.assertLessEqual(body["average"], 1.0)
        self.assertGreater(body["scores"]["easy"], body["scores"]["medium"])
        self.assertGreater(body["scores"]["medium"], body["scores"]["hard"])
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
