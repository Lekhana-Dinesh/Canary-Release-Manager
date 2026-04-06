import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_review_artifacts import (
    AMBIGUOUS_HARD_ACTIONS,
    ARTIFACT_FILENAMES,
    BAD_HARD_ACTIONS,
    GOOD_HARD_ACTIONS,
    _build_grader_request,
    benchmark_policies,
)
from models import CanaryAction
from server.canary_environment import CanaryEnvironment


def _run_actions(task_id: str, actions: list[dict]) -> tuple[dict, dict]:
    env = CanaryEnvironment()
    observation = env.reset(task=task_id)
    for action in actions:
        observation = env.step(CanaryAction(**action))
    return env.episode_result(), observation.model_dump()


class ReviewArtifactTests(unittest.TestCase):
    def test_artifact_filenames_are_explicit_and_reviewer_friendly(self) -> None:
        expected = {
            "inference_stdout": "inference_stdout_fallback_sample.txt",
            "cautious_transcript": "hard_cautious_transcript.json",
            "cautious_grader": "hard_cautious_grader.json",
            "aggressive_transcript": "hard_aggressive_transcript.json",
            "aggressive_grader": "hard_aggressive_grader.json",
            "watch_window_transcript": "hard_watch_window_hold_transcript.json",
            "watch_window_grader": "hard_watch_window_hold_grader.json",
            "manifest": "artifact_manifest.json",
            "index": "artifact_index.md",
        }
        for key, value in expected.items():
            self.assertEqual(ARTIFACT_FILENAMES[key], value)

    def test_reference_hard_runs_are_meaningfully_ordered(self) -> None:
        good_result, _ = _run_actions("hard", GOOD_HARD_ACTIONS)
        bad_result, _ = _run_actions("hard", BAD_HARD_ACTIONS)
        ambiguous_result, _ = _run_actions("hard", AMBIGUOUS_HARD_ACTIONS)

        self.assertEqual(good_result["outcome"], "rollback")
        self.assertEqual(bad_result["outcome"], "rollback")
        self.assertEqual(ambiguous_result["outcome"], "in_progress")
        self.assertGreater(good_result["episode_score"], bad_result["episode_score"])
        self.assertEqual(
            good_result["rollback_assessment"]["relative_to_benchmark_point"],
            "timely",
        )
        self.assertEqual(
            bad_result["rollback_assessment"]["relative_to_benchmark_point"],
            "late",
        )

    def test_transcript_snapshots_include_consecutive_holds_for_replay(self) -> None:
        result, current_observation = _run_actions("hard", AMBIGUOUS_HARD_ACTIONS)
        last_entry = result["transcript"][-1]

        self.assertIn("consecutive_holds", last_entry["pre_observation"])
        self.assertIn("consecutive_holds", last_entry["post_observation"])
        self.assertEqual(last_entry["pre_observation"]["consecutive_holds"], 0)
        self.assertEqual(last_entry["post_observation"]["consecutive_holds"], 1)

        grader_request = _build_grader_request(
            task_id="hard",
            observation=current_observation,
            action={
                "action_type": "hold",
                "reasoning": "The canary is still slower than stable after the shared-noise event, so hold for one more confirmation step.",
                "slo_breach_detected": False,
            },
        )
        self.assertEqual(grader_request["consecutive_holds"], 1)

    def test_policy_benchmark_keeps_scores_normalized_and_hard_is_discriminative(self) -> None:
        benchmark = benchmark_policies()
        policies = benchmark["policies"]

        for policy_result in policies.values():
            self.assertGreaterEqual(policy_result["average"], 0.0)
            self.assertLessEqual(policy_result["average"], 1.0)
            for score in policy_result["scores"].values():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

        hard_scores = {
            name: result["scores"]["hard"]
            for name, result in policies.items()
        }
        self.assertGreater(hard_scores["cautious_policy"], hard_scores["shallow_baseline"])
        self.assertGreater(hard_scores["shallow_baseline"], hard_scores["aggressive_policy"])
        self.assertEqual(benchmark["hard_task_ordering"][0], "cautious_policy")


if __name__ == "__main__":
    unittest.main()
