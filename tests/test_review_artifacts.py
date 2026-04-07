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
    benchmark_seed_sweep,
    scenario_variant_catalog,
)
from models import CanaryAction
from server.canary_environment import CanaryEnvironment


def _run_actions(task_id: str, actions: list[dict]) -> tuple[dict, dict]:
    env = CanaryEnvironment()
    observation = env.reset(task=task_id)
    for action in actions:
        observation = env.step(CanaryAction(**action))
    return env.episode_result(), observation.model_dump()


def _event_step(task_id: str, seed: int, matcher) -> int | None:
    env = CanaryEnvironment()
    observation = env.reset(task=task_id, seed=seed)
    while not observation.done and observation.step_number < 8:
        observation = env.step(
            CanaryAction(
                action_type="increase_10",
                reasoning=(
                    "Metrics are healthy with low error and manageable latency, so take the standard 10% promotion step."
                ),
                slo_breach_detected=False,
                state_assessment="healthy",
            )
        )
        if matcher(observation):
            return int(observation.step_number)
    return None


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
            "expert_transcript": "expert_cautious_transcript.json",
            "expert_grader": "expert_phantom_hold_grader.json",
            "recovery_transcript": "recovery_cautious_transcript.json",
            "recovery_grader": "recovery_transient_hold_grader.json",
            "seed_sweep": "policy_seed_sweep_results.json",
            "variant_catalog": "scenario_variant_catalog.json",
            "validation": "openenv_validation_results.json",
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
                "state_assessment": "warning",
            },
        )
        self.assertEqual(grader_request["consecutive_holds"], 1)

    def test_policy_benchmark_keeps_scores_normalized_and_covers_expert_recovery_and_silent(self) -> None:
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
        expert_scores = {
            name: result["scores"]["expert"]
            for name, result in policies.items()
        }
        recovery_scores = {
            name: result["scores"]["recovery"]
            for name, result in policies.items()
        }
        silent_scores = {
            name: result["scores"]["silent"]
            for name, result in policies.items()
        }
        self.assertGreater(hard_scores["cautious_policy"], hard_scores["shallow_baseline"])
        self.assertGreater(hard_scores["shallow_baseline"], hard_scores["aggressive_policy"])
        self.assertGreater(expert_scores["cautious_policy"], expert_scores["shallow_baseline"])
        self.assertGreater(recovery_scores["cautious_policy"], recovery_scores["aggressive_policy"])
        self.assertGreater(silent_scores["cautious_policy"], silent_scores["shallow_baseline"])
        self.assertGreater(silent_scores["shallow_baseline"], silent_scores["aggressive_policy"])
        self.assertEqual(benchmark["hard_task_ordering"][0], "cautious_policy")
        self.assertEqual(benchmark["recovery_task_ordering"][0], "cautious_policy")
        self.assertEqual(benchmark["silent_task_ordering"][0], "cautious_policy")

    def test_seed_sweep_keeps_policy_ordering_coherent_across_deterministic_variants(self) -> None:
        seed_sweep = benchmark_seed_sweep()
        policies = seed_sweep["policies"]

        self.assertEqual(seed_sweep["aggregate_hard_ordering"][0], "cautious_policy")
        self.assertEqual(seed_sweep["aggregate_expert_ordering"][0], "cautious_policy")
        self.assertEqual(seed_sweep["aggregate_recovery_ordering"][0], "cautious_policy")
        self.assertEqual(seed_sweep["aggregate_silent_ordering"][0], "cautious_policy")

        for seed in seed_sweep["seeds"]:
            seed_key = str(seed)
            hard_cautious = policies["cautious_policy"]["per_seed"][seed_key]["scores"]["hard"]
            hard_shallow = policies["shallow_baseline"]["per_seed"][seed_key]["scores"]["hard"]
            expert_cautious = policies["cautious_policy"]["per_seed"][seed_key]["scores"]["expert"]
            expert_shallow = policies["shallow_baseline"]["per_seed"][seed_key]["scores"]["expert"]
            recovery_cautious = policies["cautious_policy"]["per_seed"][seed_key]["scores"]["recovery"]
            recovery_aggressive = policies["aggressive_policy"]["per_seed"][seed_key]["scores"]["recovery"]
            silent_cautious = policies["cautious_policy"]["per_seed"][seed_key]["scores"]["silent"]
            silent_shallow = policies["shallow_baseline"]["per_seed"][seed_key]["scores"]["silent"]

            self.assertGreaterEqual(hard_cautious, hard_shallow)
            self.assertGreater(expert_cautious, expert_shallow)
            self.assertGreater(recovery_cautious, recovery_aggressive)
            self.assertGreater(silent_cautious, silent_shallow)

    def test_seeded_variants_change_when_special_events_arrive(self) -> None:
        hard_noise_seed_0 = _event_step(
            "hard",
            0,
            lambda observation: (
                observation.canary_p99_ms >= 180.0
                and observation.stable_p99_ms >= 180.0
                and observation.differential_p99_ms <= 15.0
                and observation.alert_count == 0
            ),
        )
        hard_noise_seed_7 = _event_step(
            "hard",
            7,
            lambda observation: (
                observation.canary_p99_ms >= 180.0
                and observation.stable_p99_ms >= 180.0
                and observation.differential_p99_ms <= 15.0
                and observation.alert_count == 0
            ),
        )
        expert_phantom_seed_0 = _event_step(
            "expert",
            0,
            lambda observation: (
                observation.alert_count >= 2
                and observation.canary_error_rate <= observation.rollback_on_error_rate
                and observation.differential_p99_ms <= 15.0
            ),
        )
        expert_phantom_seed_11 = _event_step(
            "expert",
            11,
            lambda observation: (
                observation.alert_count >= 2
                and observation.canary_error_rate <= observation.rollback_on_error_rate
                and observation.differential_p99_ms <= 15.0
            ),
        )

        self.assertEqual(hard_noise_seed_0, 3)
        self.assertEqual(hard_noise_seed_7, 2)
        self.assertEqual(expert_phantom_seed_0, 2)
        self.assertEqual(expert_phantom_seed_11, 3)

    def test_recovery_seeded_variants_shift_transient_window_timing(self) -> None:
        recovery_seed_0 = _event_step(
            "recovery",
            0,
            lambda observation: (
                observation.differential_p99_ms >= 20.0
                and observation.differential_p99_ms < observation.rollback_on_differential_p99_ms
                and observation.alert_count == 0
            ),
        )
        recovery_seed_19 = _event_step(
            "recovery",
            19,
            lambda observation: (
                observation.differential_p99_ms >= 20.0
                and observation.differential_p99_ms < observation.rollback_on_differential_p99_ms
                and observation.alert_count == 0
            ),
        )

        self.assertEqual(recovery_seed_0, 3)
        self.assertNotEqual(recovery_seed_0, recovery_seed_19)

    def test_silent_seeded_variants_shift_breach_timing(self) -> None:
        silent_breach_seed_0 = _event_step(
            "silent",
            0,
            lambda observation: (
                observation.differential_p99_ms > observation.rollback_on_differential_p99_ms
                and observation.alert_count == 0
            ),
        )
        silent_breach_seed_3 = _event_step(
            "silent",
            3,
            lambda observation: (
                observation.differential_p99_ms > observation.rollback_on_differential_p99_ms
                and observation.alert_count == 0
            ),
        )

        self.assertEqual(silent_breach_seed_0, 4)
        self.assertNotEqual(silent_breach_seed_0, silent_breach_seed_3)

    def test_variant_catalog_reports_seed_diversity_for_all_public_tasks(self) -> None:
        catalog = scenario_variant_catalog()

        self.assertIn("silent", catalog["tasks"])
        self.assertIn("task_summaries", catalog)
        self.assertIn("distinct_breach_steps", catalog["task_summaries"]["silent"])
        self.assertGreaterEqual(len(catalog["task_summaries"]["hard"]["distinct_noise_steps"]), 2)
        self.assertGreaterEqual(len(catalog["task_summaries"]["expert"]["distinct_phantom_steps"]), 2)
        self.assertGreaterEqual(len(catalog["task_summaries"]["recovery"]["distinct_recovery_clear_steps"]), 1)
        self.assertGreaterEqual(len(catalog["task_summaries"]["silent"]["distinct_breach_steps"]), 2)


if __name__ == "__main__":
    unittest.main()
