import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CanaryAction
from server.canary_environment import CanaryEnvironment
from server.grader import grade
from server.scenarios import SCENARIOS


def make_action(
    action_type: str,
    reasoning: str,
    breach: bool,
    state_assessment: str = "healthy",
) -> CanaryAction:
    return CanaryAction(
        action_type=action_type,
        reasoning=reasoning,
        slo_breach_detected=breach,
        state_assessment=state_assessment,
    )


class GraderTests(unittest.TestCase):
    def test_grader_scores_are_normalized_between_zero_and_one(self) -> None:
        bad_action = make_action(
            "increase_10",
            "Increasing traffic during a breach to gather more data.",
            breach=False,
            state_assessment="healthy",
        )
        good_action = make_action(
            "increase_10",
            "Canary and stable are healthy with low error, low latency, and a safe 10% promotion step.",
            breach=False,
            state_assessment="healthy",
        )

        bad_result = grade(
            bad_action,
            traffic_pct=0.20,
            canary_error=0.009,
            canary_p99=220.0,
            stable_error=0.001,
            stable_p99=140.0,
            alert_count=1,
            scenario=SCENARIOS["easy"],
            step_number=2,
            consecutive_holds=0,
        )
        good_result = grade(
            good_action,
            traffic_pct=0.10,
            canary_error=0.001,
            canary_p99=142.0,
            stable_error=0.001,
            stable_p99=142.0,
            alert_count=0,
            scenario=SCENARIOS["hard"],
            step_number=1,
            consecutive_holds=0,
        )

        self.assertGreaterEqual(bad_result.total_score, 0.0)
        self.assertLessEqual(bad_result.total_score, 1.0)
        self.assertGreaterEqual(good_result.total_score, 0.0)
        self.assertLessEqual(good_result.total_score, 1.0)
        self.assertLess(bad_result.total_score, good_result.total_score)

    def test_rollback_is_graded_against_pre_action_breach_state(self) -> None:
        env = CanaryEnvironment()
        observation = env.reset(task="easy")

        observation = env.step(
            make_action(
                "increase_10",
                "Metrics are healthy and traffic can increase safely.",
                breach=False,
                state_assessment="healthy",
            )
        )
        observation = env.step(
            make_action(
                "increase_5",
                "Metrics remain healthy and a cautious increase is acceptable.",
                breach=False,
                state_assessment="healthy",
            )
        )

        self.assertGreater(observation.canary_error_rate, 0.005)

        observation = env.step(
            make_action(
                "rollback",
                "The canary error rate exceeds the threshold, so rollback is required immediately.",
                breach=True,
                state_assessment="breach",
            )
        )
        evaluation = env.last_evaluation()

        self.assertIsNotNone(evaluation)
        assert evaluation is not None
        self.assertEqual(evaluation["policy_assessment"], "correct_rollback")
        self.assertTrue(evaluation["actual_breach"])
        self.assertGreaterEqual(evaluation["step_reward"], 0.85)

    def test_hold_during_hard_task_global_noise_scores_better_than_promotion(self) -> None:
        hold = make_action(
            "hold",
            "Stable and canary latencies spiked together with no differential gap, so hold.",
            breach=False,
            state_assessment="noise",
        )
        increase = make_action(
            "increase_10",
            "Latency is elevated but traffic should keep moving forward.",
            breach=False,
            state_assessment="healthy",
        )

        hold_result = grade(
            hold,
            traffic_pct=0.30,
            canary_error=0.001,
            canary_p99=186.0,
            stable_error=0.001,
            stable_p99=186.0,
            alert_count=0,
            scenario=SCENARIOS["hard"],
            step_number=3,
            consecutive_holds=0,
        )
        increase_result = grade(
            increase,
            traffic_pct=0.30,
            canary_error=0.001,
            canary_p99=186.0,
            stable_error=0.001,
            stable_p99=186.0,
            alert_count=0,
            scenario=SCENARIOS["hard"],
            step_number=3,
            consecutive_holds=0,
        )

        self.assertTrue(hold_result.is_global_noise)
        self.assertEqual(hold_result.policy_assessment, "correct_noise_hold")
        self.assertGreater(hold_result.total_score, increase_result.total_score)

    def test_hard_warning_window_rewards_hold_over_increase(self) -> None:
        hold = make_action(
            "hold",
            "The canary stayed slower than stable after the shared noise event, so hold for confirmation.",
            breach=False,
            state_assessment="warning",
        )
        increase = make_action(
            "increase_10",
            "The canary has not breached yet, so continue the promotion.",
            breach=False,
            state_assessment="healthy",
        )

        hold_result = grade(
            hold,
            traffic_pct=0.30,
            canary_error=0.0026,
            canary_p99=168.0,
            stable_error=0.001,
            stable_p99=144.0,
            alert_count=0,
            scenario=SCENARIOS["hard"],
            step_number=4,
            consecutive_holds=1,
        )
        increase_result = grade(
            increase,
            traffic_pct=0.30,
            canary_error=0.0026,
            canary_p99=168.0,
            stable_error=0.001,
            stable_p99=144.0,
            alert_count=0,
            scenario=SCENARIOS["hard"],
            step_number=4,
            consecutive_holds=1,
        )

        self.assertEqual(hold_result.policy_assessment, "correct_warning_hold")
        self.assertGreater(hold_result.total_score, increase_result.total_score)

    def test_expert_phantom_alert_rewards_metric_verification_over_rollback(self) -> None:
        hold = make_action(
            "hold",
            "Alert count is elevated but the metrics are still healthy with only a 6ms differential, so hold and verify before rollback.",
            breach=False,
            state_assessment="phantom_alert",
        )
        rollback = make_action(
            "rollback",
            "Alert count is elevated, so rollback immediately.",
            breach=True,
            state_assessment="breach",
        )

        hold_result = grade(
            hold,
            traffic_pct=0.10,
            canary_error=0.0018,
            canary_p99=146.0,
            stable_error=0.001,
            stable_p99=140.0,
            alert_count=4,
            scenario=SCENARIOS["expert"],
            step_number=2,
            consecutive_holds=0,
        )
        rollback_result = grade(
            rollback,
            traffic_pct=0.10,
            canary_error=0.0018,
            canary_p99=146.0,
            stable_error=0.001,
            stable_p99=140.0,
            alert_count=4,
            scenario=SCENARIOS["expert"],
            step_number=2,
            consecutive_holds=0,
        )

        self.assertTrue(hold_result.is_phantom_alert)
        self.assertEqual(hold_result.policy_assessment, "correct_phantom_ignore")
        self.assertEqual(rollback_result.policy_assessment, "false_positive_phantom_rollback")
        self.assertGreater(hold_result.total_score, rollback_result.total_score)

    def test_reasoning_score_is_fully_structured_and_ignores_prose_detail(self) -> None:
        generic_hold = make_action(
            "hold",
            "Holding because it feels safer to wait for more information.",
            breach=False,
            state_assessment="phantom_alert",
        )
        specific_hold = make_action(
            "hold",
            "Alert count is elevated but canary_error_rate is still 0.0018 and differential_p99 is only 6ms, so hold and verify before acting on alerts alone.",
            breach=False,
            state_assessment="phantom_alert",
        )

        generic_result = grade(
            generic_hold,
            traffic_pct=0.10,
            canary_error=0.0018,
            canary_p99=146.0,
            stable_error=0.001,
            stable_p99=140.0,
            alert_count=4,
            scenario=SCENARIOS["expert"],
            step_number=2,
            consecutive_holds=0,
        )
        specific_result = grade(
            specific_hold,
            traffic_pct=0.10,
            canary_error=0.0018,
            canary_p99=146.0,
            stable_error=0.001,
            stable_p99=140.0,
            alert_count=4,
            scenario=SCENARIOS["expert"],
            step_number=2,
            consecutive_holds=0,
        )

        self.assertEqual(generic_result.reasoning_score, specific_result.reasoning_score)
        self.assertAlmostEqual(specific_result.reasoning_score, 0.10, places=4)

    def test_recovery_warning_prefers_hold_over_rollback_and_large_promotion(self) -> None:
        hold = make_action(
            "hold",
            "The canary is about 33ms slower than stable with no breach yet, so hold to confirm whether the transient latency spike clears.",
            breach=False,
            state_assessment="warning",
        )
        rollback = make_action(
            "rollback",
            "Latency looks elevated, so rollback immediately.",
            breach=True,
            state_assessment="breach",
        )
        promote = make_action(
            "increase_10",
            "It has not breached yet, so increase traffic by 10%.",
            breach=False,
            state_assessment="healthy",
        )

        hold_result = grade(
            hold,
            traffic_pct=0.30,
            canary_error=0.0027,
            canary_p99=176.0,
            stable_error=0.001,
            stable_p99=143.0,
            alert_count=0,
            scenario=SCENARIOS["recovery"],
            step_number=3,
            consecutive_holds=0,
        )
        rollback_result = grade(
            rollback,
            traffic_pct=0.30,
            canary_error=0.0027,
            canary_p99=176.0,
            stable_error=0.001,
            stable_p99=143.0,
            alert_count=0,
            scenario=SCENARIOS["recovery"],
            step_number=3,
            consecutive_holds=0,
        )
        promote_result = grade(
            promote,
            traffic_pct=0.30,
            canary_error=0.0027,
            canary_p99=176.0,
            stable_error=0.001,
            stable_p99=143.0,
            alert_count=0,
            scenario=SCENARIOS["recovery"],
            step_number=3,
            consecutive_holds=0,
        )

        self.assertEqual(hold_result.policy_assessment, "correct_recovery_hold")
        self.assertEqual(rollback_result.policy_assessment, "premature_recovery_rollback")
        self.assertEqual(promote_result.policy_assessment, "risky_recovery_promotion")
        self.assertGreater(hold_result.total_score, rollback_result.total_score)
        self.assertGreater(hold_result.total_score, promote_result.total_score)

    def test_structured_state_assessment_outweighs_generic_metric_language(self) -> None:
        wrong_label = make_action(
            "hold",
            "Both services are around 186ms with almost no differential gap, so hold because the latency evidence points to shared infrastructure noise.",
            breach=False,
            state_assessment="healthy",
        )
        correct_label = make_action(
            "hold",
            "Both services are around 186ms with almost no differential gap, so hold because the latency evidence points to shared infrastructure noise.",
            breach=False,
            state_assessment="noise",
        )

        wrong_result = grade(
            wrong_label,
            traffic_pct=0.30,
            canary_error=0.001,
            canary_p99=186.0,
            stable_error=0.001,
            stable_p99=186.0,
            alert_count=0,
            scenario=SCENARIOS["hard"],
            step_number=3,
            consecutive_holds=0,
        )
        correct_result = grade(
            correct_label,
            traffic_pct=0.30,
            canary_error=0.001,
            canary_p99=186.0,
            stable_error=0.001,
            stable_p99=186.0,
            alert_count=0,
            scenario=SCENARIOS["hard"],
            step_number=3,
            consecutive_holds=0,
        )

        self.assertEqual(wrong_result.reasoning_score, 0.03)
        self.assertEqual(correct_result.reasoning_score, 0.10)

    def test_silent_warning_prefers_hold_over_promotion_and_premature_rollback(self) -> None:
        hold = make_action(
            "hold",
            "The canary is materially slower than stable even without alert confirmation, so hold for one more observation.",
            breach=False,
            state_assessment="warning",
        )
        promote = make_action(
            "increase_10",
            "No alert has fired yet, so continue the promotion.",
            breach=False,
            state_assessment="healthy",
        )
        rollback = make_action(
            "rollback",
            "The canary seems suspicious, so rollback now.",
            breach=True,
            state_assessment="breach",
        )

        hold_result = grade(
            hold,
            traffic_pct=0.30,
            canary_error=0.0018,
            canary_p99=171.0,
            stable_error=0.001,
            stable_p99=145.0,
            alert_count=0,
            scenario=SCENARIOS["silent"],
            step_number=3,
            consecutive_holds=0,
        )
        promote_result = grade(
            promote,
            traffic_pct=0.30,
            canary_error=0.0018,
            canary_p99=171.0,
            stable_error=0.001,
            stable_p99=145.0,
            alert_count=0,
            scenario=SCENARIOS["silent"],
            step_number=3,
            consecutive_holds=0,
        )
        rollback_result = grade(
            rollback,
            traffic_pct=0.30,
            canary_error=0.0018,
            canary_p99=171.0,
            stable_error=0.001,
            stable_p99=145.0,
            alert_count=0,
            scenario=SCENARIOS["silent"],
            step_number=3,
            consecutive_holds=0,
        )

        self.assertEqual(hold_result.policy_assessment, "correct_silent_hold")
        self.assertEqual(promote_result.policy_assessment, "risky_silent_promotion")
        self.assertEqual(rollback_result.policy_assessment, "premature_silent_rollback")
        self.assertGreater(hold_result.total_score, promote_result.total_score)
        self.assertGreater(hold_result.total_score, rollback_result.total_score)


if __name__ == "__main__":
    unittest.main()
