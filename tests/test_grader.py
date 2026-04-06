import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CanaryAction
from server.canary_environment import CanaryEnvironment
from server.grader import grade
from server.scenarios import SCENARIOS


def make_action(action_type: str, reasoning: str, breach: bool) -> CanaryAction:
    return CanaryAction(
        action_type=action_type,
        reasoning=reasoning,
        slo_breach_detected=breach,
    )


class GraderTests(unittest.TestCase):
    def test_grader_scores_are_normalized_between_zero_and_one(self) -> None:
        bad_action = make_action(
            "increase_10",
            "Increasing traffic during a breach to gather more data.",
            breach=False,
        )
        good_action = make_action(
            "increase_10",
            "Canary and stable are healthy with low error, low latency, and a safe 10% promotion step.",
            breach=False,
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
            )
        )
        observation = env.step(
            make_action(
                "increase_5",
                "Metrics remain healthy and a cautious increase is acceptable.",
                breach=False,
            )
        )

        self.assertFalse(observation.actual_breach)
        self.assertGreater(observation.canary_error_rate, 0.005)

        observation = env.step(
            make_action(
                "rollback",
                "The canary error rate exceeds the threshold, so rollback is required immediately.",
                breach=True,
            )
        )

        self.assertEqual(observation.policy_assessment, "correct_rollback")
        self.assertTrue(observation.actual_breach)
        self.assertGreaterEqual(observation.step_reward, 0.85)

    def test_hold_during_hard_task_global_noise_scores_better_than_promotion(self) -> None:
        hold = make_action(
            "hold",
            "Stable and canary latencies spiked together with no differential gap, so hold.",
            breach=False,
        )
        increase = make_action(
            "increase_10",
            "Latency is elevated but traffic should keep moving forward.",
            breach=False,
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
        )
        increase = make_action(
            "increase_10",
            "The canary has not breached yet, so continue the promotion.",
            breach=False,
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


if __name__ == "__main__":
    unittest.main()
