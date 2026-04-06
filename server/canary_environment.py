"""
Canary Release Manager - Environment

The environment grades each action against the observation the agent just saw,
then transitions to the next traffic state and metric snapshot.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from ._compat import bootstrap_import_paths

bootstrap_import_paths()

USE_PACKAGE_IMPORTS = (__package__ or "").startswith("canary_release_env.server")

if USE_PACKAGE_IMPORTS:
    from canary_release_env.models import CanaryAction, CanaryObservation
    from canary_release_env.server.grader import GradeResult, grade
    from canary_release_env.server.scenarios import MetricSnapshot, SCENARIOS, Scenario
else:
    from models import CanaryAction, CanaryObservation
    from server.grader import GradeResult, grade
    from server.scenarios import MetricSnapshot, SCENARIOS, Scenario

MAX_STEPS = 12
TRAFFIC_STEP_MAP = {
    "increase_5": 0.05,
    "increase_10": 0.10,
    "increase_25": 0.25,
    "hold": 0.00,
    "rollback": -1.00,
}


class CanaryEnvironment(Environment):
    def __init__(self) -> None:
        self._scenario: Scenario | None = None
        self._traffic_pct: float = 0.0
        self._step_number: int = 0
        self._consecutive_holds: int = 0
        self._score_total: float = 0.0
        self._done: bool = False
        self._last_grade: GradeResult | None = None
        self._transcript: list[dict[str, Any]] = []
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, task: str = "easy") -> CanaryObservation:
        requested_task = task
        defaulted_task = requested_task not in SCENARIOS
        if defaulted_task:
            task = "easy"

        self._scenario = SCENARIOS[task]
        self._traffic_pct = 0.0
        self._step_number = 0
        self._consecutive_holds = 0
        self._score_total = 0.0
        self._done = False
        self._last_grade = None
        self._transcript = []
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._build_observation(
            snapshot=self._current_snapshot(),
            reward=0.0,
            feedback=(
                (
                    f"Unknown task '{requested_task}' requested; defaulted to 'easy' so reset() could return a valid episode. "
                    if defaulted_task
                    else ""
                )
                + f"New deployment started. Canary is at 0% traffic. "
                + f"Task: '{task}' - {self._scenario.failure_pattern}. "
                + "Use the current observation to decide whether to promote, hold, or rollback."
            ),
            grade_result=None,
        )

    def step(self, action: CanaryAction) -> CanaryObservation:
        if self._scenario is None:
            return self._uninitialized_observation()

        if self._done:
            return self._build_observation(
                snapshot=self._current_snapshot(),
                reward=0.0,
                feedback="Episode already complete. Call reset() to start a new one.",
                grade_result=self._last_grade,
            )

        pre_snapshot = self._current_snapshot()
        pre_step_number = self._step_number
        pre_traffic_pct = self._traffic_pct
        pre_consecutive_holds = self._consecutive_holds

        grade_result = grade(
            action=action,
            traffic_pct=self._traffic_pct,
            canary_error=pre_snapshot.canary_error_rate,
            canary_p99=pre_snapshot.canary_p99_ms,
            stable_error=pre_snapshot.stable_error_rate,
            stable_p99=pre_snapshot.stable_p99_ms,
            alert_count=pre_snapshot.alert_count,
            scenario=self._scenario,
            step_number=pre_step_number,
            consecutive_holds=self._consecutive_holds,
        )

        if action.action_type == "rollback":
            self._traffic_pct = 0.0
            self._done = True
            self._consecutive_holds = 0
        else:
            delta = TRAFFIC_STEP_MAP[action.action_type]
            self._traffic_pct = min(1.0, self._traffic_pct + delta)
            if action.action_type == "hold":
                self._consecutive_holds += 1
            else:
                self._consecutive_holds = 0

        self._step_number += 1
        self._state.step_count = self._step_number
        self._score_total += grade_result.total_score
        self._last_grade = grade_result

        if self._traffic_pct >= 1.0:
            self._done = True
        if self._step_number >= MAX_STEPS:
            self._done = True

        feedback = self._build_feedback(action, grade_result)
        observation = self._build_observation(
            snapshot=self._current_snapshot(),
            reward=grade_result.total_score,
            feedback=feedback,
            grade_result=grade_result,
        )

        self._transcript.append(
            {
                "decision_number": self._step_number,
                "pre_observation": self._snapshot_to_dict(
                    pre_snapshot,
                    observed_step_number=pre_step_number,
                    traffic_pct=pre_traffic_pct,
                    consecutive_holds=pre_consecutive_holds,
                ),
                "action": action.model_dump(exclude={"metadata"}),
                "step_score": round(grade_result.total_score, 4),
                "actual_breach": grade_result.actual_breach,
                "policy_assessment": grade_result.policy_assessment,
                "reward_breakdown": grade_result.reward_breakdown,
                "explanation": grade_result.explanation,
                "post_observation": self._observation_summary(observation),
            }
        )

        return observation

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Canary Release Manager",
            description=(
                "OpenEnv environment for managing ML model canary releases with increase, hold, and rollback decisions."
            ),
            version="1.0.0",
            author="Canary Release Manager Team",
        )

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return list(self._transcript)

    def episode_result(self) -> dict[str, Any]:
        if self._scenario is None:
            return {
                "episode_id": self._state.episode_id,
                "task_id": "uninitialized",
                "steps": 0,
                "episode_score": 0.0,
                "outcome": "not_started",
                "score_breakdown": self._empty_score_breakdown(),
                "first_breach_point": None,
                "rollback_assessment": self._rollback_assessment(None, "not_started"),
                "episode_summary": "No episode is active yet.",
                "transcript": [],
            }

        if self._traffic_pct >= 1.0:
            outcome = "success"
        elif self._done and self._step_number >= MAX_STEPS:
            outcome = "max_steps_reached"
        elif self._done:
            outcome = "rollback"
        else:
            outcome = "in_progress"

        first_breach_point = self._first_breach_point()
        rollback_assessment = self._rollback_assessment(first_breach_point, outcome)
        score_breakdown = self._episode_score_breakdown()

        return {
            "episode_id": self._state.episode_id,
            "task_id": self._scenario.id,
            "steps": self._step_number,
            "episode_score": round(self._running_episode_score(), 4),
            "outcome": outcome,
            "score_breakdown": score_breakdown,
            "first_breach_point": first_breach_point,
            "rollback_assessment": rollback_assessment,
            "episode_summary": self._episode_summary(
                outcome=outcome,
                first_breach_point=first_breach_point,
                rollback_assessment=rollback_assessment,
                score_breakdown=score_breakdown,
            ),
            "transcript": self.transcript,
        }

    def _current_snapshot(self) -> MetricSnapshot:
        if self._scenario is None:
            return MetricSnapshot(0.001, 140.0, 0.001, 140.0, 0)
        return self._scenario.metric_fn(self._traffic_pct, self._step_number)

    def _running_episode_score(self) -> float:
        if self._step_number == 0:
            return 0.0
        return self._score_total / self._step_number

    def _build_observation(
        self,
        *,
        snapshot: MetricSnapshot,
        reward: float,
        feedback: str,
        grade_result: GradeResult | None,
    ) -> CanaryObservation:
        scenario = self._scenario
        task_id = scenario.id if scenario else "uninitialized"
        task_description = (
            scenario.agent_instructions
            if scenario
            else "Reset the environment before calling step()."
        )

        return CanaryObservation(
            reward=round(reward, 4),
            done=self._done,
            traffic_pct=round(self._traffic_pct, 4),
            canary_error_rate=round(snapshot.canary_error_rate, 6),
            canary_p99_ms=round(snapshot.canary_p99_ms, 2),
            stable_error_rate=round(snapshot.stable_error_rate, 6),
            stable_p99_ms=round(snapshot.stable_p99_ms, 2),
            differential_error=round(
                snapshot.canary_error_rate - snapshot.stable_error_rate, 6
            ),
            differential_p99_ms=round(
                snapshot.canary_p99_ms - snapshot.stable_p99_ms, 2
            ),
            alert_count=snapshot.alert_count,
            step_number=self._step_number,
            step_reward=round(reward, 4),
            cumulative_reward=round(self._running_episode_score(), 4),
            is_done=self._done,
            consecutive_holds=self._consecutive_holds,
            actual_breach=grade_result.actual_breach if grade_result else False,
            policy_assessment=grade_result.policy_assessment if grade_result else "",
            reward_breakdown=grade_result.reward_breakdown if grade_result else {},
            step_explanation=grade_result.explanation if grade_result else "",
            task_id=task_id,
            task_description=task_description,
            feedback=feedback,
        )

    def _uninitialized_observation(self) -> CanaryObservation:
        return CanaryObservation(
            reward=0.0,
            done=True,
            traffic_pct=0.0,
            canary_error_rate=0.001,
            canary_p99_ms=140.0,
            stable_error_rate=0.001,
            stable_p99_ms=140.0,
            differential_error=0.0,
            differential_p99_ms=0.0,
            alert_count=0,
            step_number=0,
            step_reward=0.0,
            cumulative_reward=0.0,
            is_done=True,
            consecutive_holds=0,
            actual_breach=False,
            policy_assessment="reset_required",
            reward_breakdown={},
            step_explanation="Call reset() before calling step().",
            task_id="uninitialized",
            task_description="Call reset() before step(), or use /episodes for stateful HTTP.",
            feedback="No active episode. Reset first so the action is graded against a real task state.",
        )

    def _build_feedback(self, action: CanaryAction, result: GradeResult) -> str:
        lines = [
            f"Action '{action.action_type}' assessed as '{result.policy_assessment}'.",
            result.summary(),
            result.explanation,
        ]
        lines.extend(result.feedback_parts)

        if self._done:
            lines.append(
                f"Episode complete. Final normalized episode score: {self._running_episode_score():.3f}."
            )
        else:
            lines.append(
                f"Traffic is now {self._traffic_pct:.0%}. Steps remaining: {MAX_STEPS - self._step_number}."
            )
        return "\n".join(lines)

    def _snapshot_to_dict(
        self,
        snapshot: MetricSnapshot,
        observed_step_number: int,
        traffic_pct: float,
        consecutive_holds: int,
    ) -> dict[str, Any]:
        return {
            "traffic_pct": round(traffic_pct, 4),
            "step_number": observed_step_number,
            "consecutive_holds": consecutive_holds,
            "canary_error_rate": round(snapshot.canary_error_rate, 6),
            "canary_p99_ms": round(snapshot.canary_p99_ms, 2),
            "stable_error_rate": round(snapshot.stable_error_rate, 6),
            "stable_p99_ms": round(snapshot.stable_p99_ms, 2),
            "differential_error": round(
                snapshot.canary_error_rate - snapshot.stable_error_rate, 6
            ),
            "differential_p99_ms": round(
                snapshot.canary_p99_ms - snapshot.stable_p99_ms, 2
            ),
            "alert_count": snapshot.alert_count,
        }

    def _observation_summary(self, observation: CanaryObservation) -> dict[str, Any]:
        return {
            "traffic_pct": observation.traffic_pct,
            "step_number": observation.step_number,
            "consecutive_holds": observation.consecutive_holds,
            "canary_error_rate": observation.canary_error_rate,
            "canary_p99_ms": observation.canary_p99_ms,
            "stable_error_rate": observation.stable_error_rate,
            "stable_p99_ms": observation.stable_p99_ms,
            "differential_error": observation.differential_error,
            "differential_p99_ms": observation.differential_p99_ms,
            "alert_count": observation.alert_count,
            "step_reward": observation.step_reward,
            "episode_score": observation.cumulative_reward,
            "is_done": observation.is_done,
        }

    def _empty_score_breakdown(self) -> dict[str, float]:
        return {
            "breach_detection_score": 0.0,
            "rollback_timing_score": 0.0,
            "promotion_safety_score": 0.0,
            "reasoning_score": 0.0,
            "total_score": 0.0,
        }

    def _episode_score_breakdown(self) -> dict[str, float]:
        if not self._transcript:
            return self._empty_score_breakdown()

        keys = [
            "breach_detection_score",
            "rollback_timing_score",
            "promotion_safety_score",
            "reasoning_score",
        ]
        breakdown = {
            key: round(
                sum(entry["reward_breakdown"].get(key, 0.0) for entry in self._transcript)
                / len(self._transcript),
                4,
            )
            for key in keys
        }
        breakdown["total_score"] = round(self._running_episode_score(), 4)
        return breakdown

    def _first_breach_point(self) -> dict[str, Any] | None:
        for entry in self._transcript:
            if entry["actual_breach"]:
                pre = entry["pre_observation"]
                return {
                    "decision_number": entry["decision_number"],
                    "step_number": pre["step_number"],
                    "traffic_pct": pre["traffic_pct"],
                    "canary_error_rate": pre["canary_error_rate"],
                    "differential_p99_ms": pre["differential_p99_ms"],
                    "alert_count": pre["alert_count"],
                }
        return None

    def _rollback_assessment(
        self,
        first_breach_point: dict[str, Any] | None,
        outcome: str,
    ) -> dict[str, Any]:
        rollback_entry = next(
            (
                entry
                for entry in self._transcript
                if entry["action"]["action_type"] == "rollback"
            ),
            None,
        )

        threshold = self._scenario.correct_rollback_at if self._scenario else None
        rollback_traffic = (
            rollback_entry["pre_observation"]["traffic_pct"]
            if rollback_entry is not None
            else None
        )

        if rollback_entry is None:
            if outcome in {"success", "in_progress", "not_started"}:
                return {
                    "rollback_action_taken": False,
                    "relative_to_first_breach": "not_applicable",
                    "relative_to_benchmark_point": "not_applicable",
                    "rollback_traffic_pct": None,
                    "benchmark_rollback_at": threshold,
                }
            return {
                "rollback_action_taken": False,
                "relative_to_first_breach": "missed",
                "relative_to_benchmark_point": "missed",
                "rollback_traffic_pct": None,
                "benchmark_rollback_at": threshold,
            }

        if first_breach_point is None:
            breach_timing = "before_first_breach"
        elif rollback_entry["decision_number"] == first_breach_point["decision_number"]:
            breach_timing = "on_first_breach"
        else:
            breach_timing = "after_first_breach"

        if threshold is None or rollback_traffic is None:
            benchmark_timing = "not_applicable"
        elif rollback_traffic < threshold:
            benchmark_timing = "early"
        elif abs(rollback_traffic - threshold) < 1e-9:
            benchmark_timing = "timely"
        else:
            benchmark_timing = "late"

        return {
            "rollback_action_taken": True,
            "relative_to_first_breach": breach_timing,
            "relative_to_benchmark_point": benchmark_timing,
            "rollback_traffic_pct": rollback_traffic,
            "benchmark_rollback_at": threshold,
        }

    def _episode_summary(
        self,
        *,
        outcome: str,
        first_breach_point: dict[str, Any] | None,
        rollback_assessment: dict[str, Any],
        score_breakdown: dict[str, float],
    ) -> str:
        noise_seen = any("noise" in entry["policy_assessment"] for entry in self._transcript)
        warning_seen = any("warning" in entry["policy_assessment"] for entry in self._transcript)
        breach_timing = rollback_assessment["relative_to_first_breach"]
        benchmark_timing = rollback_assessment["relative_to_benchmark_point"]
        benchmark_rollback_at = rollback_assessment["benchmark_rollback_at"]

        if outcome == "success":
            return (
                f"Reached 100% traffic without a confirmed canary breach. "
                f"Final normalized score={score_breakdown['total_score']:.4f}."
            )

        if outcome == "in_progress":
            return (
                f"Episode is still running. Current normalized score={score_breakdown['total_score']:.4f}."
            )

        if breach_timing == "before_first_breach":
            summary = (
                f"Rolled back before a confirmed breach was observed. "
                f"Final normalized score={score_breakdown['total_score']:.4f}."
            )
        elif first_breach_point is not None:
            benchmark_clause = ""
            if benchmark_rollback_at is not None:
                if benchmark_timing == "early":
                    benchmark_clause = (
                        f" relative to the benchmark rollback point at {benchmark_rollback_at:.0%} traffic"
                    )
                elif benchmark_timing == "timely":
                    benchmark_clause = (
                        f" at the benchmark rollback point of {benchmark_rollback_at:.0%} traffic"
                    )
                elif benchmark_timing == "late":
                    benchmark_clause = (
                        f" after the benchmark rollback point of {benchmark_rollback_at:.0%} traffic"
                    )

            if breach_timing == "on_first_breach":
                timing_phrase = "Rolled back on the first confirmed breach"
            else:
                timing_phrase = "Rolled back after the first confirmed breach"

            summary = (
                f"{timing_phrase}{benchmark_clause}; "
                f"the first confirmed breach was observed at {first_breach_point['traffic_pct']:.0%} traffic. "
                f"Final normalized score={score_breakdown['total_score']:.4f}."
            )
        else:
            summary = (
                f"The rollout ended without a successful rollback decision after the breach. "
                f"Final normalized score={score_breakdown['total_score']:.4f}."
            )

        if noise_seen:
            summary += " Shared infrastructure noise appeared during the episode."
        if warning_seen:
            summary += " The transcript includes an explicit warning window before the final breach."
        return summary
