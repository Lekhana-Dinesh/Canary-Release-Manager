from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent
    parent = repo_root.parent
    for path in (repo_root, parent):
        candidate = str(path)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
        from models import CanaryAction, CanaryObservation
        from server.canary_environment import CanaryEnvironment
        from server.policies import (
            baseline_action,
            observation_has_actual_breach,
            observation_is_phantom_alert,
            observation_is_shared_noise,
            observation_is_warning_window,
            safe_fallback_action,
            state_assessment_for_observation,
        )
        from server.scenarios import PUBLIC_TASK_IDS
    except ModuleNotFoundError:
        from canary_release_env.models import CanaryAction, CanaryObservation
        from canary_release_env.server.canary_environment import CanaryEnvironment
        from canary_release_env.server.policies import (
            baseline_action,
            observation_has_actual_breach,
            observation_is_phantom_alert,
            observation_is_shared_noise,
            observation_is_warning_window,
            safe_fallback_action,
            state_assessment_for_observation,
        )
        from canary_release_env.server.scenarios import PUBLIC_TASK_IDS
else:
    from canary_release_env.models import CanaryAction, CanaryObservation
    from canary_release_env.server.canary_environment import CanaryEnvironment
    from canary_release_env.server.policies import (
        baseline_action,
        observation_has_actual_breach,
        observation_is_phantom_alert,
        observation_is_shared_noise,
        observation_is_warning_window,
        safe_fallback_action,
        state_assessment_for_observation,
    )
    from canary_release_env.server.scenarios import PUBLIC_TASK_IDS

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "review_artifacts"
STRICT_INFERENCE_TAGS = ("[START]", "[STEP]", "[END]")
ARTIFACT_FILENAMES = {
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
    "benchmark": "policy_benchmark_results.json",
    "seed_sweep": "policy_seed_sweep_results.json",
    "variant_catalog": "scenario_variant_catalog.json",
    "validation": "openenv_validation_results.json",
    "endpoint_sample": "endpoint_contract_sample.json",
    "manifest": "artifact_manifest.json",
    "index": "artifact_index.md",
    "audit": "benchmark_audit_summary.md",
}
LEGACY_ARTIFACT_FILENAMES = {
    "inference_stdout_sample.txt",
    "good_hard_transcript.json",
    "good_hard_grader.json",
    "bad_hard_transcript.json",
    "bad_hard_grader.json",
    "ambiguous_hard_transcript.json",
    "ambiguous_hard_grader.json",
    "sample_endpoint_responses.json",
    "final_audit_summary.md",
    "transcript_summary.md",
    "local_server.log",
}
MANAGED_ARTIFACT_FILENAMES = set(ARTIFACT_FILENAMES.values()) | LEGACY_ARTIFACT_FILENAMES
BENCHMARK_SEEDS = (0, 3, 5, 7, 11, 13, 19, 23)


def make_action(
    action_type: str,
    reasoning: str,
    breach: bool,
    state_assessment: str,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "reasoning": reasoning,
        "slo_breach_detected": breach,
        "state_assessment": state_assessment,
    }


GOOD_HARD_ACTIONS = [
    make_action(
        "increase_10",
        "Metrics are healthy with low error and no differential latency, so take the standard 10% promotion step.",
        False,
        "healthy",
    ),
    make_action(
        "increase_10",
        "Stable and canary remain aligned with healthy metrics, so continue the rollout by another 10%.",
        False,
        "healthy",
    ),
    make_action(
        "increase_10",
        "The canary still matches stable performance, so increase traffic by 10% again.",
        False,
        "healthy",
    ),
    make_action(
        "hold",
        "Stable and canary latencies spiked together with no differential gap, so hold for confirmation.",
        False,
        "noise",
    ),
    make_action(
        "hold",
        "The canary remains slower than stable after the shared-noise event, so hold through the watch window.",
        False,
        "warning",
    ),
    make_action(
        "rollback",
        "The canary now breaches the differential guardrail, so rollback immediately.",
        True,
        "breach",
    ),
]

BAD_HARD_ACTIONS = [
    *GOOD_HARD_ACTIONS[:4],
    make_action(
        "increase_10",
        "The canary has not breached yet and more traffic should make the signal clearer, so increase by 10%.",
        False,
        "warning",
    ),
    make_action(
        "rollback",
        "The canary is now breaching and should be rolled back.",
        True,
        "breach",
    ),
]

AMBIGUOUS_HARD_ACTIONS = GOOD_HARD_ACTIONS[:4]
AMBIGUOUS_WATCH_ACTION = make_action(
    "hold",
    "The canary is still slower than stable after the shared-noise event, so hold for one more confirmation step.",
    False,
    "warning",
)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "episode_id":
                cleaned[key] = "artifact-generated"
            else:
                cleaned[key] = _sanitize(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_sanitize(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _cleanup_managed_artifacts(output_dir: Path) -> None:
    for filename in MANAGED_ARTIFACT_FILENAMES:
        artifact_path = output_dir / filename
        if artifact_path.exists():
            artifact_path.unlink()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with {exc.code}: {details}") from exc


def _start_local_server(output_dir: Path) -> tuple[subprocess.Popen[str], Any, str]:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    tmp_dir = ROOT / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (tmp_dir / "review_artifacts_local_server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, log_handle, base_url


def _wait_for_health(base_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            payload = _http_json(base_url, "/health")
            if payload.get("status") == "healthy":
                return
        except Exception as exc:  # pragma: no cover - exercised in integration flow
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {base_url}/health. Last error: {last_error}")


def _stop_local_server(process: subprocess.Popen[str], log_handle: Any) -> None:
    try:
        process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    finally:
        log_handle.close()


def _run_episode_via_http(
    base_url: str,
    *,
    task_id: str,
    actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    created = _http_json(base_url, "/episodes", method="POST", payload={"task": task_id})
    episode_id = created["episode_id"]
    current_observation = created["observation"]
    final_response: dict[str, Any] | None = None

    for action in actions:
        final_response = _http_json(
            base_url,
            f"/episodes/{episode_id}/step",
            method="POST",
            payload={"action": action},
        )
        current_observation = final_response["observation"]
        if final_response["done"]:
            return final_response["episode_result"], current_observation

    return _http_json(base_url, f"/episodes/{episode_id}/transcript"), current_observation


def _run_policy_episode_via_http(
    base_url: str,
    *,
    task_id: str,
    policy_fn: Callable[[Any], CanaryAction],
    stop_after_steps: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    created = _http_json(base_url, "/episodes", method="POST", payload={"task": task_id})
    episode_id = created["episode_id"]
    observation = CanaryObservation.model_validate(created["observation"])

    while not observation.done:
        action = policy_fn(observation)
        response = _http_json(
            base_url,
            f"/episodes/{episode_id}/step",
            method="POST",
            payload={"action": action.model_dump(exclude={"metadata"})},
        )
        observation = CanaryObservation.model_validate(response["observation"])

        if response["done"]:
            return response["episode_result"], observation.model_dump()
        if stop_after_steps is not None and observation.step_number >= stop_after_steps:
            return _http_json(base_url, f"/episodes/{episode_id}/transcript"), observation.model_dump()

    return _http_json(base_url, f"/episodes/{episode_id}/transcript"), observation.model_dump()


def _build_grader_request(
    *,
    task_id: str,
    observation: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "action": action,
        "traffic_pct": observation["traffic_pct"],
        "canary_error_rate": observation["canary_error_rate"],
        "canary_p99_ms": observation["canary_p99_ms"],
        "stable_error_rate": observation["stable_error_rate"],
        "stable_p99_ms": observation["stable_p99_ms"],
        "alert_count": observation["alert_count"],
        "step_number": observation["step_number"],
        "consecutive_holds": observation.get("consecutive_holds", 0),
    }


def _run_inference_sample(base_url: str, output_dir: Path) -> None:
    env = os.environ.copy()
    env["API_BASE_URL"] = ""
    env["MODEL_NAME"] = ""
    env["HF_TOKEN"] = ""
    env["API_KEY"] = ""

    completed = subprocess.run(
        [sys.executable, "inference.py", "--env-url", base_url],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"inference.py failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    if completed.stderr.strip():
        raise RuntimeError(f"inference.py wrote to stderr: {completed.stderr.strip()}")

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("inference.py produced no stdout lines.")
    for line in lines:
        if not any(line.startswith(tag) for tag in STRICT_INFERENCE_TAGS):
            raise RuntimeError(f"inference.py emitted a non-contract line: {line}")

    _write_text(output_dir / ARTIFACT_FILENAMES["inference_stdout"], completed.stdout)


def _run_openenv_validate(*args: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "openenv.cli", "validate", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    payload: dict[str, Any] = {
        "command": " ".join(["python", "-m", "openenv.cli", "validate", *args]),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "passed": completed.returncode == 0,
    }
    if stdout.startswith("{"):
        try:
            payload["parsed_stdout"] = json.loads(stdout)
        except json.JSONDecodeError:
            payload["parsed_stdout"] = None
    return payload


def _find_transcript_entry(
    episode: dict[str, Any],
    *,
    preferred_assessments: tuple[str, ...],
) -> dict[str, Any]:
    for entry in episode["transcript"]:
        if entry["policy_assessment"] in preferred_assessments:
            return entry
    return episode["transcript"][-1]


def _probe_rollout_trace(task_id: str, seed: int, max_steps: int = 10) -> list[CanaryObservation]:
    env = CanaryEnvironment()
    observation = env.reset(task=task_id, seed=seed)
    trace = [observation]

    while not observation.done and observation.step_number < max_steps:
        observation = env.step(
            CanaryAction(
                action_type="increase_10",
                reasoning=(
                    "Probe rollout used only to surface deterministic event timing and signal shapes."
                ),
                slo_breach_detected=False,
                state_assessment=state_assessment_for_observation(observation),
            )
        )
        trace.append(observation)

    return trace


def _first_matching_observation(
    trace: list[CanaryObservation],
    predicate: Callable[[CanaryObservation], bool],
) -> CanaryObservation | None:
    for observation in trace[1:]:
        if predicate(observation):
            return observation
    return None


def _observation_signature(observation: CanaryObservation | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        "step_number": int(observation.step_number),
        "traffic_pct": round(float(observation.traffic_pct), 4),
        "canary_error_rate": round(float(observation.canary_error_rate), 6),
        "differential_error": round(float(observation.differential_error), 6),
        "differential_p99_ms": round(float(observation.differential_p99_ms), 2),
        "alert_count": int(observation.alert_count),
        "public_state_assessment": state_assessment_for_observation(observation),
    }


def _variant_catalog_entry(task_id: str, seed: int) -> dict[str, Any]:
    trace = _probe_rollout_trace(task_id, seed)
    first_warning = _first_matching_observation(trace, observation_is_warning_window)
    first_noise = _first_matching_observation(trace, observation_is_shared_noise)
    first_phantom = _first_matching_observation(trace, observation_is_phantom_alert)
    first_breach = _first_matching_observation(trace, observation_has_actual_breach)

    recovery_clear = None
    if first_warning is not None:
        for observation in trace:
            if observation.step_number <= first_warning.step_number:
                continue
            if (
                state_assessment_for_observation(observation) == "healthy"
                and not observation_has_actual_breach(observation)
            ):
                recovery_clear = observation
                break

    return {
        "trace_steps_captured": len(trace) - 1,
        "first_warning": _observation_signature(first_warning),
        "first_shared_noise": _observation_signature(first_noise),
        "first_phantom_alert": _observation_signature(first_phantom),
        "first_breach": _observation_signature(first_breach),
        "recovery_clear": _observation_signature(recovery_clear),
        "final_probe_state": _observation_signature(trace[-1]),
    }


def scenario_variant_catalog(seeds: tuple[int, ...] = BENCHMARK_SEEDS) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for task_id in PUBLIC_TASK_IDS:
        per_seed = {
            str(seed): _variant_catalog_entry(task_id, seed)
            for seed in seeds
        }
        tasks[task_id] = per_seed

        def _distinct_steps(field_name: str) -> list[int]:
            return sorted(
                {
                    entry[field_name]["step_number"]
                    for entry in per_seed.values()
                    if entry[field_name] is not None
                }
            )

        summaries[task_id] = {
            "distinct_warning_steps": _distinct_steps("first_warning"),
            "distinct_noise_steps": _distinct_steps("first_shared_noise"),
            "distinct_phantom_steps": _distinct_steps("first_phantom_alert"),
            "distinct_breach_steps": _distinct_steps("first_breach"),
            "distinct_recovery_clear_steps": _distinct_steps("recovery_clear"),
        }

    return {
        "probe_policy": (
            "Deterministic fixed 10% promotions used only to surface event ordering and signal shape diversity; this is not a benchmark policy."
        ),
        "seeds": list(seeds),
        "tasks": tasks,
        "task_summaries": summaries,
    }


def _build_artifact_manifest(
    *,
    benchmark_results: dict[str, Any],
    seed_sweep_results: dict[str, Any],
    validation_results: dict[str, Any],
    cautious_episode: dict[str, Any],
    cautious_grader: dict[str, Any],
    aggressive_episode: dict[str, Any],
    aggressive_grader: dict[str, Any],
    watch_window_episode: dict[str, Any],
    watch_window_grader: dict[str, Any],
    expert_episode: dict[str, Any],
    expert_grader: dict[str, Any],
    recovery_episode: dict[str, Any],
    recovery_grader: dict[str, Any],
) -> dict[str, Any]:
    cautious_hard_score = benchmark_results["policies"]["cautious_policy"]["scores"]["hard"]
    return {
        "generated_by": "python generate_review_artifacts.py",
        "score_semantics": {
            "episode_score": "Normalized running average across all decisions taken so far in an episode.",
            "grader_total_score": "Normalized score for exactly one decision under POST /grader.",
            "policy_benchmark_score": "Final episode_score for a named policy on a named task.",
            "partial_episode_note": (
                "The watch-window transcript is intentionally in_progress. Its episode_score is a running average through that point, "
                "not a completed-policy benchmark average."
            ),
        },
        "artifacts": [
            {
                "filename": ARTIFACT_FILENAMES["index"],
                "artifact_kind": "reviewer_guide",
                "policy": None,
                "task_id": None,
                "run_scope": "documentation",
                "why_it_exists": "Human-readable starting point for reviewing the artifact pack.",
            },
            {
                "filename": ARTIFACT_FILENAMES["manifest"],
                "artifact_kind": "manifest",
                "policy": None,
                "task_id": None,
                "run_scope": "documentation",
                "why_it_exists": "Machine-readable index describing every artifact and its score semantics.",
            },
            {
                "filename": ARTIFACT_FILENAMES["inference_stdout"],
                "artifact_kind": "stdout_sample",
                "policy": "fallback_inference_policy",
                "task_id": None,
                "run_scope": "all_tasks",
                "why_it_exists": "Shows the strict inference stdout contract using only [START], [STEP], and [END].",
            },
            {
                "filename": ARTIFACT_FILENAMES["cautious_transcript"],
                "artifact_kind": "transcript",
                "policy": "cautious_policy",
                "task_id": "hard",
                "run_scope": "full_episode",
                "paired_with": ARTIFACT_FILENAMES["cautious_grader"],
                "why_it_exists": "Full hard-task run from the cautious policy used in the benchmark comparison.",
                "score_observed": cautious_episode["episode_score"],
                "score_note": (
                    f"This full-episode score should match the cautious_policy hard benchmark score ({cautious_hard_score:.4f})."
                ),
            },
            {
                "filename": ARTIFACT_FILENAMES["cautious_grader"],
                "artifact_kind": "grader_payload",
                "policy": "cautious_policy",
                "task_id": "hard",
                "run_scope": "single_step",
                "paired_with": ARTIFACT_FILENAMES["cautious_transcript"],
                "why_it_exists": "Representative final rollback decision from the cautious hard-task run.",
                "score_observed": cautious_grader["total_score"],
                "score_note": "This is a single-step grader score, not an episode score.",
            },
            {
                "filename": ARTIFACT_FILENAMES["aggressive_transcript"],
                "artifact_kind": "transcript",
                "policy": "aggressive_rollout_example",
                "task_id": "hard",
                "run_scope": "full_episode",
                "paired_with": ARTIFACT_FILENAMES["aggressive_grader"],
                "why_it_exists": "Full hard-task aggressive rollout example that promotes through the hard watch window.",
                "score_observed": aggressive_episode["episode_score"],
                "score_note": (
                    "This is a hard-specific illustrative rollout example. It is not the same object as the multi-task "
                    "`aggressive_policy` comparator in the benchmark results."
                ),
            },
            {
                "filename": ARTIFACT_FILENAMES["aggressive_grader"],
                "artifact_kind": "grader_payload",
                "policy": "aggressive_rollout_example",
                "task_id": "hard",
                "run_scope": "single_step",
                "paired_with": ARTIFACT_FILENAMES["aggressive_transcript"],
                "why_it_exists": "Captures the aggressive rollout example's signature mistake: promotion during the hard watch window.",
                "score_observed": aggressive_grader["total_score"],
                "score_note": "This is the risky watch-window promotion step, not the full-episode score.",
            },
            {
                "filename": ARTIFACT_FILENAMES["watch_window_transcript"],
                "artifact_kind": "transcript",
                "policy": "cautious_policy",
                "task_id": "hard",
                "run_scope": "partial_episode",
                "paired_with": ARTIFACT_FILENAMES["watch_window_grader"],
                "why_it_exists": "Shows the hard-task watch window after the hold decision has been taken.",
                "score_observed": watch_window_episode["episode_score"],
                "score_note": "This is an in-progress episode score through the watch-window hold, not a completed benchmark score.",
            },
            {
                "filename": ARTIFACT_FILENAMES["watch_window_grader"],
                "artifact_kind": "grader_payload",
                "policy": "cautious_policy",
                "task_id": "hard",
                "run_scope": "single_step",
                "paired_with": ARTIFACT_FILENAMES["watch_window_transcript"],
                "why_it_exists": "Isolates the watch-window hold decision that makes the hard task discriminative.",
                "score_observed": watch_window_grader["total_score"],
                "score_note": "This is a single-step grader score for the hold decision on the hard watch window.",
            },
            {
                "filename": ARTIFACT_FILENAMES["expert_transcript"],
                "artifact_kind": "transcript",
                "policy": "cautious_policy",
                "task_id": "expert",
                "run_scope": "full_episode",
                "paired_with": ARTIFACT_FILENAMES["expert_grader"],
                "why_it_exists": "Full expert-task run showing that the cautious policy ignores the phantom alert and later rolls back on the real differential breach.",
                "score_observed": expert_episode["episode_score"],
                "score_note": "This is a completed expert-task episode from the cautious policy.",
            },
            {
                "filename": ARTIFACT_FILENAMES["expert_grader"],
                "artifact_kind": "grader_payload",
                "policy": "cautious_policy",
                "task_id": "expert",
                "run_scope": "single_step",
                "paired_with": ARTIFACT_FILENAMES["expert_transcript"],
                "why_it_exists": "Single-step proof that the phantom alert is rewarded when the agent verifies metrics instead of rolling back.",
                "score_observed": expert_grader["total_score"],
                "score_note": "This is the phantom-alert hold decision score, not a full-episode score.",
            },
            {
                "filename": ARTIFACT_FILENAMES["recovery_transcript"],
                "artifact_kind": "transcript",
                "policy": "cautious_policy",
                "task_id": "recovery",
                "run_scope": "full_episode",
                "paired_with": ARTIFACT_FILENAMES["recovery_grader"],
                "why_it_exists": "Full recovery-task run showing that the cautious policy holds through a transient canary-only degradation and then continues the rollout safely.",
                "score_observed": recovery_episode["episode_score"],
                "score_note": "This is a completed recovery-task episode from the cautious policy.",
            },
            {
                "filename": ARTIFACT_FILENAMES["recovery_grader"],
                "artifact_kind": "grader_payload",
                "policy": "cautious_policy",
                "task_id": "recovery",
                "run_scope": "single_step",
                "paired_with": ARTIFACT_FILENAMES["recovery_transcript"],
                "why_it_exists": "Single-step proof that a transient canary-only degradation is rewarded as a hold, not a rollback.",
                "score_observed": recovery_grader["total_score"],
                "score_note": "This is the transient-recovery hold decision score, not a full-episode score.",
            },
            {
                "filename": ARTIFACT_FILENAMES["benchmark"],
                "artifact_kind": "benchmark_comparison",
                "policy": "multiple",
                "task_id": "easy,medium,hard,expert,recovery,silent",
                "run_scope": "full_episode",
                "why_it_exists": "Compares shallow baseline, cautious policy, and aggressive policy across all tasks.",
            },
            {
                "filename": ARTIFACT_FILENAMES["seed_sweep"],
                "artifact_kind": "benchmark_comparison",
                "policy": "multiple",
                "task_id": "easy,medium,hard,expert,recovery,silent",
                "run_scope": "multi_seed",
                "why_it_exists": (
                    "Shows that the main policy ordering remains coherent across deterministic non-zero seeds, "
                    "not just the canonical seed=0 profiles."
                ),
                "score_observed": {
                    "aggregate_hard_ordering": seed_sweep_results["aggregate_hard_ordering"],
                    "aggregate_expert_ordering": seed_sweep_results["aggregate_expert_ordering"],
                    "aggregate_recovery_ordering": seed_sweep_results["aggregate_recovery_ordering"],
                    "aggregate_silent_ordering": seed_sweep_results["aggregate_silent_ordering"],
                },
                "score_note": "Aggregate scores are the mean completed-episode scores across the documented benchmark seeds.",
            },
            {
                "filename": ARTIFACT_FILENAMES["variant_catalog"],
                "artifact_kind": "variant_catalog",
                "policy": "deterministic_probe",
                "task_id": "easy,medium,hard,expert,recovery,silent",
                "run_scope": "multi_seed",
                "why_it_exists": "Enumerates event ordering and signal-shape differences across deterministic seeds so reviewers can inspect task-family breadth directly.",
            },
            {
                "filename": ARTIFACT_FILENAMES["validation"],
                "artifact_kind": "validation_capture",
                "policy": None,
                "task_id": None,
                "run_scope": "structural_and_live_validation",
                "why_it_exists": "Captures structural OpenEnv validation and live local-server validation from the generated artifact run.",
                "score_observed": {
                    "structural_passed": validation_results["structural"]["passed"],
                    "live_passed": validation_results["live"]["passed"],
                },
                "score_note": "These are validator compatibility proofs, not benchmark scores.",
            },
            {
                "filename": ARTIFACT_FILENAMES["endpoint_sample"],
                "artifact_kind": "endpoint_capture",
                "policy": None,
                "task_id": None,
                "run_scope": "contract_sample",
                "why_it_exists": "Shows representative validator-safe endpoints plus the recommended /episodes HTTP flow.",
            },
            {
                "filename": ARTIFACT_FILENAMES["audit"],
                "artifact_kind": "summary",
                "policy": None,
                "task_id": None,
                "run_scope": "documentation",
                "why_it_exists": "Short benchmark-level summary of what the artifact pack proves.",
            },
        ],
    }


def _artifact_index(markdown_manifest: dict[str, Any]) -> str:
    lines = [
        "# Artifact Index",
        "",
        "Start here if you want to inspect the benchmark quickly.",
        "",
        "## Quick Read Order",
        "",
        f"- `{ARTIFACT_FILENAMES['index']}`: this reviewer guide.",
        f"- `{ARTIFACT_FILENAMES['benchmark']}`: final seed=0 cross-policy comparison across easy, medium, hard, expert, recovery, and silent.",
        f"- `{ARTIFACT_FILENAMES['seed_sweep']}`: deterministic multi-seed comparison that proves the ordering is not tied to one authored trace.",
        f"- `{ARTIFACT_FILENAMES['variant_catalog']}`: machine-readable event catalog showing how seed variants change warning, noise, phantom, recovery, and breach timing.",
        f"- `{ARTIFACT_FILENAMES['validation']}`: structural and live OpenEnv validation results captured from this tree.",
        f"- `{ARTIFACT_FILENAMES['cautious_transcript']}`: full hard-task run from the cautious policy.",
        f"- `{ARTIFACT_FILENAMES['expert_transcript']}`: full expert-task run showing phantom-alert handling.",
        f"- `{ARTIFACT_FILENAMES['recovery_transcript']}`: full recovery-task run showing transient degradation handling.",
        f"- `{ARTIFACT_FILENAMES['aggressive_grader']}`: single-step proof of the aggressive policy's watch-window mistake.",
        f"- `{ARTIFACT_FILENAMES['expert_grader']}`: single-step proof that the expert phantom alert is not a rollback.",
        f"- `{ARTIFACT_FILENAMES['recovery_grader']}`: single-step proof that the recovery warning window rewards holding.",
        f"- `{ARTIFACT_FILENAMES['inference_stdout']}`: strict inference stdout contract sample.",
        "",
        "## Score Interpretation",
        "",
        "- `episode_score` means normalized running-average score across decisions taken so far.",
        "- `total_score` in a grader artifact means a single decision score from `POST /grader`.",
        "- `policy_benchmark_results.json` reports completed-episode scores for named policies across all public tasks.",
        "- `policy_seed_sweep_results.json` reports the same policy comparison across multiple deterministic seeds and then averages them.",
        "- `openenv_validation_results.json` records validator outcomes and should not be interpreted as a benchmark score artifact.",
        "- The watch-window transcript is intentionally partial and `in_progress`, so its `episode_score` should not be compared directly with full benchmark averages.",
        "",
        "## Files",
        "",
    ]
    for entry in markdown_manifest["artifacts"]:
        policy = entry["policy"] or "n/a"
        task_id = entry["task_id"] or "n/a"
        lines.append(
            f"- `{entry['filename']}`: kind={entry['artifact_kind']}, policy={policy}, task={task_id}. {entry['why_it_exists']}"
        )
        if entry.get("score_note"):
            lines.append(f"  Score note: {entry['score_note']}")
    lines.append("")
    return "\n".join(lines)


def _aggressive_policy_action(observation) -> CanaryAction:
    state_assessment = state_assessment_for_observation(observation)
    if observation_has_actual_breach(observation):
        return CanaryAction(
            action_type="rollback",
            reasoning=(
                f"The canary is already breaching with error={observation.canary_error_rate:.4f} "
                f"and differential_p99={observation.differential_p99_ms:.1f}ms, so rollback."
            ),
            slo_breach_detected=True,
            state_assessment=state_assessment,
        )

    action_type = "increase_25" if observation.traffic_pct < 0.50 else "increase_10"
    return CanaryAction(
        action_type=action_type,
        reasoning=(
            f"Traffic is only {observation.traffic_pct:.0%} and the canary has not breached yet, "
            f"so use an aggressive promotion step with differential_p99={observation.differential_p99_ms:.1f}ms."
        ),
        slo_breach_detected=False,
        state_assessment=state_assessment,
    )


def _run_policy_episode(
    task_id: str,
    policy_name: str,
    policy_fn: Callable[[Any], CanaryAction],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    env = CanaryEnvironment()
    observation = env.reset(task=task_id, seed=seed)
    actions: list[dict[str, Any]] = []

    while not observation.done:
        action = policy_fn(observation)
        actions.append(
            {
                "step_number": observation.step_number,
                "traffic_pct": observation.traffic_pct,
                "consecutive_holds": observation.consecutive_holds,
                "action_type": action.action_type,
                "reasoning": action.reasoning,
                "slo_breach_detected": action.slo_breach_detected,
                "state_assessment": action.state_assessment,
            }
        )
        observation = env.step(action)

    episode_result = env.episode_result()
    return {
        "policy": policy_name,
        "task_id": task_id,
        "seed": seed,
        "score": episode_result["episode_score"],
        "steps": episode_result["steps"],
        "outcome": episode_result["outcome"],
        "score_breakdown": episode_result["score_breakdown"],
        "rollback_assessment": episode_result["rollback_assessment"],
        "episode_summary": episode_result["episode_summary"],
        "actions": actions,
    }


def benchmark_policies() -> dict[str, Any]:
    policies: dict[str, tuple[str, Callable[[Any], CanaryAction]]] = {
        "shallow_baseline": (
            "Deterministic benchmark baseline that ignores warning windows.",
            baseline_action,
        ),
        "cautious_policy": (
            "Safer observation-aware policy used as the inference fallback.",
            safe_fallback_action,
        ),
        "aggressive_policy": (
            "Intentionally overconfident policy that promotes hard and medium cases too aggressively.",
            _aggressive_policy_action,
        ),
    }

    tasks = PUBLIC_TASK_IDS
    policy_results: dict[str, Any] = {}
    for policy_name, (description, policy_fn) in policies.items():
        per_task = {
            task_id: _run_policy_episode(task_id, policy_name, policy_fn, seed=0)
            for task_id in tasks
        }
        scores = {task_id: per_task[task_id]["score"] for task_id in tasks}
        average = round(sum(scores.values()) / len(scores), 4)
        policy_results[policy_name] = {
            "description": description,
            "scores": scores,
            "average": average,
            "details": per_task,
        }

    return {
        "score_contract": {
            "per_step_score_range": [0.0, 1.0],
            "episode_score_range": [0.0, 1.0],
            "episode_score_definition": "running average of normalized step scores",
        },
        "policies": policy_results,
        "hard_task_ordering": sorted(
            policy_results,
            key=lambda name: policy_results[name]["scores"]["hard"],
            reverse=True,
        ),
        "recovery_task_ordering": sorted(
            policy_results,
            key=lambda name: policy_results[name]["scores"]["recovery"],
            reverse=True,
        ),
        "silent_task_ordering": sorted(
            policy_results,
            key=lambda name: policy_results[name]["scores"]["silent"],
            reverse=True,
        ),
    }


def benchmark_seed_sweep(seeds: tuple[int, ...] = BENCHMARK_SEEDS) -> dict[str, Any]:
    policies: dict[str, tuple[str, Callable[[Any], CanaryAction]]] = {
        "shallow_baseline": (
            "Deterministic benchmark baseline that ignores warning windows.",
            baseline_action,
        ),
        "cautious_policy": (
            "Safer observation-aware policy used as the inference fallback.",
            safe_fallback_action,
        ),
        "aggressive_policy": (
            "Intentionally overconfident policy that promotes hard and medium cases too aggressively.",
            _aggressive_policy_action,
        ),
    }

    policy_results: dict[str, Any] = {}
    for policy_name, (description, policy_fn) in policies.items():
        per_seed: dict[str, Any] = {}
        aggregate_scores: dict[str, list[float]] = {task_id: [] for task_id in PUBLIC_TASK_IDS}

        for seed in seeds:
            per_task = {
                task_id: _run_policy_episode(task_id, policy_name, policy_fn, seed=seed)
                for task_id in PUBLIC_TASK_IDS
            }
            scores = {task_id: per_task[task_id]["score"] for task_id in PUBLIC_TASK_IDS}
            for task_id, score in scores.items():
                aggregate_scores[task_id].append(score)
            per_seed[str(seed)] = {
                "scores": scores,
                "average": round(sum(scores.values()) / len(scores), 4),
                "details": per_task,
            }

        averaged_scores = {
            task_id: round(sum(scores) / len(scores), 4)
            for task_id, scores in aggregate_scores.items()
        }
        policy_results[policy_name] = {
            "description": description,
            "per_seed": per_seed,
            "aggregate_scores": averaged_scores,
            "aggregate_average": round(
                sum(averaged_scores.values()) / len(averaged_scores),
                4,
            ),
        }

    return {
        "score_contract": {
            "per_step_score_range": [0.0, 1.0],
            "episode_score_range": [0.0, 1.0],
            "episode_score_definition": "running average of normalized step scores",
        },
        "tasks": list(PUBLIC_TASK_IDS),
        "seeds": list(seeds),
        "policies": policy_results,
        "aggregate_hard_ordering": sorted(
            policy_results,
            key=lambda name: policy_results[name]["aggregate_scores"]["hard"],
            reverse=True,
        ),
        "aggregate_expert_ordering": sorted(
            policy_results,
            key=lambda name: policy_results[name]["aggregate_scores"]["expert"],
            reverse=True,
        ),
        "aggregate_recovery_ordering": sorted(
            policy_results,
            key=lambda name: policy_results[name]["aggregate_scores"]["recovery"],
            reverse=True,
        ),
        "aggregate_silent_ordering": sorted(
            policy_results,
            key=lambda name: policy_results[name]["aggregate_scores"]["silent"],
            reverse=True,
        ),
    }


def _generate_endpoint_capture(base_url: str, benchmark_results: dict[str, Any]) -> dict[str, Any]:
    tasks = _http_json(base_url, "/tasks")
    baseline = _http_json(base_url, "/baseline", method="POST", payload={})
    episode_created = _http_json(
        base_url,
        "/episodes",
        method="POST",
        payload={"task": "hard", "seed": 0},
    )
    episode_observation = CanaryObservation.model_validate(episode_created["observation"])
    episode_step = _http_json(
        base_url,
        f"/episodes/{episode_created['episode_id']}/step",
        method="POST",
        payload={
            "action": safe_fallback_action(episode_observation).model_dump(
                exclude={"metadata"}
            )
        },
    )
    return {
        "health": _http_json(base_url, "/health"),
        "tasks_summary": {
            "task_ids": [task["id"] for task in tasks["tasks"]],
            "slo_thresholds": tasks["slo_thresholds"],
            "scoring_weights": tasks["scoring_weights"],
            "notes": tasks["notes"],
        },
        "baseline_summary": {
            "agent_type": baseline["agent_type"],
            "scores": baseline["scores"],
            "average": baseline["average"],
            "normalized": baseline["normalized"],
        },
        "recommended_http_episode_sample": {
            "create_episode": {
                "episode_id": "artifact-generated",
                "episode_context": episode_created["episode_context"],
                "done": episode_created["done"],
                "observation": episode_created["observation"],
            },
            "first_step": {
                "reward": episode_step["reward"],
                "done": episode_step["done"],
                "evaluation": episode_step["evaluation"],
                "observation": episode_step["observation"],
            },
        },
        "policy_benchmark_hard_ordering": benchmark_results["hard_task_ordering"],
        "policy_benchmark_recovery_ordering": benchmark_results["recovery_task_ordering"],
        "policy_benchmark_silent_ordering": benchmark_results["silent_task_ordering"],
    }


def _generate_audit_summary(
    benchmark_results: dict[str, Any],
    seed_sweep_results: dict[str, Any],
    validation_results: dict[str, Any],
    cautious_episode: dict[str, Any],
    aggressive_episode: dict[str, Any],
    watch_window_grader: dict[str, Any],
) -> str:
    shallow = benchmark_results["policies"]["shallow_baseline"]["scores"]
    cautious = benchmark_results["policies"]["cautious_policy"]["scores"]
    aggressive = benchmark_results["policies"]["aggressive_policy"]["scores"]
    return "\n".join(
        [
            "# Review Artifact Summary",
            "",
            "This folder was generated by `python generate_review_artifacts.py`.",
            "",
            "## What It Proves",
            "",
            f"- `{ARTIFACT_FILENAMES['inference_stdout']}` shows the strict `[START]`, `[STEP]`, `[END]` stdout contract with no extra lines.",
            f"- `{ARTIFACT_FILENAMES['cautious_transcript']}` captures the full cautious hard-task run used in the benchmark comparison.",
            f"- `{ARTIFACT_FILENAMES['expert_transcript']}` captures the full cautious expert-task run used to verify phantom-alert handling.",
            f"- `{ARTIFACT_FILENAMES['recovery_transcript']}` captures the full cautious recovery-task run used to verify transient degradation handling.",
            f"- `{ARTIFACT_FILENAMES['aggressive_transcript']}` captures a hard-task aggressive rollout example that promotes through the watch window and rolls back late.",
            f"- `{ARTIFACT_FILENAMES['expert_grader']}` isolates the expert phantom-alert hold decision that should score well.",
            f"- `{ARTIFACT_FILENAMES['recovery_grader']}` isolates the recovery hold decision that should score well without rollback.",
            f"- `{ARTIFACT_FILENAMES['watch_window_grader']}` isolates the watch-window hold decision that makes the hard task meaningfully discriminative.",
            f"- `{ARTIFACT_FILENAMES['benchmark']}` compares shallow, cautious, and aggressive policies across easy, medium, hard, expert, recovery, and silent.",
            f"- `{ARTIFACT_FILENAMES['seed_sweep']}` shows the same policy ordering across deterministic seeds {', '.join(str(seed) for seed in seed_sweep_results['seeds'])}.",
            f"- `{ARTIFACT_FILENAMES['variant_catalog']}` exposes how warning, noise, phantom, recovery-clear, and breach timing vary across seeds for every public task family.",
            f"- `{ARTIFACT_FILENAMES['validation']}` records both structural and live OpenEnv validator results from the same local run.",
            f"- `{ARTIFACT_FILENAMES['endpoint_sample']}` captures the validator-safe endpoints plus a representative `/episodes` HTTP flow.",
            "- the grader is now fully structured; free-text reasoning is retained for transcript clarity but does not affect score.",
            "",
            "## Policy Benchmark Snapshot",
            "",
            f"- shallow baseline: easy={shallow['easy']:.4f}, medium={shallow['medium']:.4f}, hard={shallow['hard']:.4f}, expert={shallow['expert']:.4f}, recovery={shallow['recovery']:.4f}, silent={shallow['silent']:.4f}",
            f"- cautious policy: easy={cautious['easy']:.4f}, medium={cautious['medium']:.4f}, hard={cautious['hard']:.4f}, expert={cautious['expert']:.4f}, recovery={cautious['recovery']:.4f}, silent={cautious['silent']:.4f}",
            f"- aggressive policy: easy={aggressive['easy']:.4f}, medium={aggressive['medium']:.4f}, hard={aggressive['hard']:.4f}, expert={aggressive['expert']:.4f}, recovery={aggressive['recovery']:.4f}, silent={aggressive['silent']:.4f}",
            (
                "- seed-sweep aggregate hard ordering: "
                + ", ".join(seed_sweep_results["aggregate_hard_ordering"])
            ),
            (
                "- seed-sweep aggregate expert ordering: "
                + ", ".join(seed_sweep_results["aggregate_expert_ordering"])
            ),
            (
                "- seed-sweep aggregate recovery ordering: "
                + ", ".join(seed_sweep_results["aggregate_recovery_ordering"])
            ),
            (
                "- seed-sweep aggregate silent ordering: "
                + ", ".join(seed_sweep_results["aggregate_silent_ordering"])
            ),
            (
                f"- validation summary: structural_passed={validation_results['structural']['passed']} "
                f"live_passed={validation_results['live']['passed']}"
            ),
            "",
            "## Hard-Task Contrast",
            "",
            f"- cautious hard score={cautious_episode['episode_score']:.4f} with rollback timing={cautious_episode['rollback_assessment']['relative_to_benchmark_point']}",
            f"- aggressive hard example score={aggressive_episode['episode_score']:.4f} with rollback timing={aggressive_episode['rollback_assessment']['relative_to_benchmark_point']}",
            (
                f"- watch-window hold grader={watch_window_grader['policy_assessment']} "
                f"score={watch_window_grader['total_score']:.4f}"
            ),
            "",
        ]
    )


def generate_review_artifacts(output_dir: Path, env_url: str | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_managed_artifacts(output_dir)

    process: subprocess.Popen[str] | None = None
    log_handle = None
    base_url = env_url
    if base_url is None:
        process, log_handle, base_url = _start_local_server(output_dir)
        _wait_for_health(base_url)

    try:
        assert base_url is not None
        _run_inference_sample(base_url, output_dir)

        cautious_episode, _ = _run_policy_episode_via_http(
            base_url,
            task_id="hard",
            policy_fn=safe_fallback_action,
        )
        cautious_grader_entry = cautious_episode["transcript"][-1]
        cautious_grader = _http_json(
            base_url,
            "/grader",
            method="POST",
            payload=_build_grader_request(
                task_id="hard",
                observation=cautious_grader_entry["pre_observation"],
                action=cautious_grader_entry["action"],
            ),
        )

        aggressive_episode, _ = _run_episode_via_http(
            base_url,
            task_id="hard",
            actions=BAD_HARD_ACTIONS,
        )
        aggressive_grader_entry = _find_transcript_entry(
            aggressive_episode,
            preferred_assessments=(
                "reckless_warning_promotion",
                "risky_warning_promotion",
                "dangerous_breach_promotion",
            ),
        )
        aggressive_grader = _http_json(
            base_url,
            "/grader",
            method="POST",
            payload=_build_grader_request(
                task_id="hard",
                observation=aggressive_grader_entry["pre_observation"],
                action=aggressive_grader_entry["action"],
            ),
        )

        watch_window_episode, _ = _run_policy_episode_via_http(
            base_url,
            task_id="hard",
            policy_fn=safe_fallback_action,
            stop_after_steps=5,
        )
        watch_window_grader_entry = _find_transcript_entry(
            watch_window_episode,
            preferred_assessments=("correct_warning_hold",),
        )
        watch_window_grader = _http_json(
            base_url,
            "/grader",
            method="POST",
            payload=_build_grader_request(
                task_id="hard",
                observation=watch_window_grader_entry["pre_observation"],
                action=watch_window_grader_entry["action"],
            ),
        )

        expert_episode, _ = _run_policy_episode_via_http(
            base_url,
            task_id="expert",
            policy_fn=safe_fallback_action,
        )
        expert_grader_entry = _find_transcript_entry(
            expert_episode,
            preferred_assessments=("correct_phantom_ignore",),
        )
        expert_grader = _http_json(
            base_url,
            "/grader",
            method="POST",
            payload=_build_grader_request(
                task_id="expert",
                observation=expert_grader_entry["pre_observation"],
                action=expert_grader_entry["action"],
            ),
        )

        recovery_episode, _ = _run_policy_episode_via_http(
            base_url,
            task_id="recovery",
            policy_fn=safe_fallback_action,
        )
        recovery_grader_entry = _find_transcript_entry(
            recovery_episode,
            preferred_assessments=("correct_recovery_hold",),
        )
        recovery_grader = _http_json(
            base_url,
            "/grader",
            method="POST",
            payload=_build_grader_request(
                task_id="recovery",
                observation=recovery_grader_entry["pre_observation"],
                action=recovery_grader_entry["action"],
            ),
        )

        benchmark_results = benchmark_policies()
        seed_sweep_results = benchmark_seed_sweep()
        variant_catalog = scenario_variant_catalog()
        validation_results = {
            "structural": _run_openenv_validate("."),
            "live": _run_openenv_validate("--url", base_url),
        }
        endpoint_capture = _generate_endpoint_capture(base_url, benchmark_results)
        artifact_manifest = _build_artifact_manifest(
            benchmark_results=benchmark_results,
            seed_sweep_results=seed_sweep_results,
            validation_results=validation_results,
            cautious_episode=cautious_episode,
            cautious_grader=cautious_grader,
            aggressive_episode=aggressive_episode,
            aggressive_grader=aggressive_grader,
            watch_window_episode=watch_window_episode,
            watch_window_grader=watch_window_grader,
            expert_episode=expert_episode,
            expert_grader=expert_grader,
            recovery_episode=recovery_episode,
            recovery_grader=recovery_grader,
        )

        _write_json(output_dir / ARTIFACT_FILENAMES["cautious_transcript"], cautious_episode)
        _write_json(output_dir / ARTIFACT_FILENAMES["cautious_grader"], cautious_grader)
        _write_json(output_dir / ARTIFACT_FILENAMES["aggressive_transcript"], aggressive_episode)
        _write_json(output_dir / ARTIFACT_FILENAMES["aggressive_grader"], aggressive_grader)
        _write_json(output_dir / ARTIFACT_FILENAMES["watch_window_transcript"], watch_window_episode)
        _write_json(output_dir / ARTIFACT_FILENAMES["watch_window_grader"], watch_window_grader)
        _write_json(output_dir / ARTIFACT_FILENAMES["expert_transcript"], expert_episode)
        _write_json(output_dir / ARTIFACT_FILENAMES["expert_grader"], expert_grader)
        _write_json(output_dir / ARTIFACT_FILENAMES["recovery_transcript"], recovery_episode)
        _write_json(output_dir / ARTIFACT_FILENAMES["recovery_grader"], recovery_grader)
        _write_json(output_dir / ARTIFACT_FILENAMES["benchmark"], benchmark_results)
        _write_json(output_dir / ARTIFACT_FILENAMES["seed_sweep"], seed_sweep_results)
        _write_json(output_dir / ARTIFACT_FILENAMES["variant_catalog"], variant_catalog)
        _write_json(output_dir / ARTIFACT_FILENAMES["validation"], validation_results)
        _write_json(output_dir / ARTIFACT_FILENAMES["endpoint_sample"], endpoint_capture)
        _write_json(output_dir / ARTIFACT_FILENAMES["manifest"], artifact_manifest)
        _write_text(
            output_dir / ARTIFACT_FILENAMES["index"],
            _artifact_index(artifact_manifest),
        )
        _write_text(
            output_dir / ARTIFACT_FILENAMES["audit"],
            _generate_audit_summary(
                benchmark_results=benchmark_results,
                seed_sweep_results=seed_sweep_results,
                validation_results=validation_results,
                cautious_episode=cautious_episode,
                aggressive_episode=aggressive_episode,
                watch_window_grader=watch_window_grader,
            ),
        )

        return {
            "output_dir": str(output_dir),
            "base_url": base_url,
            "policy_benchmark_results": benchmark_results,
            "cautious_hard_score": cautious_episode["episode_score"],
            "aggressive_hard_score": aggressive_episode["episode_score"],
            "watch_window_hold_score": watch_window_grader["total_score"],
        }
    finally:
        if process is not None and log_handle is not None:
            _stop_local_server(process, log_handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reviewer-facing proof artifacts under review_artifacts/."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where reviewer artifacts will be written.",
    )
    parser.add_argument(
        "--env-url",
        default="",
        help="Reuse an already running environment instead of launching a temporary local server.",
    )
    args = parser.parse_args()

    result = generate_review_artifacts(
        output_dir=Path(args.output_dir).resolve(),
        env_url=args.env_url.strip() or None,
    )
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "cautious_hard_score": result["cautious_hard_score"],
                "aggressive_hard_score": result["aggressive_hard_score"],
                "watch_window_hold_score": result["watch_window_hold_score"],
                "hard_task_ordering": result["policy_benchmark_results"]["hard_task_ordering"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
