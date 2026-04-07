---
title: Canary Release Manager
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---
# Canary Release Manager

Canary Release Manager is a production-minded OpenEnv benchmark for canary rollout decision-making. An agent observes stable-versus-canary service metrics for a new recommendation model and decides whether to:

- `increase_5`
- `increase_10`
- `increase_25`
- `hold`
- `rollback`

This project is not a live rollout controller.

This project is a reusable evaluation environment for comparing rollout agents, scripted policies, and decision-support behavior under realistic canary ambiguity.

## Overview

The benchmark is built around three properties that matter in real rollout reviews:

- rollout decisions are sequential and stateful
- absolute thresholds are not enough; stable-versus-canary comparisons matter
- the hardest failures are not immediate crashes, but ambiguous states where hold, small probes, and rollback timing all matter

The environment is deterministic by design. The grader always evaluates the action against the observation the agent just saw, never the post-action state.

## Why Canary Decisions Are Hard

Real canary rollout decisions are not just threshold checks.

- A shared infrastructure spike can make both stable and canary look bad at the same time.
- A gradual regression can create a warning window where a 5% probe is safer than a full 10% promotion.
- A post-noise differential drift can make `hold` the correct action even though no hard breach has fired yet.

The benchmark is designed so agents must separate healthy promotion, watch states, shared noise, and true canary breach.

## Project Positioning

Canary Release Manager is intended to feel like an internal benchmark asset:

- validator-safe Phase 1 surface
- deterministic public grading
- interpretable action space
- inspectable transcripts and episode summaries
- enough realism to compare shallow versus thoughtful policies

It deliberately avoids live orchestration, persistent databases, RL infrastructure, or UI work.

## Workspace Layout

```text
canary_release_env/          ← repo root (clone this)
├── server/
│   └── Dockerfile
├── tests/
├── baseline.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
└── README.md
```

## What The Environment Simulates

### Observation Schema

Each agent-facing observation includes telemetry, rollout progress, and the active guardrails:

- `traffic_pct`
- `canary_error_rate`
- `canary_p99_ms`
- `stable_error_rate`
- `stable_p99_ms`
- `differential_error`
- `differential_p99_ms`
- `alert_count`
- `step_number`
- `consecutive_holds`
- `rollback_on_error_rate`
- `rollback_on_canary_p99_ms`
- `rollback_on_differential_p99_ms`

The agent does **not** see reviewer-only grading fields such as `actual_breach`, `policy_assessment`, `reward_breakdown`, `step_explanation`, or task labels in the main observation payload.

Those evaluator-facing details remain available through:

- `POST /grader`
- episode transcripts
- `/episodes/*` review helpers via the top-level `evaluation` payload

### Action Schema

Agents send a structured action:

```json
{
  "action": {
    "action_type": "increase_10",
    "reasoning": "Stable and canary remain healthy with low error and no differential latency gap, so continue the standard 10% rollout step.",
    "slo_breach_detected": false,
    "state_assessment": "healthy"
  }
}
```

`state_assessment` is a required structured diagnosis of the current rollout state:

- `healthy`
- `warning`
- `noise`
- `phantom_alert`
- `breach`

This public schema is intentionally coarse. The grader keeps richer hidden rollout subtypes internally, but agents only need to classify the observation into the public family that best matches what they see.

The public reasoning score is now fully structured around the diagnosis fields. Free-text reasoning is kept for transcript clarity and reviewer inspection, but it does not affect score.

## Task Descriptions

### `easy`

Named pattern: big bang regression.

- The canary looks healthy at first.
- Once traffic crosses the first threshold, the regression is obvious.
- The task checks whether the agent can observe and rollback promptly.

### `medium`

Named pattern: boiling frog regression.

- The canary drifts before it breaches.
- At the warning window, a 5% probe is better than a full 10% promotion.
- A shallow threshold-only policy rolls forward too aggressively and gives up score before the breach is obvious.

### `hard`

Named pattern: headroom cliff with infrastructure noise.

- At 30% traffic, stable and canary both hit a shared latency spike.
- The correct action on that step is `hold`, not rollback.
- After the noise clears, the canary stays slower than stable in a post-noise watch window.
- A thoughtful agent holds once more and then rolls back on the confirmed differential breach.
- A shallow agent promotes through the warning window and rolls back late relative to the benchmark rollback point.

### `expert`

Named pattern: phantom alert storm.

- Early in the rollout a monitoring burst fires `alert_count=4` while all raw metrics are healthy.
- Rolling back on alerts alone is wrong — this is a false positive.
- The correct action during the phantom step is `hold` or a cautious increase after verifying the metrics.
- After the phantom clears, a real canary-specific differential drift builds toward breach at 35% traffic.
- An agent that correctly ignores the phantom and then catches the real breach scores well. An agent that rollbacks on `alert_count` alone loses score on the phantom step.

### `recovery`

Named pattern: transient hot shard recovery.

- A canary-specific latency spike appears during rollout but stays below rollback guardrails.
- The right response is a watchful `hold`, not a panic rollback and not an aggressive promotion.
- After a short observation window the canary recovers and the rollout can continue safely.
- This task punishes agents that treat every canary-specific slowdown as a breach.

### `silent`

Named pattern: silent differential burn.

- The canary develops a real stable-versus-canary differential drift with weak alert support.
- The correct behavior is to react to raw telemetry rather than waiting for `alert_count` to confirm the issue.
- A thoughtful agent holds through the early silent warning state and then rolls back once the differential breach is real.
- A shallow policy that only trusts alerts or ignores differential drift promotes too long.

## Reward And Score Semantics

The public score contract is strict and normalized:

- the top-level `/step` `reward` is the normalized score for the latest action in `[0.0, 1.0]`
- `episode_score` in transcripts, episode summaries, and `/baseline` details is the running average episode score in `[0.0, 1.0]`
- `POST /grader` returns a normalized single-step score in `[0.0, 1.0]`
- `POST /baseline` returns normalized episode averages in `[0.0, 1.0]`

These values are related but not interchangeable:

- top-level `/step` `reward` answers: how good was the most recent decision?
- `episode_score` answers: how good has the rollout been so far?
- `total_score` from `/grader` answers: how good is this one action under the public grader?
- `average` from `/baseline` answers: how the deterministic benchmark policy performs across tasks

No public code path returns a negative score.

## Grader Methodology

The public grader uses four components:

- `breach_detection_score` = `0.35`
- `rollback_timing_score` = `0.25`
- `promotion_safety_score` = `0.30`
- `reasoning_score` = `0.10`

Important rules:

- grading uses the pre-action observation
- shared-noise steps are detected explicitly
- phantom alert steps are detected and reward holding or promoting (not rollback)
- medium warning states reward a cautious `increase_5`
- hard warning states reward `hold`
- recovery warning states reward `hold`
- silent warning states reward `hold`
- actual breaches reward rollback
- aggressive promotions and premature rollbacks lose score without going negative
- the reasoning score is intentionally conservative, mostly structured, and low-weight

Internally, the grader distinguishes richer hidden states such as trend warnings, post-noise watch windows, transient recovery windows, and silent differential drifts. Those hidden subtypes drive action-fit and timing, while the public action schema stays coarse enough to remain realistic and harder to overfit against.

The grader is deterministic and the transcript exposes the full per-step breakdown.

## Standard Endpoints vs `/episodes/*`

OpenEnv standard endpoints:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /grader`
- `POST /baseline`
- `WS /ws`

Extra stateful REST helpers:

- `POST /episodes`
- `POST /episodes/{episode_id}/step`
- `GET /episodes/{episode_id}/state`
- `GET /episodes/{episode_id}/transcript`

Contract notes:

- `WS /ws` is the canonical persistent OpenEnv interface
- `/episodes/*` is the recommended plain-HTTP interface for real stateful integrations
- the intended plain-HTTP flow is `POST /episodes -> POST /episodes/{episode_id}/step -> GET /episodes/{episode_id}/transcript`
- standard HTTP `reset/step/state` remains validator-safe and exposes the stripped agent observation contract
- separate HTTP requests are stateless by framework design
- `POST /step` without an active persistent session returns an explicit `uninitialized` observation instead of silently defaulting to a task
- `/episodes/*` is the simplest plain-HTTP path for stateful debugging, transcript review, and evaluator-facing `evaluation` payloads
- `POST /reset` accepts the standard OpenEnv `seed` field for deterministic scenario variants; `seed=0` is the canonical documented profile
- non-zero seeds vary event timing, alert intensity, drift onset, and breach timing within each scenario family
- `/episodes` rejects unknown task IDs with `422`; standard `/reset` stays validator-safe by defaulting unknown task IDs to `easy`

## Episode Inspectability

Each transcript step includes:

- `pre_observation`
- `action`
- `actual_breach`
- `policy_assessment`
- `reward_breakdown`
- `explanation`
- `post_observation`

The observation snapshots include `consecutive_holds`, which makes grader replay easier for reviewer inspection and edge-case debugging.

Final episode results also include:

- `episode_score`
- `score_breakdown`
- `first_breach_point`
- `rollback_assessment`
- `episode_summary`

`rollback_assessment` is structured rather than hand-wavy. It reports:

- whether a rollback action happened
- whether it happened before, on, or after the first confirmed breach
- whether it was early, timely, or late relative to the benchmark rollback point
- the benchmark rollback threshold and rollback traffic percentage used for that judgment

That keeps transcript review aligned with the actual timing score instead of collapsing distinct concepts into one label.

## Local Setup

From the environment root:

```bash
pip install -e .
```

Run the app directly:

```bash
python server/app.py
```

Run with Uvicorn from the repository root:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

After `pip install -e .`, the installed package entrypoint also works:

```bash
python -m canary_release_env.server.app
```

## Validation Commands

Structural validation:

```bash
python -m openenv.cli validate .
```

Runtime validation against a live local server:

```bash
python -m openenv.cli validate --url http://127.0.0.1:7860
```

Offline tests:

```bash
python -m unittest discover -s tests -v
```

Deterministic baseline:

```bash
python baseline.py
```

## Docker

Build from the repository root:

```bash
docker build -t canary-release-env .
```

The validator-compatible alternate path remains available:

```bash
docker build -t canary-release-env -f server/Dockerfile .
```

Run locally (container port 7860, mapped to host port 8001):

```bash
docker run --rm -p 8001:7860 canary-release-env
```

Smoke test:

```bash
curl http://127.0.0.1:8001/health
```

The repository-root `Dockerfile` is the authoritative Hugging Face Docker Space target. `server/Dockerfile` is intentionally kept byte-for-byte aligned for validator compatibility, and the test suite checks that they stay in sync.

The image uses a single worker intentionally because `/episodes/*` stores live in-memory episode state. A `.dockerignore` file excludes local caches, tests, and logs from the runtime image context.

## Hugging Face Spaces Notes

The environment is configured for the standard Space runtime shape:

- repository-root `Dockerfile` is the primary Space build target
- FastAPI app entrypoint is `server.app:app`
- the same code also supports `canary_release_env.server.app:app` after installation
- runtime port is `7860`
- `server/Dockerfile` remains available as a validator-compatible mirror of the root image definition
- the container command is JSON-form and delegates port handling to `server.app.main()`

This project is still an evaluation environment. A Space deployment should expose the benchmark surface, not behave like a production rollout controller.

For final submission hygiene, see [SUBMISSION_CHECKLIST.md](/c:/Users/lekha/Downloads/canary_release_env/canary_release_env/SUBMISSION_CHECKLIST.md).

## `inference.py`

`inference.py` is kept at the environment root for validator compatibility.

Environment variables read by the script:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME`

Stdout contract:

- `[START]`
- `[STEP]`
- `[END]`

There are no extra summary lines and no debug prints.

Decision-source behavior is explicit:

- if `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are all unset, the script runs in explicit `model=fallback` mode
- if model output is malformed, that step is treated as a degraded model failure
- if a configured model call fails or returns malformed output, the step still records a containment action for runner safety but the `[STEP] error=...` field is populated and the final `[END] success=false` makes the degraded run visible
- `[END] success=true` means the inference runner completed without a degraded model failure; it does not mean the rollout outcome was promotion to 100% traffic

The fallback policy is intentionally safer than `POST /baseline`. It is a runner safety net, not the benchmark baseline.

Run in fully local mode with the in-process environment:

```bash
python inference.py
```

Run against a Docker image when the evaluator provides `LOCAL_IMAGE_NAME`:

```bash
export LOCAL_IMAGE_NAME="canary-release-env"
python inference.py
```

Run against a locally served app on `7860`:

```bash
export API_BASE_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_NAME="your-model"
export HF_TOKEN="your-token"
python inference.py --env-url http://127.0.0.1:7860
```

Run against the Docker host mapping:

```bash
python inference.py --env-url http://127.0.0.1:8001
```

## Review Artifacts

Generate the full reviewer-facing evidence pack locally:

```bash
python generate_review_artifacts.py
```

This creates `review_artifacts/` with:

- `inference_stdout_fallback_sample.txt`
- `hard_cautious_transcript.json`
- `hard_cautious_grader.json`
- `hard_aggressive_transcript.json`
- `hard_aggressive_grader.json`
- `hard_watch_window_hold_transcript.json`
- `hard_watch_window_hold_grader.json`
- `expert_cautious_transcript.json`
- `expert_phantom_hold_grader.json`
- `recovery_cautious_transcript.json`
- `recovery_transient_hold_grader.json`
- `policy_benchmark_results.json`
- `policy_seed_sweep_results.json`
- `scenario_variant_catalog.json`
- `openenv_validation_results.json`
- `endpoint_contract_sample.json`
- `artifact_manifest.json`
- `artifact_index.md`
- `benchmark_audit_summary.md`

By default the script launches a temporary local server on a free port, waits for `/health`, captures the artifacts, and shuts the server down. If you already have the environment running, reuse it:

```powershell
python generate_review_artifacts.py --env-url http://127.0.0.1:7860
```

After `pip install -e .`, the packaged entrypoint also works:

```powershell
review-artifacts
```

Artifact naming rules:

- `hard_cautious_*` means a full hard-task run from `cautious_policy`
- `hard_aggressive_*` means a hard-specific aggressive rollout example used to expose the watch-window mistake clearly
- `hard_watch_window_hold_*` means the hard-task watch-window hold case, which is intentionally a partial in-progress trace plus a single-step grader capture
- `expert_cautious_*` means the full expert-task run from the cautious policy
- `expert_phantom_hold_*` means the expert phantom-alert decision that should score well without rollback
- `recovery_cautious_*` means the full recovery-task run from the cautious policy
- `recovery_transient_hold_*` means the recovery warning-window hold decision that should score well without rollback
- `policy_seed_sweep_results.json` means the same policy comparison rerun across multiple deterministic seeds, not just the canonical seed
- `scenario_variant_catalog.json` means a machine-readable event catalog showing how warning, noise, phantom, recovery-clear, and breach timing change across deterministic seeds
- `openenv_validation_results.json` means structural and live OpenEnv validation were re-run during artifact generation
- `*_transcript.json` is an episode transcript or partial episode transcript
- `*_grader.json` is a single `POST /grader` payload result for one decision

Interpretation rules:

- `hard_cautious_transcript.json` is a completed cautious-policy hard episode, so its `episode_score` should line up with the `cautious_policy` hard score in `policy_benchmark_results.json`
- `hard_aggressive_transcript.json` is an illustrative hard-task rollout example, not the multi-task `aggressive_policy` benchmark comparator; use it to inspect the watch-window failure mode, not to read the benchmark aggregate
- `hard_watch_window_hold_transcript.json` is intentionally `in_progress`; its `episode_score` is a running average through the hold decision, not a completed-policy benchmark score
- `expert_cautious_transcript.json` is a completed expert-task episode and should align with the cautious policy expert score in `policy_benchmark_results.json`
- `expert_phantom_hold_grader.json` is a single-step grader result for the phantom-alert hold decision
- `recovery_cautious_transcript.json` is a completed recovery-task episode and should align with the cautious policy recovery score in `policy_benchmark_results.json`
- `recovery_transient_hold_grader.json` is a single-step grader result for the transient-recovery hold decision
- `policy_seed_sweep_results.json` reports aggregate policy scores across the documented deterministic seeds and is the best evidence that the ordering is not tied to a single authored trace
- `scenario_variant_catalog.json` reports where each task family first enters warning, noise, phantom, recovery-clear, and breach states under a fixed probe rollout, so reviewers can inspect breadth directly instead of inferring it from code
- `openenv_validation_results.json` records the structural validator result and the live local-server validator result from the same generated artifact run
- any `*_grader.json` file reports a single-step `total_score`, not an episode average
- `artifact_manifest.json` and `artifact_index.md` both explain these distinctions so reviewers do not have to infer them from filenames alone

## How To Inspect This Benchmark Quickly

- `review_artifacts\artifact_index.md`: start here for the reviewer-friendly map of the artifact pack
- `review_artifacts\policy_benchmark_results.json`: shows the canonical seed=0 comparison across shallow, cautious, and aggressive policies
- `review_artifacts\policy_seed_sweep_results.json`: shows the same comparison across multiple deterministic seeds
- `review_artifacts\scenario_variant_catalog.json`: shows how deterministic seeds change event ordering and signal signatures across every public task family
- `review_artifacts\openenv_validation_results.json`: proves structural and live validator compatibility from the generated artifact run
- `review_artifacts\hard_cautious_transcript.json`: proves the hard task has a coherent successful cautious-policy path
- `review_artifacts\expert_cautious_transcript.json`: proves the expert task handles phantom alerts and later catches the real breach
- `review_artifacts\recovery_cautious_transcript.json`: proves the recovery task rewards watchful holds and later safe promotion
- `review_artifacts\hard_aggressive_grader.json`: shows the exact aggressive watch-window promotion mistake and how it scores
- `review_artifacts\expert_phantom_hold_grader.json`: shows the exact phantom-alert hold decision and how it scores
- `review_artifacts\recovery_transient_hold_grader.json`: shows the exact recovery hold decision and how it scores
- `review_artifacts\inference_stdout_fallback_sample.txt`: proves the strict inference stdout contract

## Expected Baseline Scores

Current deterministic `POST /baseline` output:

- `easy`: `0.9567`
- `medium`: `0.8460`
- `hard`: `0.8217`
- `expert`: `0.6967`
- `recovery`: `0.9109`
- `silent`: `0.8360`
- `average`: `0.8447`

Interpretation:

- easy is intentionally boring
- medium penalizes overconfident promotion through a warning window
- hard is lower because a shallow policy still misses the post-noise watch state
- expert is the lowest because the baseline policy rolls back on `alert_count > 0`, which fires incorrectly on the phantom step
- recovery lands well above expert because the baseline does not overreact when the transient spike stays below guardrails, but it still trails the cautious policy because it does not explicitly reason about the warning window
- silent stays below recovery because the baseline does not pause early on weak-alert differential drift unless the breach becomes obvious

## Golden Path Examples

### Easy

Good:

- `increase_10`
- `increase_5`
- `rollback`

Bad:

- `increase_10`
- `increase_10`
- `increase_10`

### Medium

Good:

- `increase_10`
- `increase_10`
- `increase_10`
- `increase_5`
- `rollback`

Bad:

- `increase_10`
- `increase_10`
- `increase_10`
- `increase_10`
- `rollback`

### Hard

Good:

- `increase_10`
- `increase_10`
- `increase_10`
- `hold`
- `hold`
- `rollback`

Bad:

- `increase_10`
- `increase_10`
- `increase_10`
- `hold`
- `increase_10`
- `rollback`

### Expert

Good:

- `increase_10`
- `increase_10`
- `hold` ← phantom alert fires here; do NOT rollback
- `increase_10`
- `increase_5`
- `rollback`

Bad:

- `increase_10`
- `increase_10`
- `rollback` ← false positive; alert_count fired but metrics are healthy
- (episode ends prematurely)

## Known Limitations

- The environment is deterministic and scenario-authored; it does not model the full stochasticity of live production telemetry.
- The public benchmark still uses an authored task family, but each task supports bounded deterministic seeds that change event timing and signal shape, and the review artifacts include both a multi-seed policy sweep and a scenario variant catalog so reviewers can inspect robustness beyond `seed=0`.
- Episode state is in-memory and intentionally single-worker.
- Completed `/episodes/*` transcripts are retained in a bounded in-memory FIFO cache, not a persistent store.
- The public reasoning score is fully structured by design; the free-text explanation is retained for transcript readability, not prose grading.
- There is no persistent experiment store or multi-service topology simulator.

## Future Extensions

- add more benchmark tasks for infra-capacity coupling and multi-stage recoveries
- add richer alert metadata without weakening determinism
- add offline transcript analysis utilities for evaluator reports
