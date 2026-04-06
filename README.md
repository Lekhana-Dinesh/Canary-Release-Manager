---
title: Canary Release Manager
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
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

The local workspace has a nested layout:

```text
C:\Users\lekha\Downloads\canary_release_env
\-- canary_release_env
    +-- server
    |   \-- Dockerfile
    +-- tests
    +-- baseline.py
    +-- client.py
    +-- inference.py
    +-- models.py
    +-- openenv.yaml
    \-- README.md
```

The inner `canary_release_env` directory is the actual environment root. Do not move the Dockerfile to the outer folder.

## What The Environment Simulates

### Observation Schema

Each observation includes the current rollout state plus public scoring feedback:

- `traffic_pct`
- `canary_error_rate`
- `canary_p99_ms`
- `stable_error_rate`
- `stable_p99_ms`
- `differential_error`
- `differential_p99_ms`
- `alert_count`
- `step_number`
- `step_reward`
- `cumulative_reward`
- `consecutive_holds`
- `actual_breach`
- `policy_assessment`
- `reward_breakdown`
- `step_explanation`
- `task_id`
- `task_description`
- `feedback`

### Action Schema

Agents send a structured action:

```json
{
  "action": {
    "action_type": "increase_10",
    "reasoning": "Stable and canary remain healthy with low error and no differential latency gap, so continue the standard 10% rollout step.",
    "slo_breach_detected": false
  }
}
```

`reasoning` is public but deliberately low-weight. It is supportive evidence, not the primary score source.

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

## Reward And Score Semantics

The public score contract is strict and normalized:

- `step_reward` is the normalized score for the latest action in `[0.0, 1.0]`
- `cumulative_reward` is the running average episode score in `[0.0, 1.0]`
- `POST /grader` returns a normalized single-step score in `[0.0, 1.0]`
- `POST /baseline` returns normalized episode averages in `[0.0, 1.0]`

These values are related but not interchangeable:

- `step_reward` answers: how good was the most recent decision?
- `cumulative_reward` answers: how good has the episode been so far?
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
- medium warning states reward a cautious `increase_5`
- hard warning states reward `hold`
- actual breaches reward rollback
- aggressive promotions and premature rollbacks lose score without going negative
- the reasoning score is intentionally conservative and low-weight

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
- standard HTTP `reset/step/state` remains validator-safe
- separate HTTP requests are stateless by framework design
- `POST /step` without an active persistent session returns an explicit `uninitialized` observation instead of silently defaulting to a task
- `/episodes/*` is the simplest plain-HTTP path for stateful debugging and transcript review
- `/episodes` rejects unknown task IDs with `422`; standard `/reset` stays validator-safe by defaulting unknown task IDs to `easy` and saying so in `feedback`

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

```powershell
cd C:\Users\lekha\Downloads\canary_release_env\canary_release_env
python -m pip install -e .
```

Run the app directly:

```powershell
python server\app.py
```

Run with Uvicorn from the repository root:

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

After `pip install -e .`, the installed package entrypoint also works:

```powershell
python -m canary_release_env.server.app
```

## Validation Commands

Structural validation:

```powershell
cd C:\Users\lekha\Downloads\canary_release_env\canary_release_env
python -m openenv.cli validate .
```

Runtime validation against a live local server:

```powershell
cd C:\Users\lekha\Downloads\canary_release_env\canary_release_env
python -m openenv.cli validate --url http://127.0.0.1:7860
```

Offline tests:

```powershell
cd C:\Users\lekha\Downloads\canary_release_env\canary_release_env
python -m unittest discover -s tests -v
```

Deterministic baseline:

```powershell
cd C:\Users\lekha\Downloads\canary_release_env\canary_release_env
python baseline.py --url http://127.0.0.1:7860
```

## Docker

### Actual Working Build Command From The Outer Workspace

```powershell
cd C:\Users\lekha\Downloads\canary_release_env
docker build -t canary-release-env -f .\canary_release_env\server\Dockerfile .\canary_release_env
```

### Equivalent Command If You Are Already In The Environment Root

```powershell
cd C:\Users\lekha\Downloads\canary_release_env\canary_release_env
docker build -t canary-release-env -f .\server\Dockerfile .
```

### Local Run Command

The app listens on container port `7860`. For local external testing from the host, use:

```powershell
docker run --rm -p 8001:7860 canary-release-env
```

If port `8001` is already in use locally, map any free host port instead:

```powershell
docker run --rm -p 8002:7860 canary-release-env
```

Smoke test:

```powershell
curl http://127.0.0.1:8001/health
```

The image uses a single worker intentionally because `/episodes/*` stores live in-memory episode state. A `.dockerignore` file excludes local caches, tests, and logs from the runtime image context.

## Hugging Face Spaces Notes

The environment is configured for the standard Space runtime shape:

- FastAPI app entrypoint is `server.app:app`
- the same code also supports `canary_release_env.server.app:app` after installation
- runtime port is `7860`
- Docker path stays at `server/Dockerfile`
- the container command is JSON-form and delegates port handling to `server.app.main()`

This project is still an evaluation environment. A Space deployment should expose the benchmark surface, not behave like a production rollout controller.

## `inference.py`

`inference.py` is kept at the environment root for validator compatibility.

Environment variables read by the script:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Stdout contract:

- `[START]`
- `[STEP]`
- `[END]`

There are no extra summary lines, no debug prints, and malformed model output falls back to a deterministic safe policy.

That fallback policy is intentionally safer than `POST /baseline`. It is a runner safety net, not the benchmark baseline.

Run against a locally served app on `7860`:

```powershell
cd C:\Users\lekha\Downloads\canary_release_env\canary_release_env
$env:API_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:MODEL_NAME = "your-model"
$env:HF_TOKEN = "your-token"
python inference.py
```

Run against the Docker host mapping if you want to smoke test from the host:

```powershell
python inference.py --env-url http://127.0.0.1:8001
```

## Review Artifacts

Generate the full reviewer-facing evidence pack locally:

```powershell
cd C:\Users\lekha\Downloads\canary_release_env\canary_release_env
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
- `policy_benchmark_results.json`
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
- `*_transcript.json` is an episode transcript or partial episode transcript
- `*_grader.json` is a single `POST /grader` payload result for one decision

Interpretation rules:

- `hard_cautious_transcript.json` is a completed cautious-policy hard episode, so its `episode_score` should line up with the `cautious_policy` hard score in `policy_benchmark_results.json`
- `hard_aggressive_transcript.json` is an illustrative hard-task rollout example, not the multi-task `aggressive_policy` benchmark comparator; use it to inspect the watch-window failure mode, not to read the benchmark aggregate
- `hard_watch_window_hold_transcript.json` is intentionally `in_progress`; its `episode_score` is a running average through the hold decision, not a completed-policy benchmark score
- any `*_grader.json` file reports a single-step `total_score`, not an episode average
- `artifact_manifest.json` and `artifact_index.md` both explain these distinctions so reviewers do not have to infer them from filenames alone

## How To Inspect This Benchmark Quickly

- `review_artifacts\artifact_index.md`: start here for the reviewer-friendly map of the artifact pack
- `review_artifacts\policy_benchmark_results.json`: shows the benchmark-level comparison across shallow, cautious, and aggressive policies
- `review_artifacts\hard_cautious_transcript.json`: proves the hard task has a coherent successful cautious-policy path
- `review_artifacts\hard_aggressive_grader.json`: shows the exact aggressive watch-window promotion mistake and how it scores
- `review_artifacts\inference_stdout_fallback_sample.txt`: proves the strict inference stdout contract

## Expected Baseline Scores

Current deterministic `POST /baseline` output:

- `easy`: `0.9500`
- `medium`: `0.8380`
- `hard`: `0.8067`
- `average`: `0.8649`

Interpretation:

- easy is intentionally boring
- medium penalizes overconfident promotion through a warning window
- hard is lower because a shallow policy still misses the post-noise watch state

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

## Known Limitations

- The environment is deterministic and scenario-authored; it does not model the full stochasticity of live production telemetry.
- Episode state is in-memory and intentionally single-worker.
- Completed `/episodes/*` transcripts are retained in a bounded in-memory FIFO cache, not a persistent store.
- The public reasoning score is low-weight by design; it is supportive context, not a hidden chain-of-thought proxy.
- There is no persistent experiment store or multi-service topology simulator.

## Future Extensions

- add more benchmark tasks for silent quality regressions and infra-capacity coupling
- add richer alert metadata without weakening determinism
- add offline transcript analysis utilities for evaluator reports
