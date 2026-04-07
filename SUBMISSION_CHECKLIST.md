# Submission Checklist

This repo is intended to be submission-clean from the repository root.

## Before submitting

1. Ensure the intended files are committed.
   Target state: `git status --short` shows no unstaged or staged changes.
2. Run the offline test suite.
   Command: `python -m unittest discover -s tests -v`
3. Run structural OpenEnv validation.
   Command: `python -m openenv.cli validate .`
4. Run live OpenEnv validation against a local server.
   Commands:
   `uvicorn server.app:app --host 127.0.0.1 --port 7860`
   `python -m openenv.cli validate --url http://127.0.0.1:7860`
5. Build the Hugging Face Docker image from the repository root.
   Command: `docker build -t canary-release-env .`
6. Run the built image and verify the required endpoints.
   Commands:
   `docker run --rm -p 8001:7860 canary-release-env`
   `curl http://127.0.0.1:8001/health`
   `curl -X POST http://127.0.0.1:8001/reset -H "Content-Type: application/json" -d "{\"task\":\"easy\"}"`
7. Regenerate reviewer artifacts from the same tree.
   Command: `python generate_review_artifacts.py`
8. Confirm the artifact pack includes:
   `policy_benchmark_results.json`
   `policy_seed_sweep_results.json`
   `scenario_variant_catalog.json`
   `openenv_validation_results.json`
   `inference_stdout_fallback_sample.txt`
9. Record the exact commit you are about to submit.
   Command: `git rev-parse HEAD`
10. Deploy the repository root to a Hugging Face Docker Space.
   Required proof after deployment:
   the public Space URL
   a successful `POST /reset`
   a successful `python -m openenv.cli validate --url <space-url>`
11. Regenerate or re-collect proof against the deployed Space if you want hosted artifacts, not just local artifacts.
   Recommended commands:
   `python -m openenv.cli validate --url <space-url>`
   `python generate_review_artifacts.py --env-url <space-url>`
12. Submit the exact committed tree that produced the validation results and artifact pack.

## Deliberate interface choices

- `WS /ws` is the canonical OpenEnv stateful interface.
- `/episodes/*` is the recommended plain-HTTP stateful interface.
- plain `/reset` + `/step` + `/state` exists for OpenEnv compatibility and validator safety.

## Deliberate benchmark choices

- Agent-facing observations expose telemetry and guardrails only.
- Reviewer-only evaluation details live in transcripts, `/grader`, and `/episodes/*`.
- `seed=0` is the canonical documented profile.
- Non-zero seeds vary timing and signal shape within each scenario family and are covered by both `policy_seed_sweep_results.json` and `scenario_variant_catalog.json`.
