# Artifact Index

Start here if you want to inspect the benchmark quickly.

## Quick Read Order

- `artifact_index.md`: this reviewer guide.
- `policy_benchmark_results.json`: final seed=0 cross-policy comparison across easy, medium, hard, expert, recovery, and silent.
- `policy_seed_sweep_results.json`: deterministic multi-seed comparison that proves the ordering is not tied to one authored trace.
- `scenario_variant_catalog.json`: machine-readable event catalog showing how seed variants change warning, noise, phantom, recovery, and breach timing.
- `openenv_validation_results.json`: structural and live OpenEnv validation results captured from this tree.
- `hard_cautious_transcript.json`: full hard-task run from the cautious policy.
- `expert_cautious_transcript.json`: full expert-task run showing phantom-alert handling.
- `recovery_cautious_transcript.json`: full recovery-task run showing transient degradation handling.
- `hard_aggressive_grader.json`: single-step proof of the aggressive policy's watch-window mistake.
- `expert_phantom_hold_grader.json`: single-step proof that the expert phantom alert is not a rollback.
- `recovery_transient_hold_grader.json`: single-step proof that the recovery warning window rewards holding.
- `inference_stdout_fallback_sample.txt`: strict inference stdout contract sample.

## Score Interpretation

- `episode_score` means normalized running-average score across decisions taken so far.
- `total_score` in a grader artifact means a single decision score from `POST /grader`.
- `policy_benchmark_results.json` reports completed-episode scores for named policies across all public tasks.
- `policy_seed_sweep_results.json` reports the same policy comparison across multiple deterministic seeds and then averages them.
- `openenv_validation_results.json` records validator outcomes and should not be interpreted as a benchmark score artifact.
- The watch-window transcript is intentionally partial and `in_progress`, so its `episode_score` should not be compared directly with full benchmark averages.

## Files

- `artifact_index.md`: kind=reviewer_guide, policy=n/a, task=n/a. Human-readable starting point for reviewing the artifact pack.
- `artifact_manifest.json`: kind=manifest, policy=n/a, task=n/a. Machine-readable index describing every artifact and its score semantics.
- `inference_stdout_fallback_sample.txt`: kind=stdout_sample, policy=fallback_inference_policy, task=n/a. Shows the strict inference stdout contract using only [START], [STEP], and [END].
- `hard_cautious_transcript.json`: kind=transcript, policy=cautious_policy, task=hard. Full hard-task run from the cautious policy used in the benchmark comparison.
  Score note: This full-episode score should match the cautious_policy hard benchmark score (0.9350).
- `hard_cautious_grader.json`: kind=grader_payload, policy=cautious_policy, task=hard. Representative final rollback decision from the cautious hard-task run.
  Score note: This is a single-step grader score, not an episode score.
- `hard_aggressive_transcript.json`: kind=transcript, policy=aggressive_rollout_example, task=hard. Full hard-task aggressive rollout example that promotes through the hard watch window.
  Score note: This is a hard-specific illustrative rollout example. It is not the same object as the multi-task `aggressive_policy` comparator in the benchmark results.
- `hard_aggressive_grader.json`: kind=grader_payload, policy=aggressive_rollout_example, task=hard. Captures the aggressive rollout example's signature mistake: promotion during the hard watch window.
  Score note: This is the risky watch-window promotion step, not the full-episode score.
- `hard_watch_window_hold_transcript.json`: kind=transcript, policy=cautious_policy, task=hard. Shows the hard-task watch window after the hold decision has been taken.
  Score note: This is an in-progress episode score through the watch-window hold, not a completed benchmark score.
- `hard_watch_window_hold_grader.json`: kind=grader_payload, policy=cautious_policy, task=hard. Isolates the watch-window hold decision that makes the hard task discriminative.
  Score note: This is a single-step grader score for the hold decision on the hard watch window.
- `expert_cautious_transcript.json`: kind=transcript, policy=cautious_policy, task=expert. Full expert-task run showing that the cautious policy ignores the phantom alert and later rolls back on the real differential breach.
  Score note: This is a completed expert-task episode from the cautious policy.
- `expert_phantom_hold_grader.json`: kind=grader_payload, policy=cautious_policy, task=expert. Single-step proof that the phantom alert is rewarded when the agent verifies metrics instead of rolling back.
  Score note: This is the phantom-alert hold decision score, not a full-episode score.
- `recovery_cautious_transcript.json`: kind=transcript, policy=cautious_policy, task=recovery. Full recovery-task run showing that the cautious policy holds through a transient canary-only degradation and then continues the rollout safely.
  Score note: This is a completed recovery-task episode from the cautious policy.
- `recovery_transient_hold_grader.json`: kind=grader_payload, policy=cautious_policy, task=recovery. Single-step proof that a transient canary-only degradation is rewarded as a hold, not a rollback.
  Score note: This is the transient-recovery hold decision score, not a full-episode score.
- `policy_benchmark_results.json`: kind=benchmark_comparison, policy=multiple, task=easy,medium,hard,expert,recovery,silent. Compares shallow baseline, cautious policy, and aggressive policy across all tasks.
- `policy_seed_sweep_results.json`: kind=benchmark_comparison, policy=multiple, task=easy,medium,hard,expert,recovery,silent. Shows that the main policy ordering remains coherent across deterministic non-zero seeds, not just the canonical seed=0 profiles.
  Score note: Aggregate scores are the mean completed-episode scores across the documented benchmark seeds.
- `scenario_variant_catalog.json`: kind=variant_catalog, policy=deterministic_probe, task=easy,medium,hard,expert,recovery,silent. Enumerates event ordering and signal-shape differences across deterministic seeds so reviewers can inspect task-family breadth directly.
- `openenv_validation_results.json`: kind=validation_capture, policy=n/a, task=n/a. Captures structural OpenEnv validation and live local-server validation from the generated artifact run.
  Score note: These are validator compatibility proofs, not benchmark scores.
- `endpoint_contract_sample.json`: kind=endpoint_capture, policy=n/a, task=n/a. Shows representative validator-safe endpoints plus the recommended /episodes HTTP flow.
- `benchmark_audit_summary.md`: kind=summary, policy=n/a, task=n/a. Short benchmark-level summary of what the artifact pack proves.
