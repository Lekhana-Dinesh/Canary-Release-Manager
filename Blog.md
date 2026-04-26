# We tried to teach an LLM to manage canary deployments. Here is what happened.

*Aparna Singha & Lekhana Dinesh · Scaler School of Technology, Bangalore · Meta OpenEnv Hackathon Finals 2026*

This is not a polished write-up. We are writing this at the end of two very long days in Bangalore, running on coffee and whatever sleep we could get between training runs. But we want to be honest about what we built, why it was hard, and what we actually proved.

## Why we picked this problem

The idea came from a simple observation. Every week, ML engineers at real companies have to decide whether a new model is safe to roll out to more users. They look at metrics — error rates, latency, alert counts — and make a judgment call: keep going, wait, or roll back.

It sounds mechanical. It is not. The hard part is the ambiguous states. Alerts fire but the raw metrics look fine — is it a false positive or a real problem? Error rate drifts slightly upward — is this the beginning of a regression or just noise? Both stable and canary degrade at the same time — is the canary broken, or is the whole infrastructure having a bad day?

A threshold-checking script cannot answer these questions. Neither can an LLM that has only ever read about deployments. You need something that has actually *made decisions* in an environment and learned from the consequences.

That is what we wanted to build.

## What we built

We built an RL environment called **Canary Release Manager**. At every step, the model sees real-looking telemetry — canary error rate, stable error rate, p99 latency, differential metrics, alert count, rollout thresholds — and has to decide: increase traffic by 5%, 10%, or 25%, hold, or roll back. The final 7B v2 run used `unsloth/Qwen2.5-7B-Instruct` with SFT warm-start + GRPO.

We designed six task families. Each one is built around a failure mode that we have seen described in real SRE post-mortems.

**easy** — the canary is fine, then suddenly terrible. An obvious regression. Any reasonable agent should catch it.  
**medium** — the canary drifts slowly before it actually breaches. Overconfident promotion gets punished.  
**hard** — at 30% traffic, both stable and canary latency spike together. Shared infrastructure noise. The correct action is hold, not rollback. Rolling back because you saw high latency is wrong here.  
**expert** — four alerts fire at once. But the raw metrics are healthy. It is a phantom alert storm. Rolling back on alert count alone is explicitly penalized. Then, later, a real drift builds and the agent has to catch it.  
**recovery** — the canary spikes but stays below the rollback guardrail and then recovers on its own. Panic rollbacks lose score.  
**silent** — a real regression builds in the differential metrics with almost no alert signal. The agent has to read the raw telemetry, not wait for an alarm to tell it something is wrong.

The reward function has four components — breach detection, rollback timing, promotion safety, reasoning quality — specifically so that no single shortcut can game the score. An agent that always rolls back gets destroyed on phantom alert and recovery tasks. An agent that always promotes gets destroyed on everything else.

## The training was harder than we expected

We are not going to pretend this went smoothly. It did not.

The first several training runs produced reward of exactly 0.000 on every step. Every single one. We spent a long time debugging before we understood what was happening: the model was generating text that looked like JSON but was not actually valid JSON with all the required fields. The parser returned failure. The environment was never called. The reward was always zero. GRPO had nothing to learn from. That failure mode is part of why we ended up treating output format and parse reliability as a first-class part of the project.

The fix was to add an SFT warm-start first — train the model on teacher-demonstrated rollout decisions so it learns the output format before we ask GRPO to improve the policy. In the final v2 run, SFT was enabled, GRPO was enabled, and the run reported `parse_ok_rate = 1.0`, `reward_error_rate = 0.0`, and a diagnostic verdict of `PASS`.

We also had API field name mismatches, server startup issues in Colab, version conflicts between TRL and transformers, and about a dozen other things that each cost us an hour. This is the reality of building RL pipelines under time pressure. The final 7B v2 run itself used 80 global GRPO steps, 80 reward calls, and 2109 environment calls on an A100.

## What the numbers actually show

We evaluated on heldout seeds — task instances the model never saw during training. Here is what happened in the final 7B v2 run:

| Task | Before | After GRPO | Change |
|---|---:|---:|---:|
| easy | 0.0500 | 0.6800 | +0.6300 |
| medium | 0.0500 | 0.7064 | +0.6564 |
| hard | 0.7298 | 0.6358 | -0.0939 |
| expert | 0.2907 | 0.6461 | +0.3554 |
| recovery | 0.4605 | 0.7282 | +0.2677 |
| silent | 0.4931 | 0.5353 | +0.0422 |
| **average** | **0.3457** | **0.6553** | **+0.3096** |

These are the heldout scores from the final run. Generalization matched the heldout average exactly at **0.6553**, and the stress average was **0.6544**. 

So the honest summary is: the model improved a lot overall, but not uniformly. Easy, medium, expert, recovery, and silent improved. Hard dropped slightly from an unusually strong pre-training score, which tells us the learned policy is still not perfect and still has tradeoffs.

It is also important that RL actually added something on top of SFT. The average score moved from **0.6105 after SFT** to **0.6553 after GRPO**, so the RL stage did not just preserve the warm-start — it improved it.

## Training curves

### Reward during GRPO
![Reward Curve](https://huggingface.co/aparnasingha400/canary-7b-job-output-v2/resolve/main/evidence/reward_curve.png)

This plot shows verifier-backed environment reward during GRPO.


The reward curve is noisy, but the EMA trends upward and ends much higher than where it started. Parse success stayed at 100% through the run. 

### Per-task reward during GRPO

![Per Task Reward](https://huggingface.co/aparnasingha400/canary-7b-job-output-v2/resolve/main/evidence/per_task_reward.png)

This plot helps verify that reward behavior is not isolated to one easy task family.

During GRPO, expert episodes had the highest mean reward, while silent stayed among the weaker categories. That matches the heldout evaluation: some behaviors improved a lot, but the model is still not equally strong everywhere.

### GRPO loss curve

![Loss Curve](https://huggingface.co/aparnasingha400/canary-7b-job-output-v2/resolve/main/evidence/loss_curve.png)

This plot shows the training loss from the final GRPO run.

The loss stayed low and fairly stable through 80 steps, ending at **0.00210**. The full run took about **460 seconds**.

### Before / after / generalization comparison

![Before vs After Scores](https://huggingface.co/aparnasingha400/canary-7b-job-output-v2/resolve/main/evidence/before_after_scores.png)

This plot compares model performance before training, after SFT, after GRPO, and on generalization/stress evaluations.

This is probably the simplest summary of the project. The trained policy is clearly above the untrained model overall, but still below the shallow and cautious hand-coded references on most tasks. That is exactly why we describe this as a meaningful step, not a solved problem.

### Action distribution during RL

![Action Distribution](https://huggingface.co/aparnasingha400/canary-7b-job-output-v2/resolve/main/evidence/action_distribution.png)

The learned policy uses multiple rollout actions during training telemetry, including `increase_25`, `increase_10`, `rollback`, and `hold`.

The final action distribution during RL was:

- `increase_25`: 155
- `increase_10`: 118
- `rollback`: 28
- `hold`: 19

So the policy is no longer collapsed into a single action, but it still leans promotive. That is one of the limitations we want to be honest about. 

## What actually changed in behavior

The clearest thing we can say is not that the model became production-ready. It did not.

What changed is that the model started behaving more like a policy trained inside an environment, instead of a generic instruct model guessing from text. The final run no longer shows the severe action collapse that appeared in the weaker earlier 7B attempt, and its diagnostic verdict moved from `WARN` to `PASS`.

The final learned policy still prefers promotive actions, but it now uses `increase_25`, `increase_10`, `rollback`, and `hold` rather than almost always doing the same thing.

That matters because rollout management is not just about knowing definitions. It is about acting under uncertainty and getting rewarded or punished for the consequences.

## What the trace taught us

The saved expert trace for `seed=950` is a good example of why we are being careful with our claims.

Its total episode score was **0.4975**. Early in the episode, the model produced several parse failures and labeled healthy low-drift states as `breach`, even while promoting traffic. Later, when the canary actually started drifting, it switched to repeated `hold` decisions with a reasonable explanation: *“Canary is drifting toward guardrails; holding to confirm.”*

So the trace shows both sides of the story:

- there is real learned behavior in the later warning phase
- but the controller is still not fully reliable end to end

That is why we are comfortable saying this is a successful environment-grounded RL run, but not a polished deployment controller.

## Why we think this matters

There is a version of this project that is easy to dismiss: it is a hackathon-scale model adapter, a short training run, and a simplified environment. All of that is true.

But the question we were actually trying to answer is not “did we solve deployment management.” It is: *can rollout decision behavior be trained, measured, and improved inside an environment?*

For this project, the answer is yes.

The final 7B v2 run improved from **0.3457 before training** to **0.6553 after GRPO**, with a clean `PASS` diagnostic, zero reward-pipeline errors, and stable heldout/generalization averages.

That matters because most professional tasks are not best framed as question-answering. They are decision loops. Observe state, take action, get feedback, improve. The infrastructure for training LLMs on those loops exists now — OpenEnv is part of it. What is missing is environments that capture real decision problems worth training on.

We tried to build one of those. We think we got somewhere.

## Everything we built

- [Environment repository (GitHub)](https://github.com/Lekhana-Dinesh/Canary-Release-Manager)
- [Live environment (Hugging Face Space)](https://huggingface.co/spaces/aparnasingha400/canary-release-manager)
- [Trained 7B model + full results](https://huggingface.co/aparnasingha400/canary-7b-job-output-v2)
- [Training notebook (Colab)](https://colab.research.google.com/drive/1-MZbE9wimHLo8bIBpGwaKg1Or2IFdwg3?usp=sharing)
- [Before/after demo notebook (Colab)](https://colab.research.google.com/drive/1yX2LCPFhLEgyFzINZF_XR9JPPSKitzVV?usp=sharing)

*Written at the end of the hackathon. We are tired. But we are proud of what we built.*
