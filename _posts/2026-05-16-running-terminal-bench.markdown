---
layout: post
title: "Running Terminal-Bench: what actually happened"
date: 2026-05-18 22:00:00 -0400
categories: benchmarks agents
---

Terminal-Bench is a benchmark for evaluating LLM agents on realistic terminal tasks --
things like completing a Coq proof, cracking a password-protected archive, or recovering
lost git changes. The paper (Merrill et al., 2026, arXiv:2601.11868) came out in January
and the benchmark is pretty new, so I wanted to get hands-on with it.

This post is mostly a log of what I tried and what I ran into, not a proper evaluation.
Sample sizes are small and I ran into enough confounds that I would not read much into
specific numbers.

## Setup

More involved than expected on Windows. Terminal-Bench assumes a Unix environment, so
running it from PowerShell fails -- the harness calls `rm -rf` at one point and crashes.
The fix is WSL2 with a proper Ubuntu install (not the Docker Desktop internal container,
which is a minimal shell with no package manager). Docker also needs its WSL integration
enabled and your user added to the docker group.

Python version matters too -- the CLI uses type annotation syntax that breaks on 3.14,
so 3.12 is needed. This is not mentioned in the README.

API access had its own issues. The Princeton AI Sandbox is not reachable off-campus
even with VPN. Ended up using OpenAI with $10 of personal credit, which works but
limits you to gpt-4o-mini on a new account due to tier restrictions. Used a personal
Anthropic key for some runs with Claude Sonnet.

## Task structure

Each Terminal-Bench task consists of an instruction, a Dockerfile that sets up the
environment, a test script that verifies the outcome, and an oracle solution written
by the task author. The benchmark is outcome-driven -- the agent is free to solve
the task however it wants, and tests only check the final container state. Tasks span
software engineering, system administration, security, scientific computing, and
several other categories.

The version I ran against is `terminal-bench-core` 0.1.1, the beta release tied to
the public leaderboard at tbench.ai.

## Results

I ran four batches across two models, 17 trials total, 12 distinct tasks. Overall:
7 resolved, 10 failed.

| Task | gpt-4o-mini | claude-sonnet-4-5 | Notes |
|------|-------------|-------------------|-------|
| `hello-world` | pass | -- | |
| `fix-git` | partial | partial | 3 runs, same failure each time |
| `prove-plus-comm` | fail | pass | |
| `csv-to-parquet` | pass | pass | |
| `fix-pandas-version` | pass | -- | |
| `openssl-selfsigned-cert` | timeout | pass | |
| `path-tracing` | timeout | -- | Hard |
| `gpt2-codegolf` | timeout | -- | Hard |
| `crack-7z-hash` | timeout | pass | |
| `sqlite-db-truncate` | -- | fail | |
| `conda-env-conflict-resolution` | -- | fail | |

Token usage across the first batch gives some texture to what "failing" looks like
in practice:

| Task | Result | Input tokens | Output tokens |
|------|--------|-------------|---------------|
| `hello-world` | pass | ~800 | 128 |
| `fix-git` | partial | 38,342 | 1,853 |
| `prove-plus-comm` | fail | 473,181 | 7,838 |

The paper (Appendix G.2) reports essentially no correlation between token count and
success rate across thousands of trials. This small sample is at least consistent with
that -- the task that used the most tokens failed, the one that used the fewest passed.

## Difficulty in practice

The tasks that resolved across both models share a common structure: there is a short,
well-defined sequence of commands that solves them and the agent does not need to
iterate much. Tasks requiring domain knowledge or multi-step reasoning showed more
variation across models.

The paper (Section 4.3) reports a positive but imperfect correlation (r=0.436) between
human-predicted difficulty and empirical difficulty, and notes that 54.5% of
human-rated medium tasks are empirically hard for models. `prove-plus-comm` is labeled
Easy in the dataset -- it is a Coq proof that an expert would finish in under an hour
-- but gpt-4o-mini failed it after 473K input tokens while Claude Sonnet solved it in
under 90 seconds. Whether that reflects a labeling issue or just a large capability gap
between these two models on Coq-specific knowledge is unclear from one trial each.

## The debug tool

Terminal-Bench includes an LLM-powered debug tool (Appendix B.3 of the paper) that
analyzes failed trajectories and checks whether the task instructions were sufficiently
specified. I ran it on two tasks:

```bash
tb tasks debug prove-plus-comm \
    --run-id 2026-05-17__20-31-13 \
    --model gpt-4o-mini \
    --tasks-dir ~/.cache/terminal-bench/terminal-bench-core/0.1.1
```

For `prove-plus-comm` it returned **FAIL** -- the agent had used Coq's `admit` tactic,
which skips a proof step and produces a file that compiles but contains an unproven
axiom. The instructions say "complete the proof" and "compile using coqc" but do not
say "do not use admit."

For `fix-git` it returned **PASS** -- instructions are sufficient, the failure is due
to incomplete execution. The tool noted that a competent agent should infer the need
to check `git reflog` from the task description.

The contrast is interesting. `prove-plus-comm` was flagged as underspecified but
eventually solved by a stronger model. `fix-git` was flagged as well-specified but
failed identically across three runs and two models. Specification sufficiency and
empirical solvability do not appear to be the same thing, at least in these cases,
though the sample is too small to say more than that.

## The fix-git partial failure

`fix-git` asks the agent to recover lost git changes after checking out master. It
failed identically in all three runs -- `test_layout_file` passed, `test_about_file`
failed -- regardless of whether the model was gpt-4o-mini or Claude Sonnet. Token
usage differed substantially (38K input tokens for gpt-4o-mini, 19K for Sonnet) but
both arrived at the same partial state.

Reading the oracle solution: the lost changes included modifications on a detached HEAD
state recoverable via `git reflog`. Both models found the branch-based changes but
missed the reflog. This looks consistent with the paper's Context Loss failure mode
(Appendix C) -- the agent did not maintain awareness of all the places uncommitted
work can live in a git repository. Whether a stronger model would catch this, or whether
there is something about how the task environment is set up that makes reflog harder to
find, would need more trials to sort out.

## Model comparison

The clearest signal across these runs is that the same tasks failed with gpt-4o-mini
and passed with Claude Sonnet -- not because the agent scaffold changed, but because
the model did. The paper (Section 4.1) reports the same thing at scale: "model
selection is usually more important than agent scaffold when optimizing for
performance." The Sonnet run on `prove-plus-comm` is probably the sharpest example
here -- same task, same agent, different model, very different result.

The caveat is that the Batch 2 timeouts under gpt-4o-mini happened while running 6
concurrent tasks on a Tier 1 account, which introduced rate limiting that slowed turn
frequency. Some of those timeouts might have been avoidable with fewer concurrent tasks.
Running `openssl-selfsigned-cert` in isolation under gpt-4o-mini would be a cleaner
test of whether it is a genuine model-dependent failure.

## Open questions

A few things worth following up on with more runs:

- Is the `fix-git` partial failure genuinely model-independent, or would a stronger
  model (e.g. Claude Opus) recover the reflog changes?
- Does making `prove-plus-comm` more explicit (adding "do not use admit") change
  gpt-4o-mini's result, or is it a pure capability gap?
- What do `sqlite-db-truncate` and `conda-env-conflict-resolution` look like in the
  debug tool -- specification issues or capability gaps?

More trials and fewer confounds would be needed before drawing stronger conclusions
from any of this.