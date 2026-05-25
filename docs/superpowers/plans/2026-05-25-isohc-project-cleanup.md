# IsoHC Project Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove misleading legacy experiment entrypoints, keep only the current IsoHC mechanism pipeline, and document the safe way to run experiments.

**Architecture:** The repository should have one canonical 5090 LM runner, one GPU pipeline shell script, one posthoc analysis script, and historical results only under `docs/0605_alldoc`. Spectral/SVD baselines are removed from training code so the main method remains Newton-Schulz IsoHC.

**Tech Stack:** Python 3.12, PyTorch 2.8, bash, unittest.

---

### Task 1: Remove Misleading Training Baselines

**Files:**
- Modify: `lm/mixing.py`
- Modify: `experiments/lm_5090_next_runs.py`
- Modify: `tests/test_lm_next_phase_contracts.py`

- [x] **Step 1: Remove `spectral` and `fixed-vector-spectral` training mixers.**

Delete the spectral classes and factory branches from `lm/mixing.py`; keep SVD only in diagnostics/posthoc code, not in the training-time mixer family.

- [x] **Step 2: Remove `spectral-hc` methods from the 5090 runner.**

`experiments/lm_5090_next_runs.py` should accept only current HC training methods: `identity-hc`, `unconstrained`, `mhc`, `isohc`, `orthogonal`.

- [x] **Step 3: Update tests.**

Remove spectral baseline tests and add a guard that `spectral-hc` is rejected by `create_model`.

### Task 2: Delete Legacy Entrypoints That Cause Wrong Runs

**Files:**
- Delete: `experiments/run_0525_fe_fair_deep.sh`
- Delete: `experiments/run_0525_fe_overnight_pipeline.sh`
- Delete: `experiments/lm_phase0_smoke.py`
- Delete: `experiments/lm_phase1_controlled.py`
- Delete: `experiments/lm_verify.py`
- Delete: `experiments/stage2_real_text_smoke.py`
- Delete: `experiments/stage2_real_text_grid.py`

- [x] **Step 1: Delete old FE shell scripts.**

These run the old matrix and no-checkpoint path. The replacement is `experiments/run_0525_mechanism_gpu_pipeline.sh`.

- [x] **Step 2: Delete early LM smoke/phase scripts.**

These predate the current `TwoBranchHCTransformer` FE plan and should not be used for paper evidence.

- [x] **Step 3: Delete old tiny real-text grid scripts.**

They targeted TinyShakespeare/PPL-style checks and were part of the metric mismatch.

### Task 3: Add Canonical Run Documentation

**Files:**
- Create: `experiments/README.md`
- Modify: `README.md`
- Modify: `agent.md`
- Modify: `docs/0605_alldoc/0525_teacher_followup_plan.md`

- [x] **Step 1: Document the only supported FE command.**

The canonical command is `bash experiments/run_0525_mechanism_gpu_pipeline.sh` with `METHODS="identity-hc mhc isohc"` or the targeted `METHODS="mhc isohc"`.

- [x] **Step 2: Document server/data-disk rules.**

All data and caches live under `/root/autodl-tmp/isoHC`; never use live HuggingFace during training.

- [x] **Step 3: Mark spectral baseline as removed.**

If needed later, it must be reintroduced as a separate appendix script after implementing a cheap approximation, not mixed into the main runner.

### Task 4: Verify

**Files:**
- Test: `tests/test_lm_next_phase_contracts.py`
- Test: `tests/test_stage1_contracts.py`

- [x] **Step 1: Run syntax check.**

Run:

```bash
python3 -m py_compile lm/*.py experiments/*.py
```

- [x] **Step 2: Run contract tests.**

Run:

```bash
python3 -m unittest tests.test_lm_next_phase_contracts tests.test_stage1_contracts -q
```

- [x] **Step 3: Check remote run status.**

Run:

```bash
ssh -p 51198 root@connect.westd.seetacloud.com 'pgrep -af "lm_5090_next_runs|run_0525_mechanism"; nvidia-smi'
```
