# R5 phase-adjoint verification and execution receipt

Status: verification receipt, CLOSED by user request, 2026-09-29. Decisions belong to [research/README](../../docs/research/README.md), [architecture](../../docs/research/architecture.md), [theory §11](../../docs/research/theory.md) and [roadmap](../../docs/research/roadmap.md).

## Mathematical and implementation scope

[algebra.json](algebra.json) comes from the independent [CPU script](../../experiments/verify_phase_initialization.py), which does not import the candidate. All 12 checks pass: baseline causal kernel and frame function, first-order real-history derivatives, global/frame functions and gradients, frozen-step spectrum, scaled-boundary and invalid three-phase counterexample. Coordinate output/gradient errors are at most 1.12e-16; local spectral error is 4.45e-16. Boundary singular values are sqrt(2), not 1.

[validation.json](validation.json) records 72 passing local contract checks and source digests. It includes phase/factory/control, source-topology AttnRes, dataset and reproducible-runner tests, plus R4 regression. Tests do perform tiny synthetic NTP optimization and interrupt/resume; this is implementation/integration evidence, not natural-language quality or a dedicated training comparison.

The main method and gated boundary-skip have matching initial prewriter gradients. Post-frozen matches the main initial gradient too; all-frozen has dead pre-routes and is explicitly a negative control. These facts rule out using a weak frozen arm to establish persistent-memory necessity.

[phase_rank.json](phase_rank.json) records 27 independent NumPy/float64 checks for the positive-scale serialized exact-reference contract, including embedding and a single terminal reader. Every strict reference-scale decrease requires a fresh orthogonal direction; reference Gram rank equals the number of distinct scales. This is a classical Gram/AR correlation characterization within that interface, not trained n3 LM evidence or a general memory bound. Approximate kernels and parallel readers are explicit counterexamples to broader claims.

The terminal alternative moves the sole switch after all body cells, retaining zero carrier, unit tied body and the same output router. It already opens every body writer's true-history gradient at initialization; internal phase must show added value beyond this smaller construction. The output router's own radial signal can be suppressed by final RMSNorm.

## Remote preparation

The user authorized code upload and data preparation on the no-card AutoDL instance, followed by Chrome GPU power-on and replacement rental if unavailable. Source is a Python execution snapshot containing lm/isohc import dependencies, not a Git checkout. [Remote CPU preflight](remote_validation.json) uses the actual remote Torch runtime; local and remote Torch versions differ, so optimizer checkpoints are not moved between them. The actual no-card cgroup memory limit was 2 GiB; a killed compiled test succeeded after limiting compilation parallelism, without an OOM-kill counter proving the initial cause.

After preparation, Chrome shutdown succeeded and the first GPU power-on failed because the source host had zero free cards. A clone was prepared but no new instance was created. After the user's further power-on authorization, the original host had a free GPU; original-instance power-on succeeded at the confirmed CNY 1.58/hour. Provider auto-shutdown is set to 2026-09-30 03:45 America/New_York (07:45 UTC), with a displayed remaining estimate of CNY 17.17 under the conservative first-round CNY 20 ceiling. These are execution controls, not scientific results.

## Actual GPU optimizer-step measurements

[gpu_profiles.json](gpu_profiles.json) records 16 independent-process eager profiles on the actual RTX 4080 SUPER with about 32 GiB usable capacity, Torch 2.12.1+cu130 and BF16 AMP. Effective batch is 32 in both micro8/accum4 and micro32/accum1; eight methods succeed in both. The first series measures 20 steady updates after 5 warmup updates, the second 50 after 10. Including warmup, this is 680 real FineWeb-Edu optimizer updates and 11,141,120 tokens. Profile models are discarded and are not quality-comparison runs.

At micro32/accum1, baseline/phase steady throughput is about 131.5k/86.0k tokens/s, with allocated peaks 12.65/15.34 GiB. The PyTorch Block AttnRes reference measures 41.5k tokens/s and 22.85 GiB allocated (27.67 GiB reserved). This does not estimate the original production fused implementation. Three additional default-Inductor measurements reach about 201.8k/170.2k/119.2k tokens/s for baseline/phase/Block. Total profiles include 860 optimizer updates and 14,090,240 NTP tokens. The default precision policy was subsequently superseded; corrected-policy throughput must be measured separately. Peaks begin after warmup.

[gpu_preflight.json](gpu_preflight.json) preserves six default-compiled one-update runs with all 3,906 validation blocks, optimizer checkpoint and matching timing sidecar. [cuda_initialization.json](cuda_initialization.json) preserves four zero-update numerical audits and their actual source identities. Eager baseline/phase initial logits are identical. Cast emulation plus backward AMP-off also makes compiled training logits identical, while compiled no-grad eval still differs (max logit difference 0.255859 in the explicit-off audit). Shared BF16 gradients differ at about the native repeated-backward noise scale. This does not establish post-update trajectory equivalence.

Canonical scientific validation therefore uses the raw model's eager BF16 evaluator. Compiled training explicitly emulates precision casts and specifies backward autocast off, matching the actual backward context. Policy and evaluator identity are locked into resume. Old default-policy checkpoints and their apparent one-update NLL differences do not seed or establish an advantage in the new quality study.

[canonical_gpu_preflight.json](canonical_gpu_preflight.json) subsequently completes seven corrected-policy profiles, seven full-validation/save preflights and two actual CUDA resumes. Baseline/phase/terminal steady throughput is about 203.0k/172.0k/171.3k tokens/s; phase/terminal both allocate 10.36 GiB at peak. Total profiling across all policies is 1,280 optimizer updates/20,971,520 NTP tokens, with weights discarded. The baseline/phase one-update canonical held-out NLL difference is about 5.16e-6 nat; preflight is not a quality study.

The separate study driver has reviewed sequential pilot/LR-selection/continuation and deadline contracts. Independent review found two recovery faults before launch: reselection after a primary had begun, and continuing scheduling while a child exit was unconfirmed. The first planning-only version is preserved in [study_plan_v1_superseded.json](study_plan_v1_superseded.json); no quality run used it. Models, runner and GPU measurements were unaffected. The fixed driver passes 15 local and 15 remote mock contracts, and independent review confirms the two counterexamples are blocked.

The reviewed [study_plan.json](study_plan.json), driver SHA `4d8aa667de48e1d5e52e8c6ce02a6fd48d411054e06520f444d8464540c4b8c7`, launched as controller PID8840 at `/root/autodl-tmp/isoHC/r5/study/canonical7-r2`. Seven methods receive two 2,048-update pilots; selected prefixes continue on the unchanged 16,384-update schedule to 268,435,456 tokens. Learning-rate selections freeze once any primary starts. Rates are the canonical measurements discounted by20%, with300s per-stage overhead and1,200s reserve before UTC07:15. Provider shutdown remains UTC07:45. This single-seed development study was stopped before any full-budget outcome; the old plan is not an active execution directive.

Dataset target: official FineWeb-Edu sample/10BT, fixed revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9; train 300M and validation 2M GPT-2 tokens. Identical exact UTF-8 documents stay in one split. A complete manifest now exists; [data_identity.json](data_identity.json) checks the exact counts, tokenizer and consumed raw-file digests against the official pinned LFS metadata. Identity checks do not establish scientific data quality. Direct HF access failed; mirror pagination redirected to the blocked primary site. The preparation path now uses the pinned metadata's siblings or only the sample/10BT directory. Tokenizer assets were checked against the package's expected SHA before upload.

## Permitted conclusions

The reciprocal phase construction has a valid baseline-exact, cross-cut first-order history contract and runnable controls. No LM advantage, whole-network stability, independent novelty, GPU speedup or main-conference acceptance follows from these checks. Later GPU/data/training receipts must state their actual scope and source identity separately.

## Closeout

[closeout.json](closeout.json) records verified own-process interruption and Chrome-confirmed instance shutdown at23:51 UTC. [pilot_snapshot](pilot_snapshot/) preserves seven completed pilot records, eight training logs including the interrupted terminal6e-4, the fixed plan/status and execution stop logs. No primary268M-token run completed and the seven-arm selection was not finalized.

[closeout_analysis.json](closeout_analysis.json) independently validates all seven completed pilot identities and endpoints. Both tested LRs favor gain over phase; same-LR phase-minus-gain NLL is+0.004512/+0.020337 nat. Canonical profile phase also costs more. This justifies stopping this round's investment, while preserving the valid math and implementation assets. It is not a mature-model or method-family impossibility claim.

The requested [theory/result closeout report](../../docs/research/reports/R5_CLOSEOUT_20260929.md) includes mechanism limits, prior work, measured costs and the process mistake of delaying the cheapest decisive control. No automatic resume or scaling is authorized. Data and completed model checkpoints remain on the powered-off instance; Git receives metadata/results/code rather than large model weights or token data.
