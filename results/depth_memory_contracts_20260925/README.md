# Depth-memory reference contracts — 2026-09-25

Status: **offline mathematical/prototype validation, 11/11 checks passed**.
No LM training, model integration, GPU run, download, learned controller, or task
benchmark was performed. These results support implementation contracts and
counterexamples; they do not establish novelty, trainability, or SOTA quality.

Sources: [reference script](../../experiments/verify_depth_memory_contracts.py),
[machine-readable receipt](contracts.json). Reproduce from the repository root:

```sh
python3 experiments/verify_depth_memory_contracts.py
```

The recorded run uses seed `20260925`, CPU, `torch.float64`, Python `3.9.6`, and
PyTorch `2.8.0`. Numeric tolerance is `1e-12` where required. Protected rows are
checked for exact equality.

## Implemented contract

The functional reference stores `K` value slots of width `d`, separate keys,
valid bits, integer versions, and lease-expiry metadata. At depth `t`, a valid slot is protected
when `lease_until > t`; expiry therefore permits reuse before the write at the
expiry depth. A current proposal is projected into the Euclidean radius-`R`
ball, giving `v`. At most one unprotected slot is updated by

`M'_j = (1 − η) M_j + η v`, with `0 ≤ η ≤ 1`.

An invalid destination contributes a zero old value even if its backing storage
is nonzero; a partial first write therefore produces `ηv`. Commit expiry must
be at least `t+1`, corresponding to `t+1+ℓ` for a nonnegative lease duration `ℓ`.
The reference directly receives this expiry; it does not predict it.

Every other row is copied exactly. A successful commit updates value, key,
validity, and expiry together and increments that slot's integer version.
Empty slots are preferred; otherwise a caller's
current eviction-risk scores select among unprotected rows. Expired content
remains valid and readable until overwritten. A rejected commit leaves all
persistent state unchanged, while the current workspace still receives `v`.
Zero write weight is treated as a rejected commit. Rejection does not change
versions, keys, or any other persistent metadata.

Keys in this reference are caller-supplied address metadata. Atomic key/value
updates do not prove that keys semantically identify their values. The reference
does not implement learned read attention, lease prediction, task loss, or a
Transformer integration.

## Recorded findings

| Check | Recorded result | Claim boundary |
|---|---|---|
| Protected content | Bitwise unchanged, including random long-run protected sets | Only leased coordinate rows are protected |
| Atomic commit | One value row and its metadata change together; version increments; invalid backing storage contributes zero | No learned addressing validation |
| Bounded state | 4,096 steps; maximum row/workspace norm `1.7000000000000006` at `R=1.7`; 3,641 commits, 455 rejections | State bound assumes initially bounded memory and bounded proposal |
| Full-capacity behavior | Persistent write rejected; workspace receives current proposal | A usable workspace output is not proof that an entire model avoids stalls |
| Lease expiry | Expired slot reused; still-live slot preserved | External expiry decision, not a learned lifetime result |
| Frozen-control transport | Exact diagonal Jacobian, operator norm `1` | Proposal, selection, gate, and lease controls are held fixed |
| Full Jacobian counterexample | Derivative `4.5` at zero for a bounded convex recurrence | Bounded state does not imply gradient stability |
| Protected-view counterexample | Spectral radius `1`, operator norm `3.3028`, 16-step gain `48.0104` | Preserving a linear read alone does not control nonnormal growth |
| Causal prefix | Identical states/workspaces for seven shared input steps after changing five future proposals | Reference-depth causality only; Transformer token causality remains to be verified |
| External-lease capacity | 64 writes retained while live using 3 slots; 2 slots reject 16 writes | Toy lifetime trace with supplied leases, not a compression or task-performance result |
| Predicted-risk selection | Sampled maximum regret `0.12177`, below `2ε=0.15` | Uniform error bound `ε=0.075` is assumed, not estimated from data |

## Why the mathematical scope is narrow

The radius bound follows from the triangle inequality:
`‖(1−η)M_j+ηv‖ ≤ (1−η)R+ηR = R`.
For fixed proposal and controls, the value carry is diagonal across slots:
its entries are `1` except `1−η` at the selected slot. Its Euclidean operator
norm is at most one. An all-protected rejection has identity carry. Neither
statement bounds derivatives through a learned proposal, router, or gate.
For example, `x' = 0.5x + 0.5 tanh(8x)` preserves `[-1,1]` and has derivative
`4.5` at zero. Controller/branch Jacobians and optimization remain empirical
and theoretical work.

Protecting arbitrary linear views without the convex slot constraint does not
provide the same bound. With protected row `pᵀ=(1,0)` and
`T=[[1,0],[3,1]]`, `pᵀT=pᵀ`, yet `‖T‖₂>3` and repeated transport grows.
This motivates the coordinate-slot prototype; it does not prove all projected
or nullspace architectures are unstable.

For indivisible versions with prescribed half-open lifetimes, at least the
maximum number of simultaneously live versions is needed for exact retention
without compression. That capacity also suffices: on every write, some slot is
free or expired unless the new interval would exceed that maximum. The trace
instantiates this elementary allocation result. It does not show a model can
predict lifetimes, compress semantic content, or improve downstream quality.
Future-informed/oracle lifetimes are permitted only as labeled toy bounds;
they cannot enter a causal model controller or a headline task score.

If all candidate eviction risks satisfy `|r̂_j−r_j|≤ε`, choosing minimum
predicted risk gives regret at most `2ε` relative to the minimum true risk.
The check samples bounded errors. Learning a useful risk estimator and
establishing calibration on held-out interventions remain unperformed.

## Evidence ownership

This directory is the canonical receipt for the reference contracts. Cite it
as **numerical contract validation**, never as a completed model experiment.
The active [research entry](../../docs/research/README.md) owns the research
decision and experimental plan. Historical gauge checks and archived LM runs
remain separate evidence with their original scopes.
