# Offline theory contract audit — 2026-09-25

This directory contains new CPU-only verification, not training results. No dataset, checkpoint, GPU, network access, or external dependency installation was needed. The script uses the actual `TwoBranchHCTransformer` and repository projection implementation for logits checks, and explicitly labeled algebraic recurrences for the general contracts and counterexamples.

- [Reproduction script](../../experiments/verify_gauge_contracts.py)
- [Primary receipt, seed 20260925](gauge_contracts.json)
- [Independent numerical example, seed 7](gauge_contracts_seed7.json)

Run from the repository root:

```bash
python3 experiments/verify_gauge_contracts.py
python3 experiments/verify_gauge_contracts.py --seed 7 --output results/theory_audit_20260925/gauge_contracts_seed7.json
```

Both runs pass all 11 checks with Python 3.9.6, PyTorch 2.8.0, CPU, and float64 arithmetic. Seeds here generate numerical examples; they are **not independent training seeds**.

## Verified contracts

Use the recurrence

\[
X_{k+1}=H_kX_k+b_k F_k(a_k^T X_k/n),\qquad y=c^T X_L/n.
\]

For invertible, input-independent matrices, let \(G_0=I\), \(G_{k+1}=H_kG_k\), and \(X_k=G_kZ_k\). Then transport becomes identity, the read vector becomes \(G_k^Ta_k\), the write vector becomes \(G_{k+1}^{-1}b_k\), and the exit becomes \(G_L^Tc\). More generally, arbitrary invertible \(G_k\) require \(H'_k=G_{k+1}^{-1}H_kG_k\) and the matching input-boundary transform. This is a coordinate statement. Whether the transformed parameters remain in a particular model class is a separate condition.

| Check | Primary receipt result | Interpretation |
|---|---:|---|
| Actual model logits, default projection | max difference 1.85e-9 | Gauge absorption works up to numerical mean-constraint errors. |
| Actual model logits, explicit float64 basis + SVD | max difference 5.55e-17 | Exact IsoHC contract remains in the fixed-sum read/write/exit class. |
| General invertible transport, unrestricted vectors | max difference 2.50e-16 | Algebraic nonlinear recurrence with free read/write/exit vectors. |
| Arbitrary gauge including input/exit boundaries | max difference 1.11e-15 | Boundary changes are part of the equivalence. |
| Depth kernel under both gauges | max difference at most 6.67e-16 | Read–transport–write coefficients are invariant. |
| Fixed mean-preserving transport, coupling scaling | normalized correction difference 2.07e-12 | With fixed centered read/write vectors, correction scales as \(\lambda^2\). |

Simply substituting identity into the actual model **without** transforming gates changes logits by 0.00309 for the default-projection example and 0.0108 for the exact-projection example. Gauge equivalence must therefore be tested with its parameter transformation; it is not a statement that a trained checkpoint is unchanged by replacing its mixer alone.

The exact-projection example preserves read/write squared norms to 6.22e-15. The default projection allows orthogonality error up to its configured fallback threshold: this example has error 4.39e-4 and squared-norm change 5.46e-4. Accordingly, exact norm preservation belongs to the mathematical IsoHC constraint, not every finite Newton–Schulz output. Existing model code was not modified.

## Counterexamples that restrict the narrative

1. **Full orthogonal transport can leave the repository parameter class.** With \(H=\operatorname{diag}(1,-1)\), fixed shared input and fixed mean exit, the output differs by 1 if the exit is left unchanged. The correct transformed exit has sum 0 rather than the class-required sum 2. This establishes a boundary/class mismatch, not a universal expressive advantage for orthogonal mixing.
2. **Singular transport cannot always be gauged to identity.** Invertible changes of state coordinates preserve rank. A rank-one two-stream transport remains rank one after arbitrary invertible left/right coordinate changes. This does not rule out coincident input-output functions in special singular examples.
3. **Mean preservation does not imply a one-state depth kernel.** For two streams, \(H=I\), \(u=(1,-1)^T/\sqrt2\), \(a=\mathbf1+\sqrt2\alpha u\), and \(b=\mathbf1+\sqrt2\beta u\), the cross-cut kernel is \(1+\alpha\beta\). Two past writes with \(\beta=(0,1)\) and two future reads with \(\alpha=(0,1)\) yield \(\left[\begin{smallmatrix}1&1\\1&2\end{smallmatrix}\right]\), which has rank 2 despite no mean–complement exchange. Read/write access to the complement can create additional state dependence.
4. **A product of projected steps omits exchange return paths.** A 90-degree mean–complement rotation has \(U^THU=0\), while two rotations give \(U^TH^2U=-1\). Thus \((U^THU)^2=0\) is not the complete two-step complement transfer. Any diagnostic that multiplies projected steps measures a path with projection after every update, not the full path when mean–complement exchange is allowed.

For the coupling check, both centered vectors and the mean-preserving transport are held fixed. It verifies the exact depth-kernel coefficient \(K_{ij}=1+\lambda_a\lambda_b u_i^TT_{ij}v_j/n\). It does not bound the trained vectors, remove initial stream offsets, or imply a particular loss or complement-ablation magnitude.

## Weight-decay counterfactual from local run records

The script reads each original run's configuration and actual optimizer-step count, then uses the currently tracked `cosine_lr_schedule`. It replays the multiplicative factor \(q=\prod_t(1-\eta_t w)\) on symmetric mixer logits \(4I\), omitting initialization noise and task gradients. For four streams, the predicted complement singular value is

\[
\sigma_\perp=\frac{e^{4q}-1}{e^{4q}+3}.
\]

| Local run | Steps | WD-only prediction | Recorded mean single-step spectrum | Absolute difference |
|---|---:|---:|---:|---:|
| 48-layer FineWeb-Edu fair run | 1954 | 0.92178339 | 0.92220568 | 0.00042229 |
| 24-layer deep-stress run | 3256 | 0.91555435 | 0.91562865 | 0.00007430 |

These are an independently reconstructed counterfactual, not Claude's missing source scripts. Their proximity shows that initialization plus weight decay can explain most of the measured mean contraction; **it does not establish that task gradients vanish**. Recovering gradients or optimizer trajectories would require additional records or controlled training. Historical source identity is not certified by this replay.

The 48-layer symmetric counterfactual predicts a 96-transport composite gain of 0.00040214. This is reported as a prediction only: the available posthoc composite record comes from a differently named run and was not silently joined to the fair-run spectrum.

The historical mHC diagnostics also report mean row-sum errors 0.00463 (48 layers) and 0.00614 (24 layers), despite very small column-sum errors. Finite Sinkhorn iteration therefore did not enforce exact doubly stochastic constraints in these records. Exact mean-preservation theorems must not be asserted directly for these finite-iteration matrices without a numerical qualification.
