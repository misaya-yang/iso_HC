<!-- doc-status: historical -->
> **历史记录：不再作为当前研究判断、投稿主张或执行计划。**
> 本文原文与数值保留；当前解释见[证据台账](../research/evidence.md)，理论边界见[理论基础](../research/theory.md)，研究目标与下一实验以[当前主线](../research/README.md)和[实验计划](../research/roadmap.md)为准。本文旧启动、删除、工具调用与投稿指令不再生效。

<!-- doc-history-body-begins -->
# The Last Mile of Hyper-Connections: Fixed-Vector Isometric Residual Transport

**Draft status:** arXiv-style technical note, first internal draft.  
**Date:** 2026-05-25.  
**Repository:** `/Users/yang/projects/isoHC`.

## Abstract

Residual connections stabilize deep networks by preserving an identity path, but
they expose only a single residual stream. Hyper-Connections (HC) generalize this
idea by expanding the residual pathway into multiple streams and learning
cross-stream residual transport. This added connectivity creates a final-mile
geometric problem: unconstrained stream mixing is expressive but can destroy the
identity mapping and amplify signals through depth, while Birkhoff-constrained
mixing, as used in manifold-constrained Hyper-Connections (mHC), restores
mean-preserving stability but can dissipate the mean-zero stream subspace.

We study this missing geometry and propose IsoHC, a fixed-vector isometric
constraint for residual transport. IsoHC constrains each residual mixing matrix
to the stabilizer

```text
M_1 = { Q in O(n) : Q 1 = 1 },
```

thereby preserving the residual mean exactly while keeping every singular value
on the invariant-complement subspace equal to one. This separates identity
preservation from diffusion: the stream mean follows the standard residual path,
while the remaining `n-1` stream degrees of freedom are transported without
contraction or expansion.

We provide a theoretical diagnosis showing that Birkhoff stability is not
isometric stability: doubly stochastic residual matrices are non-expansive on
the mean-zero subspace and generically contract it through depth. In controlled
transport detectors, mHC-like diffusion collapses mean-zero energy while
unconstrained HC explodes; IsoHC preserves energy and gradients through 1024
layers. In a 48-layer FineWeb-Edu Transformer posthoc analysis, mHC's composite
mean-zero transport gain falls to `4.39e-4`, whereas IsoHC remains near `0.996`.
At this compute scale we do not claim a decisive language-modeling perplexity
advantage. Instead, we identify and repair a concrete residual-transport
pathology: preserving the residual mean is not enough; the last mile of
Hyper-Connections requires preserving the residual complement.

## 1. Introduction

Residual connections are one of the simplest and most successful ideas in deep
learning. By adding an identity path around a transformation, a residual block
can be written as

```text
x_{l+1} = x_l + F_l(x_l).
```

This form stabilizes optimization because the network always has access to a
direct path through depth. The identity path is not merely a numerical trick: it
defines the semantics of the residual stream. Each layer writes an increment
into a state that remains legible to all subsequent layers.

Hyper-Connections extend this picture. Instead of maintaining one residual
stream, the network maintains `n` residual streams and learns how information
flows between them. In principle, this gives a deeper model more internal
capacity: different streams can carry different features, preserve alternative
views of the computation, or route information across depth. However, the moment
the residual path becomes multi-stream, the identity mapping is no longer
automatic. A learned residual mixing matrix can amplify, rotate, or collapse the
stream space, and products of such matrices can become unstable through depth.

mHC addresses this by projecting the residual mixing matrix onto the Birkhoff
polytope:

```text
B_n = { H : H 1 = 1, 1^T H = 1^T, H >= 0 }.
```

This is a natural and powerful repair. Row and column stochasticity preserve the
uniform stream mean, restoring the identity-mapping property that ordinary
residual connections enjoy. Empirically, mHC shows that this can stabilize large
Hyper-Connection models, especially compared with unconstrained HC.

Our observation is that this repair is incomplete. Birkhoff constraints preserve
the residual mean, but they do not preserve the residual complement. On the
mean-zero subspace

```text
1_perp = { z : 1^T z = 0 },
```

a doubly stochastic matrix behaves like a Markov operator: it is non-expansive
and, except in permutation-like cases, contractive. Thus the same geometry that
restores mean stability can diffuse away the very multi-stream differences that
Hyper-Connections introduce.

This gives a sharper view of the residual-connection problem:

```text
single residual stream:
  stable identity path, limited residual capacity

unconstrained HC:
  expanded stream capacity, unstable transport

mHC / Birkhoff HC:
  restored mean stability, but possible complement contraction

IsoHC:
  restored mean stability and isometric complement transport
```

We argue that fixed-vector isometry is the natural last-mile geometry for
multi-stream residual transport. IsoHC constrains the residual transport to

```text
M_1 = { Q in O(n) : Q 1 = 1 }.
```

This preserves the residual mean exactly and acts as an orthogonal transform on
`1_perp`. It therefore prevents both unconstrained amplification and
Birkhoff-style diffusion.

### Contributions

This note makes four contributions.

1. We formulate the residual-transport gap in Hyper-Connections: preserving the
   residual mean is not equivalent to preserving multi-stream residual capacity.
2. We give a fixed-vector isometric residual transport family that preserves the
   standard residual identity path while keeping the invariant-complement
   subspace non-dissipative.
3. We contrast Birkhoff and fixed-vector isometric geometry through singular
   values on `1_perp`, cumulative/composite transport gain, and stream diversity.
4. We report controlled evidence from residual-only detectors, graph
   oversmoothing experiments, and a 48-layer FineWeb-Edu Transformer posthoc
   analysis. The evidence supports the geometric mechanism, while also showing
   that a decisive large-scale LM outcome bridge remains open.

## 2. From Residual Connections to Multi-Stream Transport

### 2.1 Standard residual semantics

A pre-norm Transformer block can be abstracted as

```text
x_{l+1} = x_l + F_l(LN(x_l)).
```

The residual stream `x_l` is a shared state. Each layer reads from it and writes
an increment into it. This identity path implies that, if the branch `F_l` is
small or untrained, the block remains close to identity.

This property is easy to lose in a multi-stream residual architecture. Let

```text
X_l in R^{n x d}
```

denote `n` residual streams. A generic residual transport update has the form

```text
X_{l+1} = H_l X_l + b_l \otimes y_l,
```

where `H_l in R^{n x n}` mixes streams and `y_l` is the branch output written
into the streams. Let

```text
mu(X) = (1/n) 1^T X
```

be the stream mean. To embed the standard residual path into the mean stream, we
want

```text
mu(X_{l+1}) = mu(X_l) + y_l.
```

A sufficient condition is

```text
1^T H_l = 1^T,    (1/n) 1^T b_l = 1.
```

If the branch reads from a mean-preserving aggregate, e.g.

```text
z_l = (1/n) a_l^T X_l,
```

with `a_l` normalized around the all-ones direction, then the stream mean retains
the standard residual semantics.

The question is what should happen to the remaining `n-1` dimensions.

### 2.2 The residual complement

Let

```text
P_perp = I - (1/n) 1 1^T.
```

Then every stream state decomposes as

```text
X = 1 \otimes mu(X) + P_perp X.
```

The first term is the residual mean. The second term is the residual complement:
the difference between streams. If Hyper-Connections are meant to provide more
than a redundant copy of the residual stream, then this complement is where the
extra stream capacity lives.

Thus multi-stream residual transport has two separate responsibilities:

```text
mean direction:
  preserve the residual identity path

mean-zero complement:
  transport stream differences without uncontrolled expansion or dissipation
```

mHC solves the first responsibility by using Birkhoff constraints. IsoHC targets
both.

## 3. Birkhoff Stability Is Not Isometric Stability

### 3.1 Birkhoff residual mixing

mHC constrains residual mixing matrices to be doubly stochastic:

```text
H 1 = 1,    1^T H = 1^T,    H >= 0.
```

This ensures that the stream mean is preserved:

```text
mu(HX) = mu(X).
```

By the Birkhoff-von Neumann theorem, every doubly stochastic matrix is a convex
combination of permutation matrices:

```text
H = sum_k alpha_k P_k,
alpha_k >= 0, sum_k alpha_k = 1.
```

Since permutations are orthogonal, this makes Birkhoff mixing non-expansive in
operator norm:

```text
||H||_2 <= 1.
```

The key point is that non-expansion is not the same as isometry. A convex
combination of permutations is generally diffusive. It preserves the mean, but
it can shrink the mean-zero component.

### 3.2 Complement contraction

Let `U in R^{n x (n-1)}` be any orthonormal basis for `1_perp`. The action of a
mean-preserving matrix on the complement is

```text
H_perp = U^T H U.
```

For a Birkhoff matrix,

```text
||H_perp||_2 <= 1.
```

If the largest singular value on the complement satisfies

```text
sigma_max(H_perp) <= rho < 1,
```

then repeated transport contracts complement signals exponentially:

```text
||P_perp H_L ... H_1 P_perp x||
  <= (prod_l sigma_max(U^T H_l U)) ||P_perp x||.
```

When the average complement singular value is, for example, `0.922`, the
amplitude scale after 48 layers is approximately

```text
0.922^48 ~= 0.02.
```

Energy decays quadratically in this amplitude. Thus a moderate per-layer
contraction becomes a severe depth-wise loss of stream diversity.

This diagnosis does not contradict mHC. mHC is designed to prevent the
instability of unconstrained HC by restoring identity mapping and controlling
composite gain. Our claim is narrower: the mHC stability metrics do not imply
that the residual complement is preserved.

### 3.3 Why not require nonnegative orthogonality?

One might ask for a matrix that is both Birkhoff and orthogonal. This leaves only
a discrete family. If a matrix is nonnegative and orthogonal, distinct rows must
have disjoint support. With row and column sums equal to one, the matrix is a
permutation. Therefore continuous, learnable, non-diffusive residual mixing
cannot remain inside the nonnegative Birkhoff geometry. To avoid contraction
without becoming a discrete permutation, the complement action must allow signed
orthogonal transport.

## 4. IsoHC: Fixed-Vector Isometric Residual Transport

### 4.1 Constraint family

IsoHC replaces the Birkhoff residual manifold with the fixed-vector orthogonal
stabilizer

```text
M_1 = { Q in O(n) : Q 1 = 1 }.
```

Equivalently, using `hat{1} = 1 / sqrt(n)`,

```text
Q hat{1} = hat{1},    Q^T Q = I.
```

For any stream state `X`,

```text
mu(QX) = mu(X),
```

and

```text
||P_perp QX||_F = ||P_perp X||_F.
```

Thus the residual mean remains the standard residual path, while the complement
is transported isometrically.

### 4.2 Projection / retraction

Let `v = hat{1}` and let `U` be an orthonormal basis for `v_perp`. Any matrix in
`M_1` can be written as

```text
Q = v v^T + U R U^T,
R in O(n-1).
```

Given an unconstrained matrix `A`, the Frobenius-nearest exact projection, away
from singular polar degeneracies, is

```text
Pi_v(A) = v v^T + U polar(U^T A U) U^T.
```

The implementation uses Newton-Schulz polar iterations as an efficient
fixed-vector retraction:

```text
B = U^T A U,
R_K ~= polar(B),
Q_K = v v^T + U R_K U^T.
```

Finite Newton-Schulz iterations preserve the fixed vector exactly by
construction, while orthogonality is approximate and monitored during training.
This distinction matters: exact polar projection is an ideal geometric operator;
finite-step Newton-Schulz is an approximate retraction with measurable
orthogonality error.

### 4.3 Two-branch Transformer update

In a pre-norm Transformer, each layer has attention and MLP residual branches.
The multi-stream residual update can be written as

```text
z_l^attn = (1/n) (a_l^attn)^T X_l
y_l^attn = Attn_l(LN(z_l^attn))
X_{l+1/2} = H_l^attn X_l + b_l^attn \otimes y_l^attn

z_l^mlp = (1/n) (a_l^mlp)^T X_{l+1/2}
y_l^mlp = MLP_l(LN(z_l^mlp))
X_{l+1} = H_l^mlp X_{l+1/2} + b_l^mlp \otimes y_l^mlp.
```

If `H_l^* 1 = 1` and `(1/n) 1^T b_l^* = 1`, then

```text
mu(X_{l+1}) = mu(X_l) + y_l^attn + y_l^mlp,
```

matching the standard residual semantics in the stream mean. IsoHC then adds the
missing condition that the transport part preserves the complement norm.

## 5. Experiments

Our experiments are designed to answer a mechanism question rather than to claim
large-scale language-modeling superiority:

```text
Does the residual transport preserve both the identity mean and the
multi-stream complement through depth?
```

The current evidence supports this mechanism strongly. It does not yet establish
that IsoHC improves large-scale LM perplexity over mHC.

### 5.1 Residual-only depth detector

We first remove attention and MLP branches and study pure residual transport. The
detector compares:

```text
IsoHC
mHC-lite / Birkhoff-like diffusion
GCN-style diffusion
unconstrained mixing
```

across depths up to 1024.

For `n=8`, `L=1024`, fp32:

| method | grad mean | mean-zero energy | stream abs cosine | sigma min | sigma max |
|---|---:|---:|---:|---:|---:|
| IsoHC | 1.0000 | 1.0000 | 0.0250 | 1.0000 | 1.0000 |
| mHC-lite | 0.3536 | 0.0000 | 1.0000 | 0.5724 | 1.0000 |
| GCN diffusion | 0.3533 | 0.0000 | 1.0000 | 0.8000 | 1.0000 |
| unconstrained | 5.58e13 | 5.61e13 | 0.9877 | 0.4270 | 1.6155 |

This isolates the transport geometry:

```text
unconstrained HC:
  can explode through depth

Birkhoff / diffusion:
  prevents explosion but collapses the complement

IsoHC:
  preserves gradient scale, mean-zero energy, and stream diversity
```

The bf16/fp32 mixed precision variant, where the model uses bf16 but the
projection is computed in fp32, matches the fp32 stability behavior. In contrast,
performing the fixed-vector projection entirely in bf16 breaks the invariant
constraint and is not viable.

### 5.2 Graph oversmoothing stress test

Although IsoHC is motivated by residual transport, graph neural networks provide
a useful stress test for diffusion and collapse. In a synthetic SBM graph with
depth 128:

| method | energy | v-centered variance | cosine | Dirichlet energy |
|---|---:|---:|---:|---:|
| GCN | 0.2003 | 0.0012 | 1.0000 | 0.0000 |
| residual GCN | 0.2003 | 0.0012 | 1.0000 | 0.0000 |
| IsoNode | 1.0000 | 1.0000 | 0.0325 | 7940.2 |

The diffusion baselines oversmooth, while the isometric transport preserves
nontrivial variation.

On Cora node classification, a corrected stream architecture (`IsoStream v2`)
recovers deep performance where ordinary GCNs collapse:

| model | L2 | L16 | L32 |
|---|---:|---:|---:|
| GCN | 77.6 +/- 1.9 | 31.1 +/- 0.2 | 31.2 +/- 0.3 |
| ResGCN | 75.8 +/- 1.2 | 28.8 +/- 4.0 | 28.6 +/- 2.8 |
| IsoStream v2 | -- | 73.1 +/- 1.9 | 66.6 +/- 3.4 |

These results are not the primary LM claim, but they support the broader
principle that diffusive propagation can erase useful complement structure and
that isometric transport can preserve it.

### 5.3 48-layer FineWeb-Edu Transformer mechanism run

We trained a 48-layer GPT-style Transformer on a FineWeb-Edu token cache:

```text
layers: 48
d_model: 512
heads: 8
streams: 4
context length: 512
parameters: about 177M
budget: 20M tokens per method
```

The comparison used a common batch selected by a fair batch probe.

| method | val loss | val PPL | E_perp init -> final | stream cosine init -> final |
|---|---:|---:|---|---|
| baseline | 5.7366 | 310.01 | n/a | n/a |
| identity-HC | 5.7157 | 303.60 | 0.0222 -> 0.0503 | 0.9993 -> 0.9971 |
| unconstrained HC | 5.6760 | 291.78 | 0.0222 -> 0.0547 | 0.9993 -> 1.0000 |
| mHC | 5.7249 | 306.39 | 0.0224 -> 0.0136 | 0.9993 -> 0.9998 |
| IsoHC | 5.7206 | 305.10 | 0.0224 -> 0.0546 | 0.9993 -> 0.9967 |

The short-budget LM loss should not be overinterpreted. The key mechanism
diagnostics are:

| method | fixed-vector error | orthogonality error | `1_perp` singular values |
|---|---:|---:|---|
| identity-HC | 0.0 | 0.0 | exact identity |
| unconstrained HC | 0.1050 | 0.1694 | unconstrained |
| mHC | row err 0.0046 / col err 2.30e-7 | n/a | mean 0.9222, min 0.9208, max 0.9237 |
| IsoHC | 2.79e-7 | 5.57e-4 | mean 0.99996, min 0.99982, max 1.00012 |

mHC preserves the stochastic constraint but contracts the mean-zero stream
subspace. IsoHC preserves the fixed vector and keeps the complement singular
values near one.

### 5.4 Checkpoint posthoc analysis

To measure depth accumulation directly, we performed posthoc analysis on the
checkpointed 48-layer run, using 16 validation batches and batch size 1 for
intervention and gradient probes.

| method | base val loss | base PPL | final composite sv mean on `1_perp` | grad slope | best complement-removal delta | replace with identity delta |
|---|---:|---:|---:|---:|---:|---:|
| mHC | 5.900727 | 365.30 | 0.000439 | -0.021109 | +0.000289 | -0.000524 |
| IsoHC | 5.901274 | 365.50 | 0.996349 | -0.021373 | +0.000516 | +0.000713 |

This is the cleanest LM mechanism result:

```text
mHC:
  composite complement transport gain collapses to about 4.39e-4

IsoHC:
  composite complement transport gain remains near 0.996
```

The functional bridge remains weak at this training budget. Complement removal
changes validation loss only at the `1e-4` to `1e-3` scale, and replacing learned
IsoHC with identity or random fixed-vector isometries changes loss by less than
`0.001`. We therefore do not claim that the preserved complement is already
strongly used by this language model. The result establishes the transport
pathology, not yet a decisive outcome improvement.

## 6. Discussion

### 6.1 What problem is IsoHC solving?

IsoHC is not proposed as "another way to improve small-model perplexity." Its
target is the residual connection itself. Ordinary residual connections provide
a stable identity path but only one stream. Hyper-Connections expand the stream
space but introduce a residual-transport geometry problem. mHC repairs the mean
identity path by making mixing doubly stochastic. IsoHC completes the geometric
repair by preserving the complement as well.

In this sense, IsoHC is a last-mile residual transport primitive:

```text
identity path:
  preserved in the stream mean

multi-stream residual capacity:
  preserved in the mean-zero complement

depth stability:
  no unconstrained amplification and no Birkhoff diffusion
```

### 6.2 Why perplexity is not the right first metric

At small token budgets, language-modeling perplexity is dominated by shallow
statistics, optimization transients, and data/model scale effects. A 20M-token
run may never require the model to use the additional residual complement
capacity. Thus a lack of PPL separation does not refute the geometric mechanism.
It only means the outcome bridge has not been established.

The right first-order metrics are transport metrics:

```text
single-layer complement singular spectrum
composite complement gain
mean-zero stream energy
stream cosine / effective rank
layer-wise gradient transport
intervention sensitivity after complement removal
```

Only after these show that the complement is both preserved and functionally
used should large-scale perplexity be used as the main outcome measure.

### 6.3 Identity-HC as a necessary control

The identity matrix is a member of `M_1`. It preserves the mean and complement
perfectly, but it does not communicate between streams. This makes identity-HC
an important control. If learned IsoHC performs no better than identity-HC and
replacing learned `H_l` by identity has no effect, then the architecture has not
activated the value of learned isometric stream communication.

Our current 48-layer posthoc replacement results show only a very small
IsoHC-to-identity loss delta. This is a limitation. It suggests that the current
LM setting verifies the geometry but does not yet demonstrate strong use of
learned rotations. Future stress tasks should force cross-stream communication
to become functionally necessary.

### 6.4 Relation to signed and spectral HC variants

Recent work beyond the Birkhoff polytope has argued that nonnegativity can limit
expressivity and that signed spectral constraints can stabilize HC while
enabling subtractive interactions. IsoHC is complementary but more specific. A
spectral norm constraint such as

```text
||H||_2 <= 1
```

prevents expansion, but it does not preserve the residual mean and does not
prevent directional contraction. IsoHC imposes the stronger geometry:

```text
H 1 = 1,    H^T H = I.
```

This simultaneously preserves the identity path and all complement singular
values. The relevant comparison is therefore not simply "signed vs nonnegative",
but "non-expansive vs isometric on the residual complement."

## 7. Limitations

This draft intentionally makes a limited claim.

1. **No large-scale LM superiority claim.** We do not show that IsoHC beats mHC
   in large-scale pretraining. The current FineWeb-Edu run is a 20M-token
   mechanism probe.
2. **Weak intervention effect in the current LM.** Complement removal and
   replacement probes produce small loss changes. This means the preserved
   complement has not yet been shown to be strongly used by the trained model.
3. **Identity-HC remains a serious control.** Since identity is itself an
   isometry, future experiments must show when learned isometric stream mixing is
   better than no mixing.
4. **Newton-Schulz is approximate.** Finite-step NS preserves the fixed vector by
   construction but only approximately enforces orthogonality. Orthogonality
   errors must be monitored.
5. **The current draft is a mechanism note.** To become a full empirical method
   paper, it would need either deeper stress tasks, stronger intervention
   evidence, or larger-token language modeling runs.

## 8. Future Experiments

The most useful next experiments are not broader PPL sweeps. They are targeted
stress tests that make the residual complement matter.

### 8.1 mHC-style gain plus complement gain

Replicate the mHC diagnostic style:

```text
Amax row/column single-layer gain
Amax row/column composite gain
gradient norm vs depth
```

Then add the IsoHC-specific complement diagnostics:

```text
singular spectrum of U^T H_l U
composite singular spectrum of U^T H_L ... H_1 U
mean-zero stream energy
stream cosine / effective rank
```

This directly shows:

```text
mHC and IsoHC both stabilize residual mean gain,
but only IsoHC preserves complement isometry.
```

### 8.2 Token-level intervention

Instead of measuring only average validation loss after complement removal,
measure per-token sensitivity:

```text
KL(p_full || p_removed_perp)
top-k probability shift
tail-token loss delta
high-entropy-token loss delta
long-context-position loss delta
```

This may reveal that the preserved complement matters for difficult positions
even if the average loss delta is small.

### 8.3 Deep-thin stress models

Depth is the natural axis for this claim. Rather than increasing token count
first, run deeper and thinner models:

```text
L in {48, 72, 96, 128}
d_model reduced to fit memory
methods: mHC, IsoHC, identity-HC, unconstrained HC
```

The success criterion is not merely lower perplexity. It is a depth-dependent
separation in complement gain, stream diversity, and intervention sensitivity.

### 8.4 Synthetic routing / associative recall

Design a task that requires multiple streams to preserve different pieces of
state through depth. Examples include associative recall, multi-key routing, or
finite-state tasks with independent latent factors. Such tasks can provide an
outcome bridge at small scale by making complement capacity necessary.

## 9. Conclusion

Residual connections solve the identity-path problem for deep networks, but they
do so with a single stream. Hyper-Connections expand residual connectivity, but
this creates a final-mile question: what geometry should govern residual
transport?

mHC gives an important answer: constrain residual mixing to preserve the stream
mean and prevent the amplification of unconstrained HC. We show that this is not
the whole story. Birkhoff stability is not isometric stability. A
mean-preserving diffusive operator can erase the mean-zero stream complement,
turning multi-stream residual capacity into a shrinking transient.

IsoHC proposes the corresponding fixed-vector isometric repair. By constraining
residual transport to `Q^T Q = I` and `Q 1 = 1`, it preserves both the standard
residual mean path and the full complement norm. In controlled detectors and
48-layer Transformer posthoc analysis, this distinction is clear: mHC's
composite complement transport collapses, while IsoHC remains approximately
isometric.

The current evidence supports IsoHC as a geometric mechanism, not yet as a
large-scale LM performance claim. Framed this way, the contribution is simple:
the last mile of Hyper-Connections is not merely restoring the residual mean; it
is preserving the residual complement.

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image
Recognition. CVPR, 2016.

[2] A. Vaswani et al. Attention Is All You Need. NeurIPS, 2017.

[3] D. Zhu et al. Hyper-Connections. arXiv:2409.19606, 2024.

[4] Z. Xie et al. mHC: Manifold-Constrained Hyper-Connections.
arXiv:2512.24880, 2025.

[5] Z. Liu, H. Zhang, and A. Li. Beyond the Birkhoff Polytope:
Spectral-Sphere-Constrained Hyper-Connections. arXiv:2603.20896, 2026.

[6] L. Zhao and L. Akoglu. PairNorm: Tackling Oversmoothing in GNNs. ICLR, 2020.

[7] T. Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness. NeurIPS, 2022.

