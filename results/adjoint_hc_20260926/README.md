# Adjoint residual candidate: CPU integration receipt

Status: current experiment receipt, 2026-09-26. This records an implemented full Transformer and completed **synthetic integration training**, not a natural-language quality result. The current goal and decisions belong to [research/README](../../docs/research/README.md), [architecture](../../docs/research/architecture.md), and [roadmap](../../docs/research/roadmap.md).

## Reproduce

```bash
python3 experiments/adjoint_hc_probe.py --compile-check
```

The [script](../../experiments/adjoint_hc_probe.py) writes [probe.json](probe.json). It uses local PyTorch, generated token IDs, CPU float32, and one intra/inter-op thread. No download, GPU, or remote execution is involved. The receipt contains source digests, environment, data digests, every training loss and gradient norm, validation curves, individual timing samples, and compile consistency results. Timing is host dependent; the token source and numerical training are deterministic on the recorded environment.

## Initialization and ordinary NTP gradients

All arms use the same model seed, and their constructor-created backbone parameters are compared directly before training. There is no corrective checkpoint loading. For two layers, width 64, four heads, vocabulary 32, length 32 and batch 4:

- Baseline, dynamic signed, static signed, and dynamic signed with auxiliary writes frozen have **exactly identical initial logits** in float32; maximum absolute error is `0`.
- A real full-model cross-entropy backward gives the dynamic signed routers total gradient L2 `0.0193372597`; every attention/MLP/output router has a nonzero gradient.
- With the same zero-initialized routers and a zero auxiliary carrier, total initial router gradient is exactly `0`.

This supports the signed carrier's intended role in opening a first-order learning path at an exact baseline initial function. It does not establish superiority of this carrier over other viable initializations.

## Completed integration training

Each of four arms completes 80 AdamW steps on the identical 80 precomputed batches: **10,240 training tokens per arm**, 40,960 across the comparison. AdamW uses learning rate `0.001`, weight decay `0.01`, betas `(0.9, 0.999)`, and gradient clipping at `1.0`. All losses and gradients remain finite. No architecture-specific hyperparameter tuning or outcome-based data changes were performed.

The source is a first-order permutation Markov chain: with probability 0.8, the next token follows one fixed permutation of the current token; otherwise it is uniform random. Training and validation use independent sample seeds and the same transition kernel. Validation uses 2,048 tokens per evaluation. This tests a normal token prediction training path; it deliberately does not represent natural language or a long-range memory challenge.

| Arm | Parameters | Initial validation NLL | Final validation NLL | Router parameter L2 change | Final auxiliary write RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 102,720 | 3.465736 | 1.246839 | — | — |
| Dynamic signed | 103,045 | 3.465736 | 1.246644 | 0.283938 | 0.00504630 |
| Static signed | 102,725 | 3.465736 | 1.246806 | 0.021289 | 0.00163801 |
| Dynamic signed, frozen auxiliary writes | 103,045 | 3.465736 | 1.246640 | 0.286666 | 0 |

Auxiliary write RMS measures `m_final − m_initial` **within the same forward pass on the same diagnostic batch**, using each trained model's live carrier. It is zero at initialization for all candidates and becomes nonzero only in the arms with enabled auxiliary writes. It does not confuse embedding changes across optimization steps with auxiliary writes across depth.

The four validation curves nearly overlap; freezing auxiliary writes is equally effective on this source. **There is no useful quality advantage or evidence that learned auxiliary storage is necessary in this probe.** The result establishes that routes and auxiliary writes can activate under standard NTP without an auxiliary objective. It neither validates nor rejects the candidate's language-model value.

## Local engineering measurements

Whole-model eager forward + cross-entropy + backward + `zero_grad` is measured at two predefined shapes: (a) two layers, width 128, length 64 and vocabulary 32; (b) four layers, width 512, length 128 and vocabulary 128. Both use four heads and batch 2. There are three warmup iterations per arm, followed by 20 measured iterations per arm in (a) and eight in (b), with alternating interleaved arm order. Optimizer stepping, data loading and transfers are excluded. Parameters are shared at initialization. Both shapes are retained; the larger shape was added to examine cost beyond the initial tiny shape, without replacing the slower relative-cost result.

| Shape | Arm | Parameters | Median ms | p10 ms | p90 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| (a) width 128 | Baseline | 406,144 | 1.7756 | 1.7412 | 1.8322 |
| (a) width 128 | Dynamic signed | 406,789 | 2.5158 | 2.4821 | 2.7059 |
| (b) width 512 | Baseline | 12,718,592 | 29.9973 | 29.8911 | 30.9450 |
| (b) width 512 | Dynamic signed | 12,723,209 | 34.6991 | 34.4621 | 35.3983 |

The candidate costs **1.417×** baseline time in (a) and **1.157×** in (b), using the final complete receipt run. Small parameter overhead does not imply negligible execution overhead. Relative overhead is smaller at the larger tested shape, but width, depth, length and vocabulary all change, so this does not isolate a causal width effect. These small, unfused, one-thread CPU numbers cannot predict GPU efficiency, production throughput, activation memory, or the cost at representative LLM shapes.

`torch.compile(backend="aot_eager", fullgraph=True)` completes forward and backward at the training shape. Eager/compiled logits and parameter-gradient maximum absolute differences are both `0`. This is fixed-shape graph capture and differentiation evidence, not a fused-kernel or compiled-throughput result. The environment is PyTorch 2.8.0 on macOS arm64; exact version details are in the JSON.

## Evidence boundary

All nine recorded integration checks pass. This receipt establishes implementation, a live initialization gradient, ordinary synthetic optimization, graph capture, and one local cost observation. Mature LM quality, large-scale stability, strong single-stream/normalization baselines, dynamic HC/mHC/AttnRes comparisons, fused GPU kernels and quality–cost superiority remain untested. Unit/contract tests live separately in the test suite and are not counted as training results here.
