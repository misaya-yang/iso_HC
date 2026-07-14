# HC Causal Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. In this repository, inline execution is the default unless the user explicitly authorizes subagents.

**Goal:** Implement the approved P0 causal-control experiment package on the existing IsoHC LM pipeline, with exact provenance, persistent complement interventions, and no claims beyond verified artifacts.

**Architecture:** Reuse the current TwoBranchHCTransformer, training loop, transport analyzer, checkpoint format, and data loader. Add configuration plumbing and one experiment-matrix entrypoint; do not add a second trainer or a new dependency. Keep the old mhc method loadable, but label new evidence static-birkhoff-hc.

**Tech Stack:** Python 3.13, PyTorch 2.10, unittest, argparse, JSON, pathlib, subprocess, bash.

## Global Constraints

- Preserve existing checkpoint parameter keys and default model behavior.
- Add no dependency beyond Python standard library and installed PyTorch.
- Treat docs/0714_experiment_guidance.md as hypotheses, not authority.
- Never report the current static Birkhoff implementation as faithful dynamic mHC.
- Keep output directories variant- and seed-specific; refuse to overwrite a completed run.
- Record initial, periodic, final, and intervention evidence separately.
- Local CPU checks prove orchestration only; they do not prove RTX 5090, FineWeb-Edu, throughput, or scaling claims.
- Do not implement faithful dynamic mHC, BiLip-HC, Givens/Householder IsoHC, P1 synthetic tasks, or P2/P3 scaling in this plan.
- Do not push, deploy, delete prior results, or rewrite git history.

---

### Task 1: Add Mixer Controls and Exact Transport Geometry

**Files:**
- Modify: lm/mixing.py:93-252
- Modify: lm/transport_analysis.py:14-139
- Test: tests/test_lm_next_phase_contracts.py

**Interfaces:**
- Consumes: existing create_mixing(n_streams, mixing_type, **kwargs).
- Produces: IsoHCMixing(n_streams, complement_scale=1.0), MHCMixing(n_streams, identity_blend=1.0), and complement_spectrum(H) with all per-step and final-composite complement singular values plus two leakage values.

- [ ] **Step 1: Write failing mixer and geometry tests**

Add these imports:

~~~python
from lm.mixing import IsoHCMixing, MHCMixing
from lm.transport_analysis import complement_spectrum, collect_transport_report
~~~

Add these test methods:

~~~python
def test_mixer_controls_preserve_defaults_and_apply_requested_gain(self):
    torch.manual_seed(31)
    base = MHCMixing(
        4,
        sinkhorn_iters=10,
        temperature=1.0,
        diag_bias=4.0,
        noise_std=0.0,
    )
    blended = MHCMixing(
        4,
        sinkhorn_iters=10,
        temperature=1.0,
        diag_bias=4.0,
        noise_std=0.0,
        identity_blend=0.5,
    )
    blended.logits.data.copy_(base.logits.data)
    expected = 0.5 * torch.eye(4) + 0.5 * base()
    self.assertTrue(torch.allclose(blended(), expected, atol=1e-7))

    explicit_default = MHCMixing(
        4,
        sinkhorn_iters=10,
        temperature=1.0,
        diag_bias=4.0,
        noise_std=0.0,
        identity_blend=1.0,
    )
    explicit_default.logits.data.copy_(base.logits.data)
    self.assertTrue(torch.equal(base(), explicit_default()))

    scaled = IsoHCMixing(
        4,
        ns_steps=5,
        use_svd=True,
        svd_fallback=False,
        complement_scale=0.8,
    )
    stats = complement_spectrum(scaled())
    self.assertEqual(len(stats["singular_values"]), 3)
    for value in stats["singular_values"]:
        self.assertAlmostEqual(value, 0.8, places=5)

def test_mixer_controls_validate_ranges(self):
    with self.assertRaises(ValueError):
        MHCMixing(4, temperature=0.0)
    with self.assertRaises(ValueError):
        MHCMixing(4, sinkhorn_iters=0)
    with self.assertRaises(ValueError):
        MHCMixing(4, identity_blend=1.1)
    with self.assertRaises(ValueError):
        IsoHCMixing(4, complement_scale=0.0)

def test_transport_report_keeps_all_complement_modes_and_leakage(self):
    H = torch.eye(4)
    stats = complement_spectrum(H)
    self.assertEqual(len(stats["singular_values"]), 3)
    for value in stats["singular_values"]:
        self.assertAlmostEqual(value, 1.0, places=7)
    self.assertAlmostEqual(stats["mean_to_perp_leakage"], 0.0, places=7)
    self.assertAlmostEqual(stats["perp_to_mean_leakage"], 0.0, places=7)
    self.assertAlmostEqual(stats["row_sum_error"], 0.0, places=7)
    self.assertAlmostEqual(stats["col_sum_error"], 0.0, places=7)

    report = collect_transport_report([H, H])
    self.assertEqual(report["num_transports"], 2)
    self.assertEqual(len(report["steps"][0]["singular_values"]), 3)
    self.assertEqual(
        len(report["final"]["composite_singular_values"]),
        3,
    )
    self.assertAlmostEqual(
        report["final"]["composite_sv_mean"],
        1.0,
        places=7,
    )
~~~

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_mixer_controls_preserve_defaults_and_apply_requested_gain \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_mixer_controls_validate_ranges \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_transport_report_keeps_all_complement_modes_and_leakage -v
~~~

Expected: FAIL because complement_scale, identity_blend, singular_values, and leakage fields do not exist.

- [ ] **Step 3: Implement the minimal mixer controls**

Update IsoHCMixing construction and forward logic:

~~~python
def __init__(
    self,
    n_streams,
    ns_steps=5,
    init_scale=0.01,
    use_svd=False,
    svd_fallback=True,
    fallback_tol=1e-3,
    complement_scale=1.0,
):
    super().__init__(n_streams)
    if not 0.0 < complement_scale <= 1.0:
        raise ValueError("complement_scale must be in (0, 1]")
    self.ns_steps = ns_steps
    self.use_svd = use_svd
    self.svd_fallback = svd_fallback
    self.fallback_tol = fallback_tol
    self.complement_scale = float(complement_scale)
    self.H_raw = nn.Parameter(
        torch.eye(n_streams) + torch.randn(n_streams, n_streams) * init_scale
    )
    U = construct_orthogonal_complement(
        n_streams,
        device="cpu",
        dtype=torch.float32,
    )
    self.register_buffer("U", U)

def forward(self):
    H = iso_ns_project(
        self.H_raw,
        U=self.U,
        steps=self.ns_steps,
        use_svd=self.use_svd,
        return_diagnostics=False,
        svd_fallback=self.svd_fallback,
        fallback_tolerance=self.fallback_tol,
    )
    if self.complement_scale == 1.0:
        return H
    ones = torch.ones(
        self.n_streams,
        1,
        device=H.device,
        dtype=H.dtype,
    )
    P = ones @ ones.T / self.n_streams
    return P + self.complement_scale * (H - P)
~~~

Update MHCMixing construction and forward logic:

~~~python
def __init__(
    self,
    n_streams,
    sinkhorn_iters=10,
    temperature=1.0,
    diag_bias=4.0,
    noise_std=0.01,
    identity_blend=1.0,
):
    super().__init__(n_streams)
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be at least 1")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= identity_blend <= 1.0:
        raise ValueError("identity_blend must be in [0, 1]")
    self.sinkhorn_iters = int(sinkhorn_iters)
    self.temperature = float(temperature)
    self.diag_bias = float(diag_bias)
    self.noise_std = float(noise_std)
    self.identity_blend = float(identity_blend)
    logits_init = torch.eye(n_streams) * diag_bias
    logits_init += torch.randn(n_streams, n_streams) * noise_std
    self.logits = nn.Parameter(logits_init)

def forward(self):
    H = self.sinkhorn(self.logits)
    if self.identity_blend == 1.0:
        return H
    eye = torch.eye(
        self.n_streams,
        device=H.device,
        dtype=H.dtype,
    )
    return (1.0 - self.identity_blend) * eye + self.identity_blend * H
~~~

Pass the new values through create_mixing:

~~~python
complement_scale=kwargs.get("complement_scale", 1.0)
~~~

for IsoHCMixing, and:

~~~python
identity_blend=kwargs.get("identity_blend", 1.0)
~~~

for MHCMixing.

Update the IsoHCMixing docstring to state the implementation that actually runs: the tiny projected complement matrix is computed internally in float64 by iso_ns_project and converted back to the caller dtype. Remove the stale fp32 claim.

Correct MHCMixing diagnostics by constructing U once per call, taking svdvals(U.T @ H @ U) without filtering, and computing column error against ones.T:

~~~python
U = construct_orthogonal_complement(
    n,
    device=device,
    dtype=torch.float32,
)
s = torch.linalg.svdvals(U.T @ H.float() @ U)
col_sum_err = torch.norm(
    H.sum(dim=0, keepdim=True) - ones.T,
    p=2,
).item()
~~~

- [ ] **Step 4: Return exact singular values and leakage from transport_analysis**

Replace complement_spectrum with:

~~~python
def complement_spectrum(H, U=None):
    H = H.detach().float()
    n = H.shape[0]
    if H.shape != (n, n):
        raise ValueError(f"Expected square H, got {tuple(H.shape)}")
    if U is None:
        U = mean_zero_basis(n, device=H.device, dtype=torch.float32)
    else:
        U = U.to(device=H.device, dtype=torch.float32)

    ones = torch.ones(n, 1, device=H.device, dtype=torch.float32)
    v = ones / (n ** 0.5)
    B = U.T @ H @ U
    s = torch.linalg.svdvals(B)
    return {
        "singular_values": s.cpu().tolist(),
        "sv_min": s.min().item(),
        "sv_mean": s.mean().item(),
        "sv_max": s.max().item(),
        "identity_distance": torch.norm(
            B - torch.eye(n - 1, device=H.device),
            p="fro",
        ).item(),
        "mean_to_perp_leakage": torch.norm(U.T @ H @ v).item(),
        "perp_to_mean_leakage": torch.norm(v.T @ H @ U).item(),
        "row_sum_error": torch.norm(H @ ones - ones).item(),
        "col_sum_error": torch.norm(
            ones.T @ H - ones.T
        ).item(),
    }
~~~

In collect_transport_report, call complement_spectrum(H, U) for each step, copy its fields into steps, and retain the existing prefix-product calculation. Add the exact composite values to every prefix row:

~~~python
"composite_singular_values": c.cpu().tolist(),
~~~

Build final with measured maxima:

~~~python
final = dict(prefix[-1])
for key in (
    "mean_to_perp_leakage",
    "perp_to_mean_leakage",
    "row_sum_error",
    "col_sum_error",
):
    final[f"{key}_max"] = max(step[key] for step in steps)
~~~

Return this final mapping instead of prefix[-1]. Do not infer leakage or stochasticity errors from singular values.

- [ ] **Step 5: Run focused and existing transport tests**

Run the Step 2 command, then:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_transport_report_tracks_composite_complement_gain -v
~~~

Expected: all listed tests PASS.

- [ ] **Step 6: Commit Task 1**

~~~bash
git add lm/mixing.py lm/transport_analysis.py tests/test_lm_next_phase_contracts.py
git commit -m "feat: add causal transport controls"
~~~

### Task 2: Plumb Configurations, Preserve Aliases, and Freeze Only Mixers

**Files:**
- Modify: lm/models.py:516-610
- Modify: experiments/lm_5090_next_runs.py:106-220,344-447
- Test: tests/test_lm_next_phase_contracts.py

**Interfaces:**
- Consumes: Task 1 mixer kwargs.
- Produces: TwoBranchHCTransformer with mixing_kwargs=None, create_model support for static-birkhoff-hc and scaled-isohc, freeze_mixing behavior, and failure summaries written before exceptions are re-raised.

- [ ] **Step 1: Add failing configuration tests**

Add:

~~~python
def test_runner_propagates_mixer_controls_and_freezes_only_mixers(self):
    cfg = build_preset_configs(
        preset="run0",
        methods=["static-birkhoff-hc"],
        output_dir="outputs/test",
        batch_size=2,
    )[0]
    cfg.update({
        "mixing_kwargs": {
            "diag_bias": 6.0,
            "temperature": 2.0,
            "noise_std": 0.0,
            "sinkhorn_iters": 5,
            "identity_blend": 0.75,
        },
        "lambda_a": 0.03,
        "lambda_b": 0.1,
        "freeze_mixing": True,
    })
    model = create_model(cfg, vocab_size=128, device=torch.device("cpu"))

    self.assertEqual(model.mixing_type, "mhc")
    self.assertAlmostEqual(model.readout_lambda.item(), 0.03)
    self.assertAlmostEqual(model.injection_lambda.item(), 0.1)
    for mixing in list(model.attn_mixings) + list(model.mlp_mixings):
        self.assertEqual(mixing.sinkhorn_iters, 5)
        self.assertAlmostEqual(mixing.temperature, 2.0)
        self.assertAlmostEqual(mixing.identity_blend, 0.75)
        self.assertTrue(all(not p.requires_grad for p in mixing.parameters()))
    self.assertTrue(model.readout_lambda.requires_grad)
    self.assertTrue(model.attns[0].q_proj.weight.requires_grad)

def test_legacy_mhc_alias_and_scaled_isohc_remain_supported(self):
    legacy = build_preset_configs(
        preset="run0",
        methods=["mhc"],
        output_dir="outputs/test",
        batch_size=2,
    )[0]
    scaled = build_preset_configs(
        preset="run0",
        methods=["scaled-isohc"],
        output_dir="outputs/test",
        batch_size=2,
    )[0]
    scaled["mixing_kwargs"] = {"complement_scale": 0.9}

    legacy_model = create_model(legacy, 128, torch.device("cpu"))
    scaled_model = create_model(scaled, 128, torch.device("cpu"))
    self.assertEqual(legacy_model.mixing_type, "mhc")
    self.assertEqual(scaled_model.mixing_type, "isohc")
    self.assertAlmostEqual(
        scaled_model.attn_mixings[0].complement_scale,
        0.9,
    )
    reloaded_legacy = create_model(
        legacy,
        128,
        torch.device("cpu"),
    )
    reloaded_legacy.load_state_dict(
        legacy_model.state_dict(),
        strict=True,
    )

def test_failed_run_writes_summary_before_reraising(self):
    cfg = build_preset_configs(
        preset="run0",
        methods=["identity-hc"],
        output_dir="outputs/test",
        total_tokens=4096,
        batch_size=2,
        dataset="random",
        use_compile=False,
    )[0]
    cfg.update({
        "num_workers": 0,
        "max_samples": 8,
        "max_samples_val": 4,
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["save_dir"] = tmpdir
        with patch(
            "experiments.lm_5090_next_runs.create_model",
            side_effect=RuntimeError("expected failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "expected failure",
            ):
                run_single(cfg)
        summary = json.loads(
            (Path(tmpdir) / "run_summary.json").read_text()
        )
    self.assertFalse(summary["success"])
    self.assertIn("expected failure", summary["error"])
~~~

Add json, pathlib.Path, unittest.mock.patch, and run_single to the test imports.

- [ ] **Step 2: Run and confirm failure**

Run:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_runner_propagates_mixer_controls_and_freezes_only_mixers \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_legacy_mhc_alias_and_scaled_isohc_remain_supported \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_failed_run_writes_summary_before_reraising -v
~~~

Expected: FAIL with unknown method or unexpected mixing_kwargs.

- [ ] **Step 3: Add model-level mixing_kwargs without removing legacy arguments**

Add mixing_kwargs=None at the end of TwoBranchHCTransformer.__init__. Resolve values once:

~~~python
resolved_mixing_kwargs = {
    "ns_steps": ns_steps,
    "svd_fallback": svd_fallback,
    "sinkhorn_iters": sinkhorn_iters,
}
resolved_mixing_kwargs.update(dict(mixing_kwargs or {}))
self.mixing_kwargs = resolved_mixing_kwargs
~~~

Use self.mixing_kwargs for both attention and MLP create_mixing calls. Do not rename any existing parameter or state-dict field.

- [ ] **Step 4: Add runner aliases, gates, and freeze handling**

Define:

~~~python
HC_METHOD_TO_MIXING = {
    "identity-hc": "identity",
    "unconstrained": "unconstrained",
    "mhc": "mhc",
    "static-birkhoff-hc": "mhc",
    "isohc": "isohc",
    "scaled-isohc": "isohc",
    "orthogonal": "orthogonal",
}
~~~

Replace the HC method branch with a membership check against this mapping and construct the model with:

~~~python
model = TwoBranchHCTransformer(
    vocab_size=vocab_size,
    d_model=config["d_model"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    n_streams=config["n_streams"],
    context_length=config["context_length"],
    mixing_type=HC_METHOD_TO_MIXING[method],
    mlp_ratio=config["mlp_ratio"],
    dropout=config["dropout"],
    lambda_a=config.get("lambda_a", 0.01),
    lambda_b=config.get("lambda_b", 0.01),
    ns_steps=5,
    svd_fallback=(method not in {"isohc", "scaled-isohc"}),
    sinkhorn_iters=10,
    use_flash=config["use_flash"],
    mixing_kwargs=config.get("mixing_kwargs"),
)
if config.get("freeze_mixing", False):
    for module_list in (model.attn_mixings, model.mlp_mixings):
        for mixing in module_list:
            mixing.requires_grad_(False)
~~~

- [ ] **Step 5: Persist failed run summaries before re-raising**

Add:

~~~python
def write_run_summary(config, summary):
    os.makedirs(config["save_dir"], exist_ok=True)
    path = os.path.join(config["save_dir"], "run_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return path
~~~

Move the try boundary in run_single so it starts before tokenizer, data-loader, and model construction. Call write_run_summary inside the exception handler before bare raise, and once on the success path. Remove the old write block after the try/except so every attempted run writes exactly once, including setup failures. Keep device selection and config validation outside only when no run has started.

- [ ] **Step 6: Run configuration tests and the existing preset tests**

Run the Step 2 command, then:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_5090_presets_match_teacher_run_plan \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_identity_hc_method_uses_multistream_identity_mixing \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_spectral_hc_methods_are_not_in_main_experiment_runner -v
~~~

Expected: all listed tests PASS.

- [ ] **Step 7: Commit Task 2**

~~~bash
git add lm/models.py experiments/lm_5090_next_runs.py tests/test_lm_next_phase_contracts.py
git commit -m "feat: plumb causal experiment configs"
~~~

### Task 3: Add Schema-2 State Metrics and Geometry Trajectories

**Files:**
- Modify: lm/diagnostics.py:1-234
- Modify: lm/train.py:42-215
- Modify: experiments/lm_5090_next_runs.py:297-447
- Test: tests/test_lm_next_phase_contracts.py

**Interfaces:**
- Consumes: collect_transport_report from Task 1 and config from Task 2.
- Produces: mean_zero_norm_ratio, squared mean_zero_energy, centered stream cosine, initial_transport, training_transport_history, final_transport, runtime_provenance.

- [ ] **Step 1: Write failing schema and history tests**

Add imports:

~~~python
from lm.diagnostics import (
    DiagnosticsCollector,
    compute_centered_stream_cosine,
    compute_mean_zero_energy,
    compute_mean_zero_norm_ratio,
)
from experiments.lm_5090_next_runs import (
    build_transport_history,
    runtime_provenance,
)
~~~

Add:

~~~python
def test_schema2_norm_and_energy_have_square_relationship(self):
    torch.manual_seed(37)
    X = torch.randn(4, 2, 3, 5)
    norm_ratio = compute_mean_zero_norm_ratio(X)
    energy = compute_mean_zero_energy(X)
    self.assertAlmostEqual(energy, norm_ratio ** 2, places=6)
    self.assertTrue(torch.isfinite(torch.tensor(
        compute_centered_stream_cosine(X)
    )))

def test_transport_history_is_aligned_by_step_and_tokens(self):
    diagnostics = DiagnosticsCollector()
    report = {
        "n_streams": 4,
        "num_transports": 1,
        "steps": [{
            "singular_values": [0.7, 0.8, 0.9],
            "sv_min": 0.7,
            "sv_mean": 0.8,
            "sv_max": 0.9,
        }],
        "prefix": [{
            "composite_singular_values": [0.7, 0.8, 0.9],
            "composite_sv_min": 0.7,
            "composite_sv_mean": 0.8,
            "composite_sv_max": 0.9,
        }],
        "final": {
            "composite_singular_values": [0.7, 0.8, 0.9],
            "composite_sv_min": 0.7,
            "composite_sv_mean": 0.8,
            "composite_sv_max": 0.9,
            "row_sum_error_max": 0.03,
        },
    }
    diagnostics.record_snapshot("transport", {
        "optimizer_step": 20,
        "tokens_processed": 2000,
        "transport": report,
        "gate_values": {
            "readout_lambda": 0.02,
            "injection_lambda": 0.03,
            "final_readout_lambda": 0.04,
        },
    })
    history = build_transport_history(diagnostics)
    self.assertEqual(history[0]["optimizer_step"], 20)
    self.assertEqual(history[0]["tokens_processed"], 2000)
    self.assertEqual(
        len(history[0]["transport"]["steps"][0]["singular_values"]),
        3,
    )
    self.assertEqual(
        len(history[0]["transport"]["final"][
            "composite_singular_values"
        ]),
        3,
    )
    self.assertAlmostEqual(
        history[0]["gate_values"]["readout_lambda"],
        0.02,
    )

def test_runtime_provenance_matches_constructed_isohc(self):
    cfg = build_preset_configs(
        preset="run0",
        methods=["isohc"],
        output_dir="outputs/test",
        batch_size=2,
        use_compile=False,
    )[0]
    cfg["mixing_kwargs"] = {
        "ns_steps": 7,
        "use_svd": False,
        "svd_fallback": True,
    }
    model = create_model(cfg, 128, torch.device("cpu"))
    provenance = runtime_provenance(
        model,
        cfg,
        torch.device("cpu"),
    )
    self.assertEqual(model.attn_mixings[0].ns_steps, 7)
    self.assertTrue(model.attn_mixings[0].svd_fallback)
    self.assertEqual(
        provenance["projection_internal_dtype"],
        "torch.float64",
    )
    self.assertEqual(provenance["ns_steps"], 7)
    self.assertFalse(provenance["use_svd"])
    self.assertTrue(provenance["svd_fallback"])
    self.assertIsNone(provenance["amp_dtype"])
~~~

- [ ] **Step 2: Run and confirm failure**

Run:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_schema2_norm_and_energy_have_square_relationship \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_transport_history_is_aligned_by_step_and_tokens \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_runtime_provenance_matches_constructed_isohc -v
~~~

Expected: FAIL because the new functions and history builder do not exist.

- [ ] **Step 3: Implement unambiguous state metrics**

Use:

~~~python
def compute_mean_zero_norm_ratio(X, eps=1e-12):
    mean = X.mean(dim=0, keepdim=True)
    X_perp = X - mean
    return (
        torch.norm(X_perp, p="fro")
        / (torch.norm(X, p="fro") + eps)
    ).item()

def compute_mean_zero_energy(X, eps=1e-12):
    ratio = compute_mean_zero_norm_ratio(X, eps=eps)
    return ratio * ratio

def compute_centered_stream_cosine(X, eps=1e-12):
    s = X.shape[0]
    if s <= 1:
        return 1.0
    centered = X - X.mean(dim=0, keepdim=True)
    flat = centered.reshape(s, -1)
    normed = flat / (torch.norm(flat, dim=1, keepdim=True) + eps)
    cosine = normed @ normed.T
    mask = ~torch.eye(s, dtype=torch.bool, device=X.device)
    return cosine[mask].mean().item()
~~~

Keep compute_stream_cosine unchanged for schema-1 comparison.

- [ ] **Step 4: Add complete periodic geometry snapshots to HC diagnostics**

Add a separate structured channel to DiagnosticsCollector so exact lists are not squeezed into its scalar-only history:

~~~python
# in __init__
self.snapshots = defaultdict(list)

def record_snapshot(self, name, value):
    if self.should_collect():
        self.snapshots[name].append(value)

# in clear
self.snapshots.clear()
~~~

Import collect_transport_report inside collect_hc_diagnostics to avoid widening module initialization:

~~~python
if hasattr(model, "get_named_mixing_matrices"):
    from .transport_analysis import collect_transport_report

    transport_report = collect_transport_report(model)
    results["transport_report"] = transport_report
    final = transport_report.get("final", {})
    for key in (
        "composite_sv_min",
        "composite_sv_mean",
        "composite_sv_max",
        "product_sv_min",
        "product_sv_mean",
        "product_sv_max",
        "mean_to_perp_leakage_max",
        "perp_to_mean_leakage_max",
        "row_sum_error_max",
        "col_sum_error_max",
    ):
        if key in final:
            results[f"transport/{key}"] = final[key]

for result_key, attribute in (
    ("readout_lambda", "readout_lambda"),
    ("injection_lambda", "injection_lambda"),
    ("final_readout_lambda", "readout_final_lambda"),
):
    parameter = getattr(model, attribute, None)
    if parameter is not None:
        results[f"gates/{result_key}"] = (
            parameter.detach().float().item()
        )
~~~

In train_epoch, use the collector's global optimizer-step counter rather than the per-epoch local step. Use the same global token accounting already used by run_experiment. Replace the current HC recording block with the following shape:

~~~python
diagnostics.step()
optimizer_step = diagnostics.step_count
tokens_processed = min(
    optimizer_step * tokens_per_step,
    total_tokens_target,
)
diagnostics.record(
    optimizer_step=optimizer_step,
    tokens_processed=tokens_processed,
    train_loss=batch_loss,
    lr=lr,
    grad_norm=get_total_grad_norm(model),
)

if diagnostics.should_collect():
    hc_stats = collect_hc_diagnostics(model)
    transport_report = hc_stats.pop("transport_report", None)
    gate_snapshot = {
        key: hc_stats[f"gates/{key}"]
        for key in (
            "readout_lambda",
            "injection_lambda",
            "final_readout_lambda",
        )
        if f"gates/{key}" in hc_stats
    }
    diagnostics.record_dict("hc", hc_stats)
    if transport_report is not None:
        diagnostics.record_snapshot("transport", {
            "optimizer_step": optimizer_step,
            "tokens_processed": tokens_processed,
            "transport": transport_report,
            "gate_values": gate_snapshot,
        })
~~~

Do not call collect_hc_diagnostics a second time in the same step. Add diagnostic_snapshots=dict(diagnostics.snapshots) beside diagnostics in the final checkpoint payload so the structured history survives checkpoint-only inspection.

- [ ] **Step 5: Build aligned history and provenance**

Add to experiments/lm_5090_next_runs.py:

~~~python
def build_transport_history(diagnostics):
    return list(diagnostics.snapshots.get("transport", []))

def runtime_provenance(model, config, device):
    method = config["method"]
    is_iso = method in {"isohc", "scaled-isohc"}
    mixing = model.attn_mixings[0] if is_iso else None
    return {
        "parameter_dtype": str(next(model.parameters()).dtype),
        "amp_dtype": (
            "torch.bfloat16"
            if config.get("use_amp", True) and device.type == "cuda"
            else None
        ),
        "projection_internal_dtype": "torch.float64" if is_iso else None,
        "ns_steps": getattr(mixing, "ns_steps", None),
        "use_svd": getattr(mixing, "use_svd", None),
        "svd_fallback": getattr(mixing, "svd_fallback", None),
        "use_compile": bool(config.get("use_compile", False)),
        "compile_mode": config.get("compile_mode"),
    }

def gate_values(model):
    values = {}
    for result_key, attribute in (
        ("readout_lambda", "readout_lambda"),
        ("injection_lambda", "injection_lambda"),
        ("final_readout_lambda", "readout_final_lambda"),
    ):
        parameter = getattr(model, attribute, None)
        if parameter is not None:
            values[result_key] = parameter.detach().float().item()
    return values
~~~

Capture initial_transport and initial_gate_values immediately after create_model and before training. Guard transport collection with hasattr(model, "get_named_mixing_matrices") so baseline and head-mixing runs retain their existing behavior. On success, store:

~~~python
"metric_schema_version": 2,
"initial_transport": initial_transport,
"initial_gate_values": initial_gate_values,
"training_transport_history": build_transport_history(
    results["diagnostics"]
),
"final_transport": posthoc.get("transport_complement"),
"final_gate_values": gate_values(model),
"runtime_provenance": runtime_provenance(model, config, device),
~~~

Update collect_posthoc_diagnostics with explicit, non-overlapping schema-2 keys:

~~~python
norm_ratios = [compute_mean_zero_norm_ratio(state) for state in states]
energies = [compute_mean_zero_energy(state) for state in states]
cosines = [compute_stream_cosine(state) for state in states]
centered_cosines = [
    compute_centered_stream_cosine(state)
    for state in states
]
result.update({
    "mean_zero_norm_ratio_initial": norm_ratios[0],
    "mean_zero_norm_ratio_final": norm_ratios[-1],
    "mean_zero_norm_ratio_curve": norm_ratios,
    "mean_zero_energy_initial": energies[0],
    "mean_zero_energy_final": energies[-1],
    "mean_zero_energy_curve": energies,
    "stream_cosine_initial": cosines[0],
    "stream_cosine_final": cosines[-1],
    "stream_cosine_curve": cosines,
    "centered_stream_cosine_initial": centered_cosines[0],
    "centered_stream_cosine_final": centered_cosines[-1],
    "centered_stream_cosine_curve": centered_cosines,
})
~~~

Do not reinterpret a schema-1 mean_zero_energy artifact as squared energy; schema version remains the comparison boundary.

- [ ] **Step 6: Run focused tests and the gradient-accumulation regression**

Run the Step 2 command, then:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_run_experiment_respects_gradient_accumulation_steps -v
~~~

Expected: all listed tests PASS.

- [ ] **Step 7: Commit Task 3**

~~~bash
git add lm/diagnostics.py lm/train.py experiments/lm_5090_next_runs.py tests/test_lm_next_phase_contracts.py
git commit -m "feat: record schema2 transport trajectories"
~~~

### Task 4: Add Paired Persistent Complement Interventions

**Files:**
- Modify: experiments/analyze_lm_mechanisms.py:1-381
- Test: tests/test_lm_next_phase_contracts.py

**Interfaces:**
- Consumes: existing stream_intervention mapping and validation loader.
- Produces: evaluate_paired_intervention, persistent_complement_scale_curve, and report field persistent_complement_scale.

- [ ] **Step 1: Write a failing identity-intervention test**

Add imports:

~~~python
from torch.utils.data import DataLoader, TensorDataset
from experiments.analyze_lm_mechanisms import (
    evaluate_paired_intervention,
    persistent_complement_scale_curve,
)
~~~

Add:

~~~python
def test_persistent_scale_one_is_an_exact_paired_noop(self):
    torch.manual_seed(41)
    model = TwoBranchHCTransformer(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        n_streams=4,
        context_length=8,
        mixing_type="identity",
        use_flash=True,
    )
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    metrics = evaluate_paired_intervention(
        model,
        loader,
        torch.device("cpu"),
        use_amp=False,
        max_batches=1,
        stream_intervention={
            "state_index": list(range(1, 2 * model.num_layers + 1)),
            "mode": "scale_perp",
            "scale": 1.0,
        },
    )
    self.assertAlmostEqual(metrics["delta_nll"], 0.0, places=7)
    self.assertAlmostEqual(metrics["mean_token_kl"], 0.0, places=7)
    self.assertAlmostEqual(metrics["top1_change_rate"], 0.0, places=7)

    curve = persistent_complement_scale_curve(
        model,
        loader,
        torch.device("cpu"),
        start_indices=[1],
        scales=[0.0, 1.0],
        use_amp=False,
        max_batches=1,
    )
    self.assertEqual(len(curve), 2)
    self.assertEqual(curve[0]["active_state_count"], 4)
    for row in curve:
        values = torch.tensor([
            row["delta_nll"],
            row["mean_token_kl"],
            row["top1_change_rate"],
        ])
        self.assertTrue(torch.isfinite(values).all())
~~~

- [ ] **Step 2: Run and confirm failure**

Run:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_persistent_scale_one_is_an_exact_paired_noop -v
~~~

Expected: FAIL because paired functions do not exist.

- [ ] **Step 3: Implement paired metrics on the same batches**

Import torch.nn.functional as F. Add:

~~~python
@torch.no_grad()
def evaluate_paired_intervention(
    model,
    val_loader,
    device,
    use_amp=True,
    max_batches=4,
    stream_intervention=None,
):
    model.eval()
    base_nll_sum = 0.0
    intervention_nll_sum = 0.0
    kl_sum = 0.0
    changed = 0
    token_count = 0
    batch_count = 0

    for batch_index, (x, y) in enumerate(val_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_autocast(device, enabled=use_amp):
            base_logits, _ = model(x)
            intervention_logits, _ = model(
                x,
                stream_intervention=stream_intervention,
            )

        valid = y.ne(-100)
        count = int(valid.sum().item())
        if count:
            base_float = base_logits.float()
            intervention_float = intervention_logits.float()
            base_nll_sum += F.cross_entropy(
                base_float.reshape(-1, base_float.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            intervention_nll_sum += F.cross_entropy(
                intervention_float.reshape(-1, intervention_float.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            base_logp = F.log_softmax(base_float, dim=-1)
            intervention_logp = F.log_softmax(
                intervention_float,
                dim=-1,
            )
            token_kl = (
                base_logp.exp() * (base_logp - intervention_logp)
            ).sum(dim=-1)
            kl_sum += token_kl[valid].sum().item()
            changed += (
                base_float.argmax(dim=-1)[valid]
                != intervention_float.argmax(dim=-1)[valid]
            ).sum().item()
            token_count += count
        batch_count += 1
        if max_batches and batch_index >= max_batches - 1:
            break

    denominator = max(token_count, 1)
    base_nll = base_nll_sum / denominator
    intervention_nll = intervention_nll_sum / denominator
    return {
        "base_nll": base_nll,
        "intervention_nll": intervention_nll,
        "delta_nll": intervention_nll - base_nll,
        "base_ppl": math.exp(min(base_nll, 20)),
        "intervention_ppl": math.exp(min(intervention_nll, 20)),
        "mean_token_kl": kl_sum / denominator,
        "top1_change_rate": changed / denominator,
        "tokens": token_count,
        "batches": batch_count,
    }
~~~

- [ ] **Step 4: Build the persistent gamma curve and expose CLI values**

Add:

~~~python
def persistent_complement_scale_curve(
    model,
    val_loader,
    device,
    start_indices,
    scales,
    use_amp=True,
    max_batches=4,
):
    if not hasattr(model, "num_layers"):
        return []
    final_state = 2 * model.num_layers
    rows = []
    for start in start_indices:
        if not 0 <= start <= final_state:
            raise ValueError(
                f"start state {start} is outside [0, {final_state}]"
            )
        active_states = list(range(start, final_state + 1))
        for scale in scales:
            if not 0.0 <= scale <= 1.0:
                raise ValueError("intervention scales must be in [0, 1]")
            metrics = evaluate_paired_intervention(
                model,
                val_loader,
                device,
                use_amp=use_amp,
                max_batches=max_batches,
                stream_intervention={
                    "state_index": active_states,
                    "mode": "scale_perp",
                    "scale": float(scale),
                },
            )
            rows.append({
                "start_state": start,
                "scale": float(scale),
                "active_state_count": len(active_states),
                **metrics,
            })
    return rows
~~~

In analyze_run, select start states with intervention_stride and save the rows under persistent_complement_scale. Add:

~~~python
parser.add_argument(
    "--intervention_scales",
    nargs="+",
    type=float,
    default=[0.0, 0.25, 0.5, 0.75, 1.0],
)
~~~

Update write_markdown with the maximum absolute delta NLL and maximum KL from this curve. Keep single-state complement_removal unchanged and label it weak.

- [ ] **Step 5: Run focused and existing posthoc contract tests**

Run Step 2, then:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_two_branch_forward_supports_complement_removal_and_stream_grad_capture -v
~~~

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

~~~bash
git add experiments/analyze_lm_mechanisms.py tests/test_lm_next_phase_contracts.py
git commit -m "feat: add persistent complement interventions"
~~~

### Task 5: Define Deterministic Causal Suites and Depth Presets

**Files:**
- Create: experiments/hc_causal_controls.py
- Modify: tests/test_lm_next_phase_contracts.py

**Interfaces:**
- Consumes: Task 1 mixers, Task 2 build_preset_configs, Task 3 run summaries.
- Produces: symmetric_birkhoff_gain, identity_hc_parameter_count, DEPTH_PRESETS, build_geometry_report, build_suite_configs, and build_depth_summary.

- [ ] **Step 1: Write failing math and suite-shape tests**

Add imports:

~~~python
import math

from experiments.hc_causal_controls import (
    DEPTH_PRESETS,
    build_depth_summary,
    build_suite_configs,
    identity_hc_parameter_count,
    symmetric_birkhoff_gain,
)
~~~

Add:

~~~python
def test_symmetric_birkhoff_gain_matches_noise_free_sinkhorn(self):
    expected = symmetric_birkhoff_gain(
        n_streams=4,
        diag_bias=4.0,
        temperature=1.0,
    )
    mixer = MHCMixing(
        4,
        diag_bias=4.0,
        temperature=1.0,
        noise_std=0.0,
        sinkhorn_iters=10,
    )
    actual = complement_spectrum(mixer())["sv_mean"]
    self.assertAlmostEqual(actual, expected, places=6)

def test_depth_presets_match_current_parameter_formula(self):
    reference = DEPTH_PRESETS[48]["parameters"]
    for layers, preset in DEPTH_PRESETS.items():
        count = identity_hc_parameter_count(
            vocab_size=50257,
            context_length=512,
            n_streams=4,
            num_layers=layers,
            d_model=preset["d_model"],
        )
        self.assertEqual(count, preset["parameters"])
        self.assertLessEqual(abs(count - reference) / reference, 0.05)
        self.assertEqual(preset["num_transports"], 2 * layers)

def test_causal_suite_config_counts_and_labels(self):
    smoke = build_suite_configs(
        suite="p0-smoke",
        output_dir="outputs/test",
        dataset="random",
        total_tokens=4096,
        batch_size=2,
        seed=0,
        use_compile=False,
    )
    train = build_suite_configs(
        suite="p0-train",
        output_dir="outputs/test",
        dataset="random",
        total_tokens=4096,
        batch_size=2,
        seed=0,
        use_compile=False,
    )
    depth = build_suite_configs(
        suite="p0-depth",
        output_dir="outputs/test",
        dataset="random",
        total_tokens=4096,
        batch_size=2,
        seed=0,
        use_compile=False,
    )
    self.assertEqual(len(smoke), 6)
    self.assertEqual(len(train), 14)
    self.assertEqual(len(depth), 25)
    self.assertEqual(
        len({cfg["experiment_variant"] for cfg in depth}),
        25,
    )
    self.assertTrue(
        all(cfg["metric_schema_version"] == 2 for cfg in train)
    )

def test_depth_summary_uses_measured_transport_logs(self):
    report = build_depth_summary([{
        "success": True,
        "config": {
            "method": "static-birkhoff-hc",
            "experiment_variant": "depth1_test",
            "num_layers": 1,
            "d_model": 32,
            "target_parameters": 123,
        },
        "final_transport": {
            "num_transports": 2,
            "steps": [
                {"sv_min": 0.5, "sv_max": 0.8},
                {"sv_min": 0.25, "sv_max": 0.9},
            ],
            "final": {
                "composite_sv_min": 0.1,
                "composite_sv_max": 0.7,
            },
        },
    }])
    row = report["rows"][0]
    self.assertEqual(row["num_layers"], 1)
    self.assertEqual(row["num_transports"], 2)
    self.assertAlmostEqual(
        row["log_composite_sv_min"],
        math.log(0.1),
    )
    self.assertAlmostEqual(
        row["sum_step_log_sv_min"],
        math.log(0.5) + math.log(0.25),
    )
~~~

- [ ] **Step 2: Run and confirm failure**

Run:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_symmetric_birkhoff_gain_matches_noise_free_sinkhorn \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_depth_presets_match_current_parameter_formula \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_causal_suite_config_counts_and_labels \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_depth_summary_uses_measured_transport_logs -v
~~~

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Add exact constants and pure helpers**

Start experiments/hc_causal_controls.py with:

~~~python
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.lm_5090_next_runs import (
    build_preset_configs,
    run_single,
)
from lm.mixing import MHCMixing
from lm.transport_analysis import collect_transport_report


DEPTH_PRESETS = {
    24: {
        "d_model": 704,
        "num_heads": 8,
        "parameters": 178_516_487,
        "num_transports": 48,
    },
    48: {
        "d_model": 512,
        "num_heads": 8,
        "parameters": 177_041_159,
        "num_transports": 96,
    },
    72: {
        "d_model": 416,
        "num_heads": 8,
        "parameters": 170_703_431,
        "num_transports": 144,
    },
    96: {
        "d_model": 368,
        "num_heads": 8,
        "parameters": 174_765_479,
        "num_transports": 192,
    },
    128: {
        "d_model": 320,
        "num_heads": 8,
        "parameters": 173_618_055,
        "num_transports": 256,
    },
}


def symmetric_birkhoff_gain(n_streams, diag_bias, temperature):
    if n_streams < 2:
        raise ValueError("n_streams must be at least 2")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    diagonal = math.exp(diag_bias / temperature)
    return (diagonal - 1.0) / (diagonal + n_streams - 1.0)


def identity_hc_parameter_count(
    vocab_size,
    context_length,
    n_streams,
    num_layers,
    d_model,
):
    per_layer = 12 * d_model * d_model + 2 * d_model + 4 * n_streams
    shared = (
        vocab_size * d_model
        + context_length * d_model
        + n_streams * d_model
        + d_model
        + n_streams
        + 3
    )
    return num_layers * per_layer + shared


def build_depth_summary(summaries, eps=1e-300):
    rows = []
    for summary in summaries:
        config = summary["config"]
        row = {
            "success": bool(summary.get("success", False)),
            "method": config["method"],
            "experiment_variant": config["experiment_variant"],
            "num_layers": int(config["num_layers"]),
            "d_model": int(config["d_model"]),
            "target_parameters": config.get("target_parameters"),
        }
        if not row["success"]:
            row["error"] = summary.get("error")
            rows.append(row)
            continue

        transport = summary["final_transport"]
        final = transport["final"]
        steps = transport["steps"]
        row.update({
            "num_transports": int(transport["num_transports"]),
            "log_composite_sv_min": math.log(max(
                float(final["composite_sv_min"]),
                eps,
            )),
            "log_composite_sv_max": math.log(max(
                float(final["composite_sv_max"]),
                eps,
            )),
            "sum_step_log_sv_min": sum(
                math.log(max(float(step["sv_min"]), eps))
                for step in steps
            ),
            "sum_step_log_sv_max": sum(
                math.log(max(float(step["sv_max"]), eps))
                for step in steps
            ),
        })
        rows.append(row)
    return {
        "metric_schema_version": 2,
        "rows": rows,
    }
~~~

- [ ] **Step 4: Build the geometry grid without an LM**

Add:

~~~python
@torch.no_grad()
def build_geometry_report(n_streams, num_transports, seed):
    rows = []
    grid = itertools.product(
        (2.0, 4.0, 6.0, 8.0),
        (0.5, 1.0, 2.0),
        (0.0, 0.01),
        (5, 10, 20),
    )
    for diag_bias, temperature, noise_std, sinkhorn_iters in grid:
        torch.manual_seed(seed)
        matrices = []
        for index in range(num_transports):
            H = MHCMixing(
                n_streams,
                sinkhorn_iters=sinkhorn_iters,
                temperature=temperature,
                diag_bias=diag_bias,
                noise_std=noise_std,
            )()
            matrices.append({
                "index": index,
                "branch": "transport",
                "layer": index,
                "H": H,
            })
        transport = collect_transport_report(matrices)
        analytic = (
            symmetric_birkhoff_gain(
                n_streams,
                diag_bias,
                temperature,
            )
            if noise_std == 0.0
            else None
        )
        rows.append({
            "diag_bias": diag_bias,
            "temperature": temperature,
            "noise_std": noise_std,
            "sinkhorn_iters": sinkhorn_iters,
            "n_streams": n_streams,
            "num_transports": num_transports,
            "analytic_single_step_gain": analytic,
            "analytic_composite_gain": (
                analytic ** num_transports
                if analytic is not None
                else None
            ),
            "measured": transport,
        })
    return {
        "metric_schema_version": 2,
        "seed": seed,
        "rows": rows,
    }
~~~

- [ ] **Step 5: Build unique training configurations**

Use this exact configuration builder:

~~~python
def _birkhoff_kwargs(diag_bias, identity_blend=1.0):
    return {
        "diag_bias": float(diag_bias),
        "temperature": 1.0,
        "noise_std": 0.0,
        "sinkhorn_iters": 10,
        "identity_blend": float(identity_blend),
    }


def build_suite_configs(
    suite,
    output_dir,
    dataset,
    total_tokens,
    batch_size,
    seed,
    use_compile,
):
    if suite not in {"p0-smoke", "p0-train", "p0-depth"}:
        raise ValueError(f"Unknown training suite: {suite}")

    configs = []

    def add(
        preset,
        method,
        variant,
        mixing_kwargs=None,
        freeze_mixing=False,
        structural=None,
    ):
        config = build_preset_configs(
            preset=preset,
            methods=[method],
            output_dir=output_dir,
            total_tokens=total_tokens,
            batch_size=batch_size,
            seed=seed,
            dataset=dataset,
            use_compile=use_compile,
        )[0]
        config.update({
            "experiment_variant": variant,
            "mixing_kwargs": dict(mixing_kwargs or {}),
            "lambda_a": 0.01,
            "lambda_b": 0.01,
            "freeze_mixing": bool(freeze_mixing),
            "metric_schema_version": 2,
            "save_dir": str(
                Path(output_dir) / f"{variant}_seed{seed}"
            ),
        })
        if structural:
            config.update(structural)
        if suite == "p0-smoke":
            config["diagnostics_every"] = 1
        configs.append(config)

    if suite == "p0-smoke":
        preset = "run0"
        gain = symmetric_birkhoff_gain(4, 4.0, 1.0)
        add(preset, "identity-hc", "identity_hc")
        add(preset, "isohc", "isohc")
        add(
            preset,
            "static-birkhoff-hc",
            "birkhoff_d4_trainable",
            _birkhoff_kwargs(4.0),
        )
        add(
            preset,
            "static-birkhoff-hc",
            "birkhoff_d4_frozen",
            _birkhoff_kwargs(4.0),
            freeze_mixing=True,
        )
        add(
            preset,
            "static-birkhoff-hc",
            "birkhoff_d4_blend05",
            _birkhoff_kwargs(4.0, identity_blend=0.5),
        )
        add(
            preset,
            "scaled-isohc",
            "scaled_isohc_match_d4",
            {"complement_scale": gain},
        )
        return configs

    if suite == "p0-train":
        preset = "fe-deep-48l-512"
        add(preset, "identity-hc", "identity_hc")
        add(preset, "isohc", "isohc")
        for bias in (2.0, 4.0, 6.0, 8.0):
            label = f"d{int(bias)}"
            add(
                preset,
                "static-birkhoff-hc",
                f"birkhoff_{label}_trainable",
                _birkhoff_kwargs(bias),
            )
            add(
                preset,
                "static-birkhoff-hc",
                f"birkhoff_{label}_frozen",
                _birkhoff_kwargs(bias),
                freeze_mixing=True,
            )
            add(
                preset,
                "scaled-isohc",
                f"scaled_isohc_match_{label}",
                {
                    "complement_scale": symmetric_birkhoff_gain(
                        4,
                        bias,
                        1.0,
                    )
                },
            )
        return configs

    for layers, depth in DEPTH_PRESETS.items():
        preset = "fe-deep-48l-512"
        structural = {
            "num_layers": layers,
            "d_model": depth["d_model"],
            "num_heads": depth["num_heads"],
            "target_parameters": depth["parameters"],
            "expected_num_transports": depth["num_transports"],
        }
        prefix = f"depth{layers}"
        gain = symmetric_birkhoff_gain(4, 4.0, 1.0)
        add(
            preset,
            "identity-hc",
            f"{prefix}_identity_hc",
            structural=structural,
        )
        add(
            preset,
            "isohc",
            f"{prefix}_isohc",
            structural=structural,
        )
        add(
            preset,
            "static-birkhoff-hc",
            f"{prefix}_birkhoff_d4_trainable",
            _birkhoff_kwargs(4.0),
            structural=structural,
        )
        add(
            preset,
            "static-birkhoff-hc",
            f"{prefix}_birkhoff_d4_frozen",
            _birkhoff_kwargs(4.0),
            freeze_mixing=True,
            structural=structural,
        )
        add(
            preset,
            "scaled-isohc",
            f"{prefix}_scaled_isohc_match_d4",
            {"complement_scale": gain},
            structural=structural,
        )
    return configs
~~~

This deliberately keeps the matched training controls noise-free. The geometry suite measures noise_std 0.01 separately.

- [ ] **Step 6: Run the focused tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

~~~bash
git add experiments/hc_causal_controls.py tests/test_lm_next_phase_contracts.py
git commit -m "feat: define HC causal experiment suites"
~~~

### Task 6: Add Safe CLI Orchestration and Posthoc Handoff

**Files:**
- Modify: experiments/hc_causal_controls.py
- Test: tests/test_lm_next_phase_contracts.py

**Interfaces:**
- Consumes: build_geometry_report and build_suite_configs from Task 5.
- Produces: geometry_summary.json, suite_summary.json, depth_summary.json for p0-depth, safe no-overwrite behavior, optional posthoc subprocess.

- [ ] **Step 1: Add failing output-safety tests**

Add:

~~~python
def test_completed_causal_run_cannot_be_overwritten(self):
    from experiments.hc_causal_controls import ensure_run_is_new

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "variant"
        run_dir.mkdir()
        (run_dir / "run_summary.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            ensure_run_is_new(run_dir)
~~~

Add pathlib.Path to test imports.

- [ ] **Step 2: Run and confirm failure**

Run:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_completed_causal_run_cannot_be_overwritten -v
~~~

Expected: FAIL because ensure_run_is_new does not exist.

- [ ] **Step 3: Implement output safety and JSON writing**

Add:

~~~python
def ensure_run_is_new(run_dir):
    summary = Path(run_dir) / "run_summary.json"
    if summary.exists():
        raise FileExistsError(
            f"Completed run already exists: {summary}"
        )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")
    return path
~~~

- [ ] **Step 4: Implement CLI and posthoc handoff**

Add:

~~~python
def run_posthoc(run_dirs, args):
    analyzer = Path(__file__).with_name("analyze_lm_mechanisms.py")
    command = [
        sys.executable,
        "-u",
        str(analyzer),
        "--run_dirs",
        *[str(path) for path in run_dirs],
        "--output_dir",
        str(Path(args.output_dir) / "posthoc"),
        "--dataset",
        args.dataset,
        "--eval_batches",
        "4",
        "--intervention_stride",
        "8",
        "--intervention_scales",
        "0",
        "0.25",
        "0.5",
        "0.75",
        "1",
        "--num_workers",
        "0",
    ]
    if args.val_cache_path:
        command.extend([
            "--val_cache_path",
            args.val_cache_path,
        ])
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HC causal-control experiment suites"
    )
    parser.add_argument(
        "--suite",
        choices=[
            "geometry",
            "p0-smoke",
            "p0-train",
            "p0-depth",
        ],
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="random")
    parser.add_argument("--total_tokens", type=int, default=262_144)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_streams", type=int, default=4)
    parser.add_argument("--num_transports", type=int, default=96)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--auto_batch", action="store_true")
    parser.add_argument("--memory_target_gb", type=float, default=30.0)
    parser.add_argument("--train_cache_path")
    parser.add_argument("--val_cache_path")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--skip_posthoc", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.suite == "geometry":
        path = output_dir / "geometry_summary.json"
        if path.exists():
            raise FileExistsError(
                f"Completed geometry report already exists: {path}"
            )
        report = build_geometry_report(
            n_streams=args.n_streams,
            num_transports=args.num_transports,
            seed=args.seed,
        )
        write_json(path, report)
        print(f"Wrote {path}")
        return

    if (
        args.suite in {"p0-train", "p0-depth"}
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"{args.suite} requires CUDA; use p0-smoke for CPU checks"
        )

    summary_path = output_dir / "suite_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Completed suite summary already exists: {summary_path}"
        )

    configs = build_suite_configs(
        suite=args.suite,
        output_dir=args.output_dir,
        dataset=args.dataset,
        total_tokens=args.total_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        use_compile=not args.no_compile,
    )
    summaries = []
    run_dirs = []
    for config in configs:
        config["train_cache_path"] = args.train_cache_path
        config["val_cache_path"] = args.val_cache_path
        config["vocab_size"] = args.vocab_size
        ensure_run_is_new(config["save_dir"])
        summaries.append(
            run_single(
                config,
                auto_batch=args.auto_batch,
                memory_target_gb=args.memory_target_gb,
            )
        )
        run_dirs.append(Path(config["save_dir"]))

    write_json(summary_path, summaries)
    print(f"Wrote {summary_path}")
    if args.suite == "p0-depth":
        depth_path = output_dir / "depth_summary.json"
        write_json(depth_path, build_depth_summary(summaries))
        print(f"Wrote {depth_path}")
    if not args.skip_posthoc:
        run_posthoc(run_dirs, args)


if __name__ == "__main__":
    main()
~~~

- [ ] **Step 5: Run the safety test and CLI help**

Run Step 2, then:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  experiments/hc_causal_controls.py --help
~~~

Expected: test PASS and help lists all four suites.

- [ ] **Step 6: Commit Task 6**

~~~bash
git add experiments/hc_causal_controls.py tests/test_lm_next_phase_contracts.py
git commit -m "feat: orchestrate HC causal experiment runs"
~~~

### Task 7: Document the New Evidence Boundary and Commands

**Files:**
- Modify: README.md
- Modify: experiments/README.md
- Modify: agent.md

**Interfaces:**
- Consumes: Task 6 CLI.
- Produces: one local smoke command, one geometry command, one server P0 command, and explicit static-baseline wording.

- [ ] **Step 1: Add a documentation contract test**

Add:

~~~python
def test_canonical_docs_name_static_birkhoff_and_causal_runner(self):
    for path in ("README.md", "experiments/README.md", "agent.md"):
        text = Path(path).read_text()
        self.assertIn("static-birkhoff-hc", text)
        self.assertIn("experiments/hc_causal_controls.py", text)
~~~

- [ ] **Step 2: Run and confirm failure**

Run:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest \
  tests.test_lm_next_phase_contracts.LMNextPhaseContractTests.test_canonical_docs_name_static_birkhoff_and_causal_runner -v
~~~

Expected: FAIL because docs retain the old label and omit the new runner.

- [ ] **Step 3: Update README.md**

Document:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  experiments/hc_causal_controls.py \
  --suite geometry \
  --output_dir outputs/hc_geometry
~~~

State that mhc remains a legacy checkpoint/CLI alias and all new evidence uses static-birkhoff-hc.

- [ ] **Step 4: Update experiments/README.md**

Document the CPU smoke:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  experiments/hc_causal_controls.py \
  --suite p0-smoke \
  --output_dir outputs/hc_p0_smoke \
  --dataset random \
  --total_tokens 4096 \
  --batch_size 2 \
  --no_compile
~~~

Document the server P0 train command with caches under /root/autodl-tmp/isoHC and no live network data.

- [ ] **Step 5: Update agent.md**

Add the new P0 entrypoint, preserve the existing 0525 pipeline as historical reproducibility, and state:

~~~text
static-birkhoff-hc is the current static Sinkhorn proxy.
It is not faithful dynamic mHC.
Do not generalize static-proxy results to official mHC without the P2 parity implementation.
~~~

- [ ] **Step 6: Run the docs test and diff check**

Run Step 2, then:

~~~bash
git diff --check
~~~

Expected: test PASS and no diff errors.

- [ ] **Step 7: Commit Task 7**

~~~bash
git add README.md experiments/README.md agent.md tests/test_lm_next_phase_contracts.py
git commit -m "docs: add HC causal experiment commands"
~~~

### Task 8: Run Full Verification and Produce a CPU Artifact Smoke

**Files:**
- Verify: all files changed in Tasks 1-7
- Generate outside git: outputs/hc_p0_smoke
- Do not commit: outputs, checkpoints, caches, logs

**Interfaces:**
- Consumes: complete P0 implementation.
- Produces: passing regression suite, validated geometry JSON, validated CPU smoke JSON, exact unverified list.

- [ ] **Step 1: Run syntax checks without writing bytecode**

Run with bytecode redirected outside the repository:

~~~bash
PYTHONPYCACHEPREFIX=/tmp/iso_hc_pycache \
"$HOME/miniconda3/envs/aidemo/bin/python" -m compileall -q \
  lm experiments tests
~~~

Expected: exit 0 and no repository-local __pycache__ directories.

- [ ] **Step 2: Run the full contract suite**

Run:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest tests.test_lm_next_phase_contracts tests.test_stage1_contracts -q
~~~

Expected: all tests PASS. Record the exact test count from output.

- [ ] **Step 3: Verify measured depth counts against the current model**

Run the five configurations sequentially with the aidemo interpreter and assert each count equals DEPTH_PRESETS. The command must delete each model before constructing the next to bound memory. Expected counts:

~~~text
24 704 178516487
48 512 177041159
72 416 170703431
96 368 174765479
128 320 173618055
~~~

- [ ] **Step 4: Run and validate the geometry artifact**

Use a fresh output directory. If outputs/hc_geometry_verify already exists, choose a new suffix; do not delete prior evidence.

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  experiments/hc_causal_controls.py \
  --suite geometry \
  --output_dir outputs/hc_geometry_verify \
  --n_streams 4 \
  --num_transports 96 \
  --seed 0
~~~

Validate with the standard library:

~~~bash
"$HOME/miniconda3/envs/aidemo/bin/python" -c '
import json
from pathlib import Path
data = json.loads(Path("outputs/hc_geometry_verify/geometry_summary.json").read_text())
assert data["metric_schema_version"] == 2
assert len(data["rows"]) == 72
noise_free = [row for row in data["rows"] if row["noise_std"] == 0.0]
assert all(row["analytic_single_step_gain"] is not None for row in noise_free)
print(len(data["rows"]))
'
~~~

Expected output: 72.

- [ ] **Step 5: Run the CPU P0 smoke artifact**

Use a fresh output directory:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  experiments/hc_causal_controls.py \
  --suite p0-smoke \
  --output_dir outputs/hc_p0_smoke_verify \
  --dataset random \
  --total_tokens 4096 \
  --batch_size 2 \
  --seed 0 \
  --no_compile
~~~

Expected: six successful run summaries plus posthoc JSON. This may take longer than unit tests but remains CPU-scale.

- [ ] **Step 6: Validate smoke provenance and intervention fields**

Run:

~~~bash
"$HOME/miniconda3/envs/aidemo/bin/python" -c '
import json
from pathlib import Path
root = Path("outputs/hc_p0_smoke_verify")
summaries = list(root.glob("*/run_summary.json"))
assert len(summaries) == 6
for path in summaries:
    data = json.loads(path.read_text())
    assert data["success"] is True
    assert data["metric_schema_version"] == 2
    assert data["initial_transport"]
    assert data["initial_gate_values"]
    assert data["training_transport_history"]
    assert data["final_transport"]
    assert data["final_gate_values"]
    assert data["runtime_provenance"]
    for snapshot in data["training_transport_history"]:
        transport = snapshot["transport"]
        expected_modes = transport["n_streams"] - 1
        assert all(
            len(step["singular_values"]) == expected_modes
            for step in transport["steps"]
        )
        assert len(
            transport["final"]["composite_singular_values"]
        ) == expected_modes
posthoc = list((root / "posthoc").glob("*_mechanism_analysis.json"))
assert len(posthoc) == 6
for path in posthoc:
    data = json.loads(path.read_text())
    assert len(data["persistent_complement_scale"]) > 0
print(len(summaries), len(posthoc))
'
~~~

Expected output: 6 6.

- [ ] **Step 7: Check repository hygiene**

Run:

~~~bash
git diff --check
git status -sb
git log -8 --oneline
~~~

Expected: only intentional source/doc commits; outputs remain ignored or untracked outside the commit. Do not stage generated artifacts.

- [ ] **Step 8: Final completion report**

Report exactly:

- files changed;
- commit hashes;
- unit-test count and command;
- geometry row count;
- six smoke variants and posthoc count;
- current branch/head and whether pushed;
- unverified: RTX 5090, FineWeb-Edu, multi-seed statistics, faithful dynamic mHC, P1 tasks, BiLip-HC, kernel profiling, and 100M–1B scaling.

Do not claim the full research guidance is experimentally proven. P0 code completion means only that the approved causal-control harness is implemented and locally verified.
