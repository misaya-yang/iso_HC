import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lm.diagnostics import (
    DiagnosticsCollector,
    compute_centered_stream_cosine,
    compute_head_output_stats,
    compute_mean_zero_energy,
    compute_mean_zero_norm_ratio,
)
from lm.data import create_dataloader, save_token_cache, TokenizedTextDataset
from lm.headmix import HeadOutputMixing
from lm.mixing import IsoHCMixing, MHCMixing
from lm.models import CausalSelfAttention, TwoBranchHCTransformer
from lm.train import run_experiment
from lm.transport_analysis import complement_spectrum, collect_transport_report
from experiments.lm_5090_next_runs import (
    build_transport_history,
    build_preset_configs,
    create_model,
    run_single,
    runtime_provenance,
)
from experiments.analyze_lm_mechanisms import (
    evaluate_paired_intervention,
    persistent_complement_scale_curve,
)
from experiments.hc_causal_controls import (
    DEPTH_PRESETS,
    build_depth_summary,
    build_suite_configs,
    ensure_run_is_new,
    identity_hc_parameter_count,
    symmetric_birkhoff_gain,
)


class LMNextPhaseContractTests(unittest.TestCase):
    def test_causal_mixer_controls_and_exact_transport_geometry(self):
        torch.manual_seed(31)
        base = MHCMixing(4, noise_std=0.0)
        blended = MHCMixing(4, noise_std=0.0, identity_blend=0.5)
        blended.logits.data.copy_(base.logits.data)
        expected = 0.5 * torch.eye(4) + 0.5 * base()
        self.assertTrue(torch.allclose(blended(), expected, atol=1e-7))

        scaled = IsoHCMixing(
            4, use_svd=True, svd_fallback=False, complement_scale=0.8
        )
        stats = complement_spectrum(scaled())
        self.assertEqual(len(stats["singular_values"]), 3)
        self.assertTrue(all(abs(value - 0.8) < 1e-5
                            for value in stats["singular_values"]))

        report = collect_transport_report([torch.eye(4), torch.eye(4)])
        self.assertEqual(len(report["steps"][0]["singular_values"]), 3)
        self.assertEqual(
            len(report["final"]["composite_singular_values"]), 3
        )
        self.assertAlmostEqual(report["final"]["row_sum_error_max"], 0.0)

        P = torch.ones(4, 4) / 4
        contraction = P + 0.3 * (torch.eye(4) - P)
        deep = collect_transport_report([contraction] * 96)
        self.assertGreater(deep["final"]["composite_sv_min"], 0.0)

    def test_causal_mixer_controls_validate_ranges(self):
        with self.assertRaises(ValueError):
            MHCMixing(4, temperature=0.0)
        with self.assertRaises(ValueError):
            MHCMixing(4, sinkhorn_iters=0)
        with self.assertRaises(ValueError):
            MHCMixing(4, identity_blend=1.1)
        with self.assertRaises(ValueError):
            IsoHCMixing(4, complement_scale=0.0)

    def test_runner_propagates_causal_controls_and_freezes_only_mixers(self):
        cfg = build_preset_configs(
            "run0", ["static-birkhoff-hc"], "outputs/test", batch_size=2
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
        model = create_model(cfg, 128, torch.device("cpu"))

        self.assertEqual(model.mixing_type, "mhc")
        self.assertAlmostEqual(model.readout_lambda.item(), 0.03)
        self.assertAlmostEqual(model.injection_lambda.item(), 0.1)
        for mixing in list(model.attn_mixings) + list(model.mlp_mixings):
            self.assertEqual(mixing.sinkhorn_iters, 5)
            self.assertAlmostEqual(mixing.identity_blend, 0.75)
            self.assertTrue(all(not p.requires_grad
                                for p in mixing.parameters()))
        self.assertTrue(model.readout_lambda.requires_grad)
        self.assertTrue(model.attns[0].q_proj.weight.requires_grad)

    def test_runner_aliases_preserve_legacy_checkpoint_keys(self):
        legacy_cfg = build_preset_configs(
            "run0", ["mhc"], "outputs/test", batch_size=2
        )[0]
        scaled_cfg = build_preset_configs(
            "run0", ["scaled-isohc"], "outputs/test", batch_size=2
        )[0]
        scaled_cfg["mixing_kwargs"] = {"complement_scale": 0.9}

        legacy = create_model(legacy_cfg, 128, torch.device("cpu"))
        reloaded = create_model(legacy_cfg, 128, torch.device("cpu"))
        reloaded.load_state_dict(legacy.state_dict(), strict=True)
        scaled = create_model(scaled_cfg, 128, torch.device("cpu"))
        self.assertEqual(scaled.mixing_type, "isohc")
        self.assertAlmostEqual(
            scaled.attn_mixings[0].complement_scale, 0.9
        )

    def test_failed_run_writes_summary_before_reraising(self):
        cfg = build_preset_configs(
            "run0",
            ["identity-hc"],
            "outputs/test",
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
                with self.assertRaisesRegex(RuntimeError, "expected failure"):
                    run_single(cfg)
            summary = json.loads(
                (Path(tmpdir) / "run_summary.json").read_text()
            )
        self.assertFalse(summary["success"])
        self.assertIn("expected failure", summary["error"])

    def test_schema2_metrics_and_structured_transport_history(self):
        torch.manual_seed(37)
        X = torch.randn(4, 2, 3, 5)
        ratio = compute_mean_zero_norm_ratio(X)
        self.assertAlmostEqual(
            compute_mean_zero_energy(X), ratio ** 2, places=6
        )
        self.assertTrue(torch.isfinite(torch.tensor(
            compute_centered_stream_cosine(X)
        )))

        diagnostics = DiagnosticsCollector()
        snapshot = {
            "optimizer_step": 20,
            "tokens_processed": 2000,
            "transport": {
                "n_streams": 4,
                "steps": [{"singular_values": [0.7, 0.8, 0.9]}],
                "final": {
                    "composite_singular_values": [0.7, 0.8, 0.9]
                },
            },
            "gate_values": {"readout_lambda": 0.02},
        }
        diagnostics.record_snapshot("transport", snapshot)
        self.assertEqual(build_transport_history(diagnostics), [snapshot])

    def test_runtime_provenance_matches_constructed_isohc(self):
        cfg = build_preset_configs(
            "run0", ["isohc"], "outputs/test", batch_size=2,
            use_compile=False,
        )[0]
        cfg["mixing_kwargs"] = {
            "ns_steps": 7,
            "use_svd": False,
            "svd_fallback": True,
        }
        model = create_model(cfg, 128, torch.device("cpu"))
        provenance = runtime_provenance(model, cfg, torch.device("cpu"))
        self.assertEqual(provenance["projection_internal_dtype"], "torch.float64")
        self.assertEqual(provenance["ns_steps"], 7)
        self.assertFalse(provenance["use_svd"])
        self.assertTrue(provenance["svd_fallback"])
        self.assertIsNone(provenance["amp_dtype"])

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
        self.assertEqual(curve[0]["active_state_count"], 4)
        for row in curve:
            self.assertTrue(torch.isfinite(torch.tensor([
                row["delta_nll"],
                row["mean_token_kl"],
                row["top1_change_rate"],
            ])).all())

    def test_causal_suite_math_counts_and_labels(self):
        gain = symmetric_birkhoff_gain(4, 4.0, 1.0)
        measured = complement_spectrum(MHCMixing(
            4, diag_bias=4.0, temperature=1.0,
            noise_std=0.0, sinkhorn_iters=10,
        )())["sv_mean"]
        self.assertAlmostEqual(gain, measured, places=6)

        reference = DEPTH_PRESETS[48]["parameters"]
        for layers, preset in DEPTH_PRESETS.items():
            count = identity_hc_parameter_count(
                50257, 512, 4, layers, preset["d_model"]
            )
            self.assertEqual(count, preset["parameters"])
            self.assertLessEqual(abs(count - reference) / reference, 0.05)
            self.assertEqual(preset["num_transports"], 2 * layers)

        kwargs = dict(
            output_dir="outputs/test",
            dataset="random",
            total_tokens=4096,
            batch_size=2,
            seed=0,
            use_compile=False,
        )
        smoke = build_suite_configs("p0-smoke", **kwargs)
        train = build_suite_configs("p0-train", **kwargs)
        depth = build_suite_configs("p0-depth", **kwargs)
        self.assertEqual((len(smoke), len(train), len(depth)), (6, 14, 25))
        self.assertEqual(len({c["experiment_variant"] for c in depth}), 25)
        self.assertEqual(
            {c["num_layers"]: c["batch_size"] for c in depth},
            {24: 29, 48: 20, 72: 16, 96: 14, 128: 12},
        )

    def test_depth_summary_and_output_safety(self):
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
        self.assertAlmostEqual(row["log_composite_sv_min"], math.log(0.1))
        self.assertAlmostEqual(
            row["sum_step_log_sv_min"],
            math.log(0.5) + math.log(0.25),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "variant"
            run_dir.mkdir()
            (run_dir / "run_summary.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                ensure_run_is_new(run_dir)

    def test_canonical_docs_name_static_proxy_and_causal_runner(self):
        for path in ("README.md", "experiments/README.md", "agent.md"):
            text = Path(path).read_text()
            self.assertIn("static-birkhoff-hc", text)
            self.assertIn("experiments/hc_causal_controls.py", text)

    def test_random_dataset_supports_offline_training_smoke(self):
        class DummyTokenizer:
            vocab_size = 257

        loader, dataset = create_dataloader(
            "random",
            DummyTokenizer(),
            context_length=17,
            batch_size=3,
            split="train",
            max_samples=19,
            num_workers=0,
        )
        x, y = next(iter(loader))

        self.assertEqual(len(dataset), 19)
        self.assertEqual(x.shape, (3, 17))
        self.assertEqual(y.shape, (3, 17))
        self.assertGreaterEqual(x.min().item(), 0)
        self.assertLess(x.max().item(), 257)

    def test_token_cache_can_be_saved_and_loaded_by_dataloader(self):
        class DummyTokenizer:
            vocab_size = 64

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/random_train.pt"
            saved = save_token_cache(
                dataset_name="random",
                tokenizer=DummyTokenizer(),
                cache_path=cache_path,
                context_length=8,
                split="train",
                max_samples=16,
            )
            self.assertEqual(saved, cache_path)

            loader, dataset = create_dataloader(
                "random",
                DummyTokenizer(),
                context_length=8,
                batch_size=2,
                split="train",
                cache_path=cache_path,
                num_workers=0,
            )
            x, y = next(iter(loader))

            self.assertIsInstance(dataset, TokenizedTextDataset)
            self.assertEqual(x.shape, (2, 8))
            self.assertEqual(y.shape, (2, 8))

    def test_attention_uses_scaled_dot_product_attention_when_flash_enabled(self):
        calls = []
        original_sdpa = F.scaled_dot_product_attention

        def fake_sdpa(q, k, v, *args, **kwargs):
            calls.append(kwargs)
            return torch.zeros_like(q)

        try:
            F.scaled_dot_product_attention = fake_sdpa
            attn = CausalSelfAttention(
                hidden_dim=32,
                num_heads=4,
                dropout=0.0,
                use_flash=True,
            )
            x = torch.randn(2, 8, 32)
            y = attn(x)
        finally:
            F.scaled_dot_product_attention = original_sdpa

        self.assertEqual(y.shape, (2, 8, 32))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["is_causal"])

    def test_iso_head_output_mix_preserves_head_mean_and_complement_energy(self):
        torch.manual_seed(7)
        mixer = HeadOutputMixing(
            num_heads=8,
            mixing_type="isohc",
            ns_steps=5,
            init_scale=0.15,
        )
        x = torch.randn(3, 11, 8, 16)
        y, stats = mixer(x, return_stats=True)

        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        x_perp = x - x_mean.unsqueeze(2)
        y_perp = y - y_mean.unsqueeze(2)
        ratio = y_perp.norm().square() / x_perp.norm().square()

        self.assertEqual(y.shape, x.shape)
        self.assertLess(torch.norm(y_mean - x_mean).item(), 1e-5)
        self.assertAlmostEqual(ratio.item(), 1.0, places=4)
        self.assertLess(stats["mean_drift"], 1e-5)
        self.assertAlmostEqual(stats["complement_energy_ratio"], 1.0, places=4)

    def test_attention_headmix_forward_handles_noncontiguous_mixed_heads(self):
        attn = CausalSelfAttention(
            hidden_dim=48,
            num_heads=6,
            dropout=0.0,
            use_flash=True,
            head_mixing_type="isohc",
            head_mixing_kwargs={"ns_steps": 3},
        )
        y = attn(torch.randn(2, 9, 48))

        self.assertEqual(y.shape, (2, 9, 48))
        self.assertIsNotNone(attn.last_headmix_stats)
        self.assertIn("head_effective_rank", attn.last_headmix_stats)

    def test_two_branch_hc_transformer_has_attention_and_mlp_transport(self):
        torch.manual_seed(11)
        model = TwoBranchHCTransformer(
            vocab_size=128,
            d_model=48,
            num_layers=2,
            num_heads=4,
            n_streams=4,
            context_length=16,
            mixing_type="isohc",
            ns_steps=3,
            use_flash=True,
        )
        x = torch.randint(0, 128, (2, 16))
        logits, loss = model(x, x)
        loss.backward()

        self.assertEqual(logits.shape, (2, 16, 128))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(model.attn_mixings), 2)
        self.assertEqual(len(model.mlp_mixings), 2)

        diags = model.get_diagnostics()
        self.assertEqual(len(diags), 4)
        self.assertTrue(all(d["branch"] in {"attn", "mlp"} for d in diags))

        states = model.get_stream_states(x)
        self.assertEqual(len(states), 1 + 2 * model.num_layers)
        self.assertEqual(states[0].shape, (4, 2, 16, 48))

    def test_transport_report_tracks_composite_complement_gain(self):
        eye = torch.eye(4)
        identity_report = collect_transport_report([
            {"branch": "attn", "layer": 0, "H": eye},
            {"branch": "mlp", "layer": 0, "H": eye},
        ])
        self.assertAlmostEqual(identity_report["prefix"][-1]["composite_sv_mean"], 1.0, places=5)
        self.assertAlmostEqual(identity_report["prefix"][-1]["product_sv_mean"], 1.0, places=5)

        contraction = 0.5 * eye + 0.5 * torch.ones(4, 4) / 4
        contraction_report = collect_transport_report([
            {"branch": "attn", "layer": 0, "H": contraction},
            {"branch": "mlp", "layer": 0, "H": contraction},
        ])
        self.assertLess(contraction_report["prefix"][-1]["composite_sv_mean"], 1.0)
        self.assertLess(contraction_report["prefix"][-1]["product_sv_mean"], 1.0)

    def test_two_branch_forward_supports_complement_removal_and_stream_grad_capture(self):
        torch.manual_seed(19)
        model = TwoBranchHCTransformer(
            vocab_size=96,
            d_model=32,
            num_layers=2,
            num_heads=4,
            n_streams=4,
            context_length=12,
            mixing_type="isohc",
            ns_steps=3,
            use_flash=True,
        )
        x = torch.randint(0, 96, (2, 12))
        logits, loss, states = model(
            x,
            x,
            stream_intervention={"state_index": 1, "mode": "mean_only"},
            return_stream_states=True,
            retain_stream_grads=True,
        )
        loss.backward()

        self.assertEqual(logits.shape, (2, 12, 96))
        self.assertEqual(len(states), 1 + 2 * model.num_layers)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(states[1].grad)
        mean_only_state = states[1].detach()
        self.assertLess(torch.norm(mean_only_state - mean_only_state.mean(dim=0, keepdim=True)).item(), 1e-5)

    def test_two_branch_named_mixing_matrices_are_ordered_by_transport_step(self):
        model = TwoBranchHCTransformer(
            vocab_size=64,
            d_model=32,
            num_layers=3,
            num_heads=4,
            n_streams=4,
            context_length=8,
            mixing_type="identity",
        )
        named = model.get_named_mixing_matrices()

        self.assertEqual(len(named), 6)
        self.assertEqual([(m["branch"], m["layer"]) for m in named[:2]], [("attn", 0), ("mlp", 0)])
        self.assertEqual(named[0]["H"].shape, (4, 4))

    def test_head_output_stats_reports_diversity_and_effective_rank(self):
        x = torch.randn(2, 5, 4, 8)
        stats = compute_head_output_stats(x)

        self.assertIn("head_offdiag_cosine", stats)
        self.assertIn("head_effective_rank", stats)
        self.assertGreaterEqual(stats["head_effective_rank"], 1.0)
        self.assertLessEqual(stats["head_effective_rank"], 4.0)

    def test_5090_presets_match_teacher_run_plan(self):
        configs = build_preset_configs(
            preset="deep-stress",
            methods=["baseline", "identity-hc", "unconstrained", "mhc", "isohc"],
            output_dir="outputs/test",
            total_tokens=1_000_000,
            batch_size=8,
        )
        self.assertEqual([c["method"] for c in configs],
                         ["baseline", "identity-hc", "unconstrained", "mhc", "isohc"])
        self.assertTrue(all(c["num_layers"] == 24 for c in configs))
        self.assertTrue(all(c["context_length"] == 512 for c in configs))
        self.assertTrue(all(c["n_streams"] == 4 for c in configs))
        self.assertTrue(all(c["use_compile"] for c in configs))

        smoke = build_preset_configs(
            preset="125m-smoke",
            methods=["mhc", "isohc"],
            output_dir="outputs/test",
            total_tokens=1_000_000,
            batch_size=4,
        )
        self.assertTrue(all(c["num_layers"] == 12 for c in smoke))
        self.assertTrue(all(c["d_model"] == 768 for c in smoke))
        self.assertTrue(all(c["num_heads"] == 12 for c in smoke))

        fe_deep = build_preset_configs(
            preset="fe-deep-36l-512",
            methods=["baseline", "identity-hc", "mhc", "isohc"],
            output_dir="outputs/test",
        )
        self.assertTrue(all(c["num_layers"] == 36 for c in fe_deep))
        self.assertTrue(all(c["d_model"] == 512 for c in fe_deep))
        self.assertTrue(all(c["context_length"] == 512 for c in fe_deep))
        self.assertTrue(all(c["batch_size"] == 8 for c in fe_deep))

    def test_identity_hc_method_uses_multistream_identity_mixing(self):
        cfg = build_preset_configs(
            preset="run0",
            methods=["identity-hc"],
            output_dir="outputs/test",
            batch_size=2,
        )[0]
        model = create_model(cfg, vocab_size=128, device=torch.device("cpu"))

        self.assertIsInstance(model, TwoBranchHCTransformer)
        self.assertEqual(model.mixing_type, "identity")

    def test_spectral_hc_methods_are_not_in_main_experiment_runner(self):
        configs = build_preset_configs(
            preset="run0",
            methods=["spectral-hc", "fixed-vector-spectral-hc"],
            output_dir="outputs/test",
            batch_size=2,
        )

        with self.assertRaises(ValueError):
            create_model(configs[0], vocab_size=128, device=torch.device("cpu"))
        with self.assertRaises(ValueError):
            create_model(configs[1], vocab_size=128, device=torch.device("cpu"))

    def test_run_experiment_respects_gradient_accumulation_steps(self):
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_layers = 1
                self.embedding = nn.Embedding(32, 8)
                self.proj = nn.Linear(8, 32)

            def forward(self, x, y=None):
                logits = self.proj(self.embedding(x))
                loss = None
                if y is not None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    )
                return logits, loss

            def count_parameters(self):
                return sum(p.numel() for p in self.parameters())

        class DummyTokenizer:
            vocab_size = 32

        loader, _ = create_dataloader(
            "random",
            DummyTokenizer(),
            context_length=8,
            batch_size=2,
            split="train",
            max_samples=16,
            num_workers=0,
        )
        val_loader, _ = create_dataloader(
            "random",
            DummyTokenizer(),
            context_length=8,
            batch_size=2,
            split="validation",
            max_samples=8,
            num_workers=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_experiment(
                TinyLM(),
                loader,
                val_loader,
                {
                    "total_tokens": 64,
                    "max_lr": 1e-3,
                    "min_lr": 1e-4,
                    "warmup_tokens": 0,
                    "grad_clip": 1.0,
                    "use_amp": False,
                    "eval_every_tokens": 64,
                    "save_dir": tmpdir,
                    "diagnostics_every": 1,
                    "weight_decay": 0.0,
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eval_max_batches": 1,
                    "grad_accum_steps": 2,
                    "save_checkpoints": False,
                    "use_compile": False,
                },
                torch.device("cpu"),
            )

        self.assertEqual(results["train_metrics"][0]["steps"], 2)
        self.assertEqual(results["train_metrics"][0]["total_tokens"], 64)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
