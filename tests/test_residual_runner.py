"""CPU checks for the new runner; fixtures are explicitly synthetic, never LM evidence."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments import residual_lm_diagnostic as runner


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.cache("train", torch.arange(57, dtype=torch.int32) % 16)
        self.val = self.cache("val", (torch.arange(41, dtype=torch.int32) + 3) % 16)
        args = runner.parse_args([
            "--methods", "baseline", "--train-cache", str(self.train), "--val-cache", str(self.val),
            "--vocab-size", "16", "--context-length", "8", "--device", "cpu",
            "--layers", "2", "--width", "16", "--heads", "4", "--micro-batch", "2",
            "--grad-accum", "2", "--updates", "5", "--warmup-updates", "1",
            "--val-blocks", "3", "--eval-every", "2", "--checkpoint-every", "2",
            "--lr", ".001", "--min-lr", ".0001", "--dropout", ".15", "--model-seed", "17",
        ])
        args.pop("methods")
        self.config = dict(args, method="baseline", output_dir=str(self.root / "full"))

    def tearDown(self):
        self.temp.cleanup()

    def cache(self, name, tokens):
        path = self.root / f"{name}.pt"
        torch.save(tokens, path)
        manifest = {"status": "complete", "dataset": "synthetic_fixture", "tokenizer": {"vocab_size": 16},
                    "outputs": {name: {"path": str(path), "tokens": tokens.numel(),
                                       "dtype": str(tokens.dtype), "file_sha256": runner.digest_file(path)}}}
        path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
        return path

    def assert_tree_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_tree_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for first, second in zip(expected, actual):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(expected, actual)

    def test_packing_is_contiguous_and_evaluation_scores_fixed_targets(self):
        data = runner.PackedTokens(self.train, 16, 8)
        x, y = data.batch([0, 1])
        torch.testing.assert_close(x.flatten(), torch.arange(16) % 16)
        torch.testing.assert_close(y.flatten(), torch.arange(1, 17) % 16)
        self.assertEqual(data.blocks, 7)
        class TokenLoss(torch.nn.Module):
            def forward(self, x, y):
                return x.float(), y.float().mean()
        model = TokenLoss().train()
        first = runner.evaluate(model, data, 3, 2, torch.device("cpu"))
        second = runner.evaluate(model, data, 3, 3, torch.device("cpu"))
        self.assertEqual(first["targets"], 24)
        self.assertAlmostEqual(first["nll"], second["nll"], places=6)
        self.assertTrue(model.training)

    def test_invalid_cache_manifest_context_and_vocabulary_fail(self):
        with self.assertRaises(FileNotFoundError):
            runner.PackedTokens(self.root / "absent.pt", 16, 8)
        orphan = self.root / "orphan.pt"
        torch.save(torch.arange(20), orphan)
        with self.assertRaises(FileNotFoundError):
            runner.PackedTokens(orphan, 16, 8)
        for name, values in [("row", torch.ones(3, 9)), ("float", torch.ones(20)),
                             ("bad_id", torch.arange(20)), ("too_short", torch.arange(8))]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.PackedTokens(self.cache(name, values), 16, 8)
        with self.assertRaisesRegex(ValueError, "Manifest vocabulary"):
            runner.PackedTokens(self.train, 17, 8)
        manifest_path = self.train.with_suffix(".manifest.json")
        metadata = json.loads(manifest_path.read_text())
        metadata["outputs"]["train"]["file_sha256"] = "incorrect"
        manifest_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Cache SHA"):
            runner.PackedTokens(self.train, 16, 8)
        metadata["outputs"]["train"].update(file_sha256=runner.digest_file(self.train), tokens=58)
        manifest_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "token count"):
            runner.PackedTokens(self.train, 16, 8)

    def test_shuffle_resume_cursor_crosses_epochs_without_global_rng(self):
        first = runner.BlockOrder(7, 29)
        expected = first.take(25)
        torch.manual_seed(999)
        resumed = runner.BlockOrder(7, 29, cursor=11)
        torch.testing.assert_close(resumed.take(14), expected[11:])
        self.assertEqual(resumed.cursor, 25)
        self.assertEqual(set(expected[:7].tolist()), set(range(7)))

    def test_dropout_accumulation_and_optimizer_resume_are_exact(self):
        full = runner.run(self.config)
        interrupted = dict(self.config, output_dir=str(self.root / "interrupted"), stop_after=2)
        partial = runner.run(interrupted)
        resumed_config = dict(self.config, output_dir=str(self.root / "resumed"),
                              resume=str(self.root / "interrupted/final.pt"))
        resumed = runner.run(resumed_config)
        a = torch.load(self.root / "full/final.pt", weights_only=True)
        b = torch.load(self.root / "resumed/final.pt", weights_only=True)
        for key in ("model", "optimizer", "rng", "update", "tokens", "data_cursor"):
            self.assert_tree_equal(a[key], b[key])
        self.assertEqual(partial["status"], "stopped")
        self.assertEqual(full["tokens"], 5 * 4 * 8)
        self.assertEqual(resumed["data_cursor"], 20)
        self.assertEqual(full["final_validation"]["nll"], resumed["final_validation"]["nll"])
        self.assertFalse(any(key.startswith("_orig_mod.") for key in b["model"]))
        updates = lambda p: [x for x in map(json.loads, p.read_text().splitlines()) if x["kind"] == "update"]
        original_rows = updates(self.root / "full/train.jsonl")[2:]
        resumed_rows = updates(self.root / "resumed/train.jsonl")
        keys = ("update", "tokens", "data_cursor", "loss", "grad_norm", "lr")
        self.assertEqual([{k: r[k] for k in keys} for r in original_rows],
                         [{k: r[k] for k in keys} for r in resumed_rows])
        self.assertTrue(resumed["timing_complete"])
        self.assertGreater(resumed_rows[0]["cumulative_elapsed_seconds"], partial["cumulative_elapsed_seconds"])
        times = [r["cumulative_elapsed_seconds"] for r in resumed_rows]
        self.assertEqual(times, sorted(times))
        for row in resumed_rows:
            self.assertGreater(row["elapsed_this_invocation"], 0)
            if "validation" in row:
                self.assertIn("cumulative_elapsed_seconds", row["validation"])
        self.assertGreater(resumed["timing_breakdown_seconds"]["save_seconds"], 0)
        records = list(map(json.loads, (self.root / "resumed/train.jsonl").read_text().splitlines()))
        self.assertEqual(records[-1]["kind"], "end")
        self.assertEqual(resumed["kind"], "training")

    def test_resume_rejects_schedule_optimizer_and_data_changes(self):
        runner.run(dict(self.config, stop_after=2))
        checkpoint = str(self.root / "full/final.pt")
        for change in ({"updates": 6}, {"lr": .002}, {"data_seed": 3}, {"context_length": 4}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Resume identity"):
                runner.run(dict(self.config, resume=checkpoint, **change))
        saved = torch.load(checkpoint, weights_only=True)
        saved["identity"]["numerical_policy"]["evaluation"] = "compiled"
        torch.save(saved, checkpoint)
        with self.assertRaisesRegex(ValueError, "Resume identity"):
            runner.run(dict(self.config, resume=checkpoint))

    def test_optimizer_groups_and_initialization_are_shared(self):
        torch.manual_seed(17)
        baseline = runner.build_model(self.config)
        torch.manual_seed(17)
        adjoint = runner.build_model(dict(self.config, method="adjoint"))
        runner.scale_residual_initialization(baseline, .5)
        runner.scale_residual_initialization(adjoint, .5)
        for name, parameter in baseline.state_dict().items():
            self.assertTrue(torch.equal(parameter, adjoint.state_dict()[name]))
        optimizer, names = runner.make_optimizer(adjoint, self.config, torch.device("cpu"))
        self.assertIn("token_embedding.weight", names["decay"])
        self.assertIn("blocks.0.norm1.weight", names["no_decay"])
        self.assertIn("attn_routers.0.weight", names["no_decay"])
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.)
        self.assertEqual(self.config["residual_init_scale"], .5)

    def test_dry_run_writes_nothing_and_never_queries_cuda(self):
        config = dict(self.config, dry_run=True, device="cuda")
        with patch("torch.cuda.is_available", side_effect=AssertionError("GPU query")):
            result = runner.run(config)
        self.assertEqual(result["validation_targets"], 24)
        self.assertEqual(result["planned_tokens"], 160)
        self.assertFalse(Path(config["output_dir"]).exists())
        self.assertEqual(result["identity"]["train"]["source"], "synthetic_fixture")

    def test_nonfinite_eval_fails_and_restores_training_mode(self):
        class Invalid(torch.nn.Module):
            def forward(self, x, y):
                return x, torch.tensor(float("nan"))
        model = Invalid().train()
        with self.assertRaises(FloatingPointError):
            runner.evaluate(model, runner.PackedTokens(self.val, 16, 8), 3, 2, torch.device("cpu"))
        self.assertTrue(model.training)

    def test_all_registered_methods_complete_one_cpu_update(self):
        for method in runner.METHODS:
            with self.subTest(method=method):
                config = dict(self.config, method=method, updates=1, warmup_updates=0, block_size=1,
                              dropout=0., output_dir=str(self.root / method))
                result = runner.run(config)
                self.assertEqual(result["tokens"], 32)
                if method == "block-attnres":
                    self.assertEqual(result["state_layout"]["max_value_vectors_per_token"], 3)

    def test_compiled_checkpoint_resume_uses_original_model_keys(self):
        config = dict(self.config, compile=True, compile_backend="aot_eager", dropout=0., updates=3)
        runner.run(config)
        runner.run(dict(config, stop_after=1, output_dir=str(self.root / "part")))
        runner.run(dict(config, resume=str(self.root / "part/final.pt"), output_dir=str(self.root / "restored")))
        original = torch.load(self.root / "full/final.pt", weights_only=True)
        restored = torch.load(self.root / "restored/final.pt", weights_only=True)
        self.assert_tree_equal(original["model"], restored["model"])
        self.assert_tree_equal(original["optimizer"], restored["optimizer"])
        self.assertFalse(any(k.startswith("_orig_mod.") for k in restored["model"]))

    def test_compiled_training_never_supplies_the_quality_evaluator(self):
        class TrainingOnly(torch.nn.Module):
            def __init__(self, raw):
                super().__init__()
                self.raw = raw
            def forward(self, *args):
                if not self.training or not torch.is_grad_enabled():
                    raise AssertionError("Compiled wrapper reached quality evaluation")
                return self.raw(*args)
        config = dict(self.config, compile=True, compile_backend="aot_eager", updates=2,
                      eval_every=1, dropout=0.)
        with patch.object(torch, "compile", side_effect=lambda model, **_: TrainingOnly(model)) as compiler, \
             patch.object(runner, "evaluate", wraps=runner.evaluate) as evaluator:
            result = runner.run(config)
        raw = compiler.call_args.args[0]
        self.assertEqual(evaluator.call_count, 3)
        self.assertTrue(all(call.args[0] is raw for call in evaluator.call_args_list))
        self.assertEqual(result["numerical_policy"]["evaluation"], "raw_eager")
        self.assertEqual(result["identity"]["numerical_policy"], result["provenance"]["numerical_policy"])

    def test_compiled_precision_settings_apply_and_missing_amp_policy_fails_closed(self):
        modules = {
            "torch._inductor.config": SimpleNamespace(emulate_precision_casts=False, __file__=runner.__file__),
            "torch._functorch.config": SimpleNamespace(backward_pass_autocast="same_as_forward", __file__=runner.__file__),
        }
        config = dict(self.config, compile=True)
        with patch.object(runner.importlib, "import_module", side_effect=lambda name: modules[name]):
            policy = runner.configure_numerical_policy(config, torch.bfloat16)
            self.assertTrue(policy["emulate_precision_casts"]["effective"])
            self.assertEqual(policy["backward_pass_autocast"]["effective"], "off")
            self.assertEqual(policy["evaluation_autocast_dtype"], "torch.bfloat16")
            del modules["torch._functorch.config"].backward_pass_autocast
            with self.assertRaisesRegex(RuntimeError, "backward_pass_autocast"):
                runner.configure_numerical_policy(config, torch.bfloat16)
            cpu = runner.configure_numerical_policy(config, None)
            self.assertFalse(cpu["backward_pass_autocast"]["applied"])
            self.assertIsNone(cpu["backward_pass_autocast"]["effective"])

    def test_mismatched_timing_sidecar_preserves_model_resume_but_marks_cost_incomplete(self):
        runner.run(dict(self.config, output_dir=str(self.root / "expected")))
        runner.run(dict(self.config, stop_after=2))
        sidecar = self.root / "full/final.pt.timing.json"
        timing = json.loads(sidecar.read_text())
        timing["checkpoint_tag"] = "different_checkpoint"
        sidecar.write_text(json.dumps(timing))
        resumed = runner.run(dict(self.config, resume=str(self.root / "full/final.pt"),
                                  output_dir=str(self.root / "resumed")))
        self.assertFalse(resumed["timing_complete"])
        self.assertEqual(resumed["tokens"], 160)
        expected = torch.load(self.root / "expected/final.pt", weights_only=True)
        actual = torch.load(self.root / "resumed/final.pt", weights_only=True)
        for key in ("model", "optimizer", "rng"):
            self.assert_tree_equal(expected[key], actual[key])

    def test_profile_separates_warmup_from_steady_without_using_a_gpu(self):
        config = dict(self.config, profile_warmup=3, profile_steps=2)
        data = runner.PackedTokens(self.train, 16, 8)
        with patch.object(runner, "optimizer_update"), \
             patch.object(runner.time, "perf_counter", side_effect=[0., 3., 4., 6.]), \
             patch.multiple(torch.cuda, synchronize=lambda *_: None, reset_peak_memory_stats=lambda *_: None,
                            max_memory_allocated=lambda *_: 100, max_memory_reserved=lambda *_: 200,
                            get_device_name=lambda *_: "mock GPU"):
            result = runner.profile(torch.nn.Linear(1, 1), None, data, config, torch.device("cuda"), None, None)
        self.assertEqual(result["warmup_seconds"], 3.)
        self.assertEqual(result["steady_seconds"], 2.)
        self.assertEqual(result["tokens_per_second"], 32.)


if __name__ == "__main__":
    unittest.main()
