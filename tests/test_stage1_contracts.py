import contextlib
import io
import os
import tempfile
import unittest

import torch

from experiments.stage1_projection_sanity import save_results as save_projection_results
from experiments.stage1_residual_only import run_residual_only
from experiments.stage1_tiny_smoke import evaluate_pass as evaluate_smoke_pass
from isohc.projection import construct_orthogonal_complement, iso_ns_project
from isohc.transformer import UnconstrainedHCTransformer


class Stage1ContractTests(unittest.TestCase):
    def test_iso_ns_k5_projects_random_matrices_to_stage1_tolerance(self):
        torch.manual_seed(123)

        for n in (4, 8, 16):
            U = construct_orthogonal_complement(n, device="cpu", dtype=torch.float32)
            eye = torch.eye(n, dtype=torch.float32)
            ones = torch.ones(n, 1, dtype=torch.float32)

            max_orth_error = 0.0
            max_fix_error = 0.0
            for _ in range(16):
                h_raw = torch.randn(n, n, dtype=torch.float32) * 0.5
                h = iso_ns_project(h_raw, U=U, steps=5, use_svd=False)
                max_orth_error = max(
                    max_orth_error,
                    torch.norm(h.T @ h - eye, p="fro").item(),
                )
                max_fix_error = max(
                    max_fix_error,
                    torch.norm(h @ ones - ones, p=2).item(),
                )

            self.assertLess(max_orth_error, 5e-3)
            self.assertLess(max_fix_error, 1e-5)

    def test_projection_results_save_supports_svd_and_integer_steps(self):
        results = {
            4: {
                "svd": {
                    "mean_orth_error": 1e-7,
                    "max_orth_error": 2e-7,
                    "mean_fix_error": 1e-7,
                    "max_fix_error": 2e-7,
                    "mean_energy_error": 1e-7,
                    "max_energy_error": 2e-7,
                },
                5: {
                    "mean_orth_error": 1e-7,
                    "max_orth_error": 2e-7,
                    "mean_fix_error": 1e-7,
                    "max_fix_error": 2e-7,
                    "mean_energy_error": 1e-7,
                    "max_energy_error": 2e-7,
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                save_projection_results(results, passed=True, output_dir=tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "projection_sanity.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "projection_sanity.md")))

    def test_smoke_pass_fails_when_training_did_not_complete_requested_steps(self):
        config = {"steps": 3, "model_type": "baseline"}
        metrics = [
            {
                "step": 0,
                "loss": 8.0,
                "grad_norm": 1.0,
                "has_nan": False,
                "has_inf": False,
            }
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            passed = evaluate_smoke_pass(config, metrics)

        self.assertFalse(passed)

    def test_unconstrained_transformer_model_option_is_implemented(self):
        model = UnconstrainedHCTransformer(
            vocab_size=128,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            n_streams=4,
            context_length=16,
        )
        input_ids = torch.randint(0, 128, (2, 16))
        logits, loss = model(input_ids, input_ids)

        self.assertEqual(logits.shape, (2, 16, 128))
        self.assertIsNotNone(loss)

    def test_residual_only_cpu_bf16_uses_representative_precision_for_mean_check(self):
        result = run_residual_only(
            method="isohc",
            n=4,
            feature_dim=32,
            depth=32,
            num_trials=2,
            ns_steps=5,
            dtype_tensor=torch.bfloat16,
            device="cpu",
            seed=123,
        )

        self.assertLess(result["mean_error_max"], 1e-5)


if __name__ == "__main__":
    unittest.main()
