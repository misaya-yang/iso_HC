import contextlib
import io
import tempfile
import unittest

import torch
import torch.nn.functional as F

from lm.diagnostics import compute_head_output_stats
from lm.data import create_dataloader, save_token_cache, TokenizedTextDataset
from lm.headmix import HeadOutputMixing
from lm.models import CausalSelfAttention, TwoBranchHCTransformer
from experiments.lm_5090_next_runs import build_preset_configs


class LMNextPhaseContractTests(unittest.TestCase):
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
            methods=["baseline", "unconstrained", "mhc", "isohc"],
            output_dir="outputs/test",
            total_tokens=1_000_000,
            batch_size=8,
        )
        self.assertEqual([c["method"] for c in configs],
                         ["baseline", "unconstrained", "mhc", "isohc"])
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


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
