"""Scientific contracts for unit-address, adjoint read/write residuals.

These CPU checks establish algebra, initialization and trainability, not language
model quality or whole-network stability. In particular, the singular-value
identity below holds with the address fixed; a separate test refutes extending
that result to input-dependent addresses.
"""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.lm_5090_next_runs import build_preset_configs, create_model, run_single
from lm.adjoint import AdjointHCTransformer, UnitAddressRouter, adjoint_read, adjoint_step
from lm.models import BaselineTransformer


MODEL_KWARGS = dict(
    vocab_size=67,
    d_model=32,
    num_layers=2,
    num_heads=4,
    context_length=8,
    mlp_ratio=2,
    dropout=0.0,
    use_flash=False,
)


def make_model(carrier="signed", **kwargs):
    torch.manual_seed(1729)
    return AdjointHCTransformer(
        **MODEL_KWARGS, carrier=carrier, **kwargs
    ).double()


def routers(model):
    return [*model.attn_routers, *model.mlp_routers, model.output_router]


class AdjointContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.tokens = torch.randint(0, MODEL_KWARGS["vocab_size"], (2, 8))
        self.targets = torch.roll(self.tokens, shifts=-1, dims=1)

    def test_same_seed_has_identical_backbone_and_exact_baseline_function(self):
        torch.manual_seed(1729)
        baseline = BaselineTransformer(**MODEL_KWARGS).double()
        model = make_model()
        for name, value in baseline.state_dict().items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    model.state_dict()[name], value, rtol=0, atol=0
                )
        for router in routers(model):
            self.assertEqual(torch.count_nonzero(router.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(router.bias).item(), 0)
        expected_logits, expected_loss = baseline(self.tokens, self.targets)
        logits, loss = model(self.tokens, self.targets)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)

    def test_signed_carrier_routes_receive_standard_ntp_gradients_at_init(self):
        model = make_model()
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        for i, router in enumerate(routers(model)):
            with self.subTest(router=i):
                self.assertTrue(torch.isfinite(router.weight.grad).all())
                self.assertGreater(router.weight.grad.norm().item(), 1e-9)
                self.assertTrue(torch.isfinite(router.bias.grad).all())

    def test_zero_carrier_has_dead_address_gradients_at_exact_baseline_init(self):
        model = make_model(carrier="zero")
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        for i, router in enumerate(routers(model)):
            with self.subTest(router=i):
                self.assertEqual(torch.count_nonzero(router.weight.grad).item(), 0)
                self.assertEqual(torch.count_nonzero(router.bias.grad).item(), 0)

    def test_copy_carrier_radial_suppression_is_local_not_whole_network_death(self):
        model = make_model(carrier="copy")
        block = model.blocks[0]
        primary = torch.randn(2, 8, 32, dtype=torch.float64)
        probe = torch.randn_like(primary)
        sign = torch.ones(32, dtype=torch.float64)
        sign[1::2] = -1

        def gate_gradient(carrier):
            angle = torch.zeros((), dtype=torch.float64, requires_grad=True)
            scale = torch.rsqrt(1 + angle.square())
            read = scale * (primary + angle * carrier)
            objective = (block.attn(block.norm1(read)) * probe).sum()
            return torch.autograd.grad(objective, angle)[0].abs().item()

        copy_gradient = gate_gradient(primary)
        signed_gradient = gate_gradient(primary * sign)
        self.assertGreater(signed_gradient, 1e-5)
        self.assertLess(copy_gradient, signed_gradient * 1e-4)

        # By later branches x has changed while the copied initial carrier has
        # not, so it is no longer necessarily a radial direction for RMSNorm.
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        self.assertGreater(model.mlp_routers[0].weight.grad.norm().item(), 1e-9)
        self.assertGreater(model.attn_routers[1].weight.grad.norm().item(), 1e-9)

    def test_unit_address_and_zero_delta_preserve_entire_state(self):
        router = UnitAddressRouter(7, routing="dynamic").double()
        with torch.no_grad():
            router.weight.normal_(std=0.7)
            router.bias.fill_(0.3)
        primary = torch.randn(2, 8, 7, dtype=torch.float64)
        memory = torch.randn_like(primary)
        address = router(primary)
        torch.testing.assert_close(
            address.square().sum(-1), torch.ones(2, 8, dtype=torch.float64),
            rtol=0, atol=5e-16,
        )
        self.assertGreater(address[..., 1].abs().max().item(), 0.1)
        next_primary, next_memory = adjoint_step(
            primary, memory, torch.zeros_like(primary), address
        )
        torch.testing.assert_close(next_primary, primary, rtol=0, atol=0)
        torch.testing.assert_close(next_memory, memory, rtol=0, atol=0)

    def test_zero_branches_preserve_all_full_model_states_with_nonzero_routing(self):
        model = make_model()
        with torch.no_grad():
            for router in routers(model):
                router.weight.normal_(std=0.4)
                router.bias.fill_(0.3)
            for block in model.blocks:
                block.attn.o_proj.weight.zero_()
                block.mlp.proj.weight.zero_()
        states = model.get_stream_states(self.tokens)
        self.assertEqual(len(states), 1 + 2 * MODEL_KWARGS["num_layers"])
        self.assertEqual(states[0].shape, (2, 2, 8, 32))
        for state in states[1:]:
            torch.testing.assert_close(state, states[0], rtol=0, atol=0)

    def test_frozen_aux_control_preserves_carrier_while_live_candidate_updates_it(self):
        live = make_model()
        frozen = make_model(freeze_aux_updates=True)
        with torch.no_grad():
            for router in routers(live):
                router.weight.normal_(std=0.4)
                router.bias.fill_(0.3)
        frozen.load_state_dict(live.state_dict())
        live_states = live.get_stream_states(self.tokens)
        frozen_states = frozen.get_stream_states(self.tokens)
        torch.testing.assert_close(
            live_states[0][1], live_states[0][0] * live.carrier_mask, rtol=0, atol=0
        )
        for state in frozen_states[1:]:
            torch.testing.assert_close(state[1], frozen_states[0][1], rtol=0, atol=0)
        self.assertGreater((live_states[1][1] - live_states[0][1]).norm().item(), 1e-6)
        torch.testing.assert_close(live_states[1][0], frozen_states[1][0], rtol=0, atol=0)

    def test_same_address_reads_residual_update_and_preserves_orthogonal_view(self):
        primary = torch.randn(2, 8, 7, dtype=torch.float64)
        memory = torch.randn_like(primary)
        delta = torch.randn_like(primary)
        raw = torch.randn(2, 8, 2, dtype=torch.float64)
        address = raw / raw.norm(dim=-1, keepdim=True)
        c0, c1 = address[..., 0:1], address[..., 1:2]
        next_primary, next_memory = adjoint_step(primary, memory, delta, address)
        torch.testing.assert_close(
            c0 * next_primary + c1 * next_memory,
            c0 * primary + c1 * memory + delta,
            rtol=1e-14, atol=1e-14,
        )
        torch.testing.assert_close(
            -c1 * next_primary + c0 * next_memory,
            -c1 * primary + c0 * memory,
            rtol=1e-14, atol=1e-14,
        )

    def test_actual_fixed_address_jacobian_has_branch_and_identity_singular_values(self):
        dim = 3
        address = torch.tensor([0.8, 0.6], dtype=torch.float64)
        branch = torch.tensor(
            [[0.2, 0.7, 0.0], [-0.4, -0.6, 0.3], [0.0, 0.1, 0.5]],
            dtype=torch.float64,
        )

        def residual_map(state):
            primary, memory = state[:dim], state[dim:]
            read = address[0] * primary + address[1] * memory
            result = adjoint_step(primary, memory, branch @ read, address)
            return torch.cat(result)

        state = torch.randn(2 * dim, dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(residual_map, state)
        expected = torch.cat([
            torch.linalg.svdvals(torch.eye(dim, dtype=torch.float64) + branch),
            torch.ones(dim, dtype=torch.float64),
        ]).sort().values
        torch.testing.assert_close(
            torch.linalg.svdvals(jacobian).sort().values, expected,
            rtol=1e-13, atol=1e-13,
        )

    def test_dynamic_router_counterexample_refutes_unconditional_jacobian_bound(self):
        router = UnitAddressRouter(2, routing="dynamic").double()
        with torch.no_grad():
            router.weight.copy_(torch.tensor([0.0, 10.0], dtype=torch.float64))
            router.bias.zero_()

        def residual_map(state, freeze_address):
            primary, memory = state[:2], state[2:]
            address = router(primary)
            if freeze_address:
                address = address.detach()
            read = address[0] * primary + address[1] * memory
            return torch.cat(adjoint_step(primary, memory, -0.5 * read, address))

        state = torch.tensor([1.0, 0.0, 0.0, 4.0], dtype=torch.float64)
        fixed_jacobian = torch.autograd.functional.jacobian(
            lambda value: residual_map(value, True), state
        )
        dynamic_jacobian = torch.autograd.functional.jacobian(
            lambda value: residual_map(value, False), state
        )
        self.assertAlmostEqual(torch.linalg.svdvals(fixed_jacobian).max().item(), 1.0)
        self.assertGreater(torch.linalg.svdvals(dynamic_jacobian).max().item(), 10.0)

    def test_learned_nonzero_routing_preserves_token_causality(self):
        model = make_model().eval()
        with torch.no_grad():
            for router in routers(model):
                router.weight.normal_(std=0.4)
                router.bias.fill_(0.3)
        changed = self.tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % MODEL_KWARGS["vocab_size"]
        before, _ = model(self.tokens)
        after, _ = model(changed)
        torch.testing.assert_close(before[:, :4], after[:, :4], rtol=0, atol=0)
        self.assertGreater((before[:, 4:] - after[:, 4:]).norm().item(), 1e-6)

    def test_full_model_output_is_independent_of_other_batch_examples(self):
        model = make_model().eval()
        with torch.no_grad():
            for router in routers(model):
                router.weight.normal_(std=0.4)
                router.bias.fill_(-0.2)
        changed = self.tokens.clone()
        changed[1] = (changed[1] + 7) % MODEL_KWARGS["vocab_size"]
        original, _ = model(self.tokens)
        replaced_neighbor, _ = model(changed)
        alone, _ = model(self.tokens[:1])
        torch.testing.assert_close(original[0], replaced_neighbor[0], rtol=0, atol=0)
        torch.testing.assert_close(original[:1], alone, rtol=1e-13, atol=1e-13)
        self.assertGreater((original[1] - replaced_neighbor[1]).norm().item(), 1e-6)

    def test_bfloat16_uses_float32_address_arithmetic_without_promoting_state(self):
        router = UnitAddressRouter(32).bfloat16()
        with torch.no_grad():
            router.weight.normal_(std=0.4)
            router.bias.fill_(0.2)
        primary = torch.randn(2, 8, 32, dtype=torch.bfloat16, requires_grad=True)
        memory = torch.randn_like(primary, requires_grad=True)
        delta = torch.randn_like(primary, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            address = router(primary)
            read = adjoint_read(primary, memory, address)
            next_primary, next_memory = adjoint_step(primary, memory, delta, address)
        self.assertEqual(address.dtype, torch.float32)
        torch.testing.assert_close(
            address.square().sum(-1), torch.ones(2, 8), rtol=0, atol=2e-7
        )
        for state in (read, next_primary, next_memory):
            self.assertEqual(state.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(state).all())
        # Casting an address into the storage dtype introduces rounding; exact
        # real-arithmetic unit-length identities are not promised for bf16.
        quantized_norm = address.bfloat16().float().square().sum(-1)
        self.assertLess((quantized_norm - 1).abs().max().item(), 0.008)
        (read.float().square().mean() + next_primary.float().square().mean()
         + next_memory.float().square().mean()).backward()
        for tensor in (primary, memory, delta, router.weight, router.bias):
            self.assertTrue(torch.isfinite(tensor.grad).all())

        model = make_model().bfloat16()
        logits, loss = model(self.tokens, self.targets)
        loss.backward()
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_generic_runner_builds_two_stream_candidate_and_mechanism_controls(self):
        methods = [
            "adjoint-hc", "adjoint-hc-static", "adjoint-hc-frozen-aux",
            "adjoint-hc-zero-carrier", "adjoint-hc-copy-carrier",
        ]
        configs = build_preset_configs(
            preset="deep-stress", methods=methods,
            output_dir="outputs/test_adjoint_contracts", use_compile=False,
        )
        for config in configs:
            with self.subTest(method=config["method"]):
                self.assertEqual(config["n_streams"], 2)
                config.update({
                    key: value for key, value in MODEL_KWARGS.items()
                    if key != "vocab_size"
                })
                model = create_model(config, MODEL_KWARGS["vocab_size"], torch.device("cpu"))
                self.assertIsInstance(model, AdjointHCTransformer)
                self.assertEqual(model.n_streams, 2)
                expected_routing = "static" if config["method"].endswith("static") else "dynamic"
                self.assertEqual(model.routing, expected_routing)
                self.assertEqual(
                    model.freeze_aux_updates, config["method"] == "adjoint-hc-frozen-aux"
                )
                expected_carrier = {
                    "adjoint-hc-zero-carrier": "zero",
                    "adjoint-hc-copy-carrier": "copy",
                }.get(config["method"], "signed")
                self.assertEqual(model.carrier, expected_carrier)
                logits, loss = model(self.tokens, self.targets)
                self.assertEqual(logits.shape, (2, 8, MODEL_KWARGS["vocab_size"]))
                self.assertTrue(torch.isfinite(loss))
        invalid_config = {**configs[0], "n_streams": 4}
        with self.assertRaises(ValueError):
            create_model(invalid_config, MODEL_KWARGS["vocab_size"], torch.device("cpu"))

    def test_generic_runner_trains_evaluates_and_records_offline_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_preset_configs(
                preset="run0", methods=["adjoint-hc"], output_dir=tmpdir,
                dataset="random", total_tokens=64, batch_size=2, use_compile=False,
            )[0]
            config.update({
                "d_model": 32,
                "num_layers": 1,
                "num_heads": 4,
                "context_length": 8,
                "vocab_size": 16,
                "max_samples": 32,
                "max_samples_val": 32,
                "num_workers": 0,
                "persistent_workers": False,
                "save_checkpoints": False,
                "save_best_checkpoints": False,
                "save_final_checkpoint": False,
                "use_amp": False,
                "warmup_tokens": 0,
                "eval_every_tokens": 32,
                "eval_max_batches": 2,
                "diagnostics_every": 1,
            })
            # Only device selection is fixed: data, model, optimizer,
            # evaluation, diagnostics and persistence all run for real.
            with patch("torch.cuda.is_available", return_value=False):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_single(config)

            self.assertTrue(result["success"])
            self.assertEqual(result["config"]["n_streams"], 2)
            self.assertTrue(math.isfinite(result["final_eval"]["val_loss"]))
            self.assertTrue(math.isfinite(result["final_eval"]["val_ppl"]))
            self.assertEqual(result["final_eval"]["val_batches"], 2)
            self.assertEqual(sum(row["steps"] for row in result["train_metrics"]), 4)
            self.assertEqual(sum(row["total_tokens"] for row in result["train_metrics"]), 64)
            for name in ("mean_zero_energy_curve", "stream_cosine_curve"):
                curve = result["posthoc"][name]
                self.assertEqual(len(curve), 3)
                self.assertTrue(all(math.isfinite(value) for value in curve))
            self.assertGreater(result["posthoc"]["mean_zero_energy_initial"], 0)

            output = Path(config["save_dir"])
            saved = json.loads((output / "run_summary.json").read_text())
            self.assertEqual(saved["config"]["n_streams"], 2)
            self.assertEqual(saved["posthoc"], result["posthoc"])
            self.assertEqual(saved["final_eval"], result["final_eval"])
            self.assertFalse((output / "final.pt").exists())
            self.assertFalse((output / "best.pt").exists())

    def test_ntp_optimizer_step_and_state_dict_roundtrip(self):
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))
        optimizer.step()
        self.assertGreater(model.attn_routers[0].weight.norm().item(), 0.0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = make_model()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        expected, expected_loss = model(self.tokens, self.targets)
        actual, actual_loss = restored(self.tokens, self.targets)
        self.assertTrue(torch.isfinite(actual_loss))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
