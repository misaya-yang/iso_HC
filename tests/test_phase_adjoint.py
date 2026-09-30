"""CPU scientific contracts for the phase candidate and matched controls.

These checks cover initialization, live NTP gradients, coordinate equivalence,
causality and numerical execution. They do not establish language-model quality,
GPU efficiency, or whole-network Jacobian stability.
"""

import copy
import io
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from lm.adjoint import UnitAddressRouter, adjoint_step
from lm.models import BaselineTransformer
from lm.phase_adjoint import (
    BoundarySkipTransformer,
    ControlledAdjointHCTransformer,
    PhaseAdjointHCTransformer,
    ResidualGainTransformer,
    ScalarResidualRouter,
    TerminalAdjointHCTransformer,
    create_residual_model,
    phase_switch,
)


MODEL_KWARGS = dict(
    vocab_size=67,
    d_model=32,
    num_layers=4,
    num_heads=4,
    context_length=8,
    mlp_ratio=2,
    dropout=0.0,
    use_flash=False,
)
METHODS = (
    "baseline", "gain", "adjoint", "adjoint-frozen", "adjoint-shear",
    "phase-adjoint", "phase-adjoint-frozen", "phase-adjoint-shear",
    "phase-adjoint-post-frozen", "boundary-skip",
    "terminal-adjoint",
)


def make_model(method="phase-adjoint", dtype=torch.float64, **kwargs):
    torch.manual_seed(1729)
    return create_residual_model(method, **{**MODEL_KWARGS, **kwargs}).to(dtype=dtype)


def branch_routers(model):
    return [*model.attn_routers, *model.mlp_routers]


def all_routers(model):
    result = branch_routers(model)
    if hasattr(model, "output_router"):
        result.append(model.output_router)
    result += [router.shear for router in branch_routers(model) if hasattr(router, "shear")]
    return result


def activate_routers(model):
    with torch.no_grad():
        for router in all_routers(model):
            if router.weight is not None:
                router.weight.normal_(std=0.4)
            router.bias.fill_(0.2)


def global_coordinate_forward(model, input_ids, targets):
    """Independent reciprocal-scale formulation, without a state switch."""
    position = torch.arange(input_ids.shape[1], device=input_ids.device)
    x = model.token_embedding(input_ids) + model.pos_embedding(position)
    m = torch.zeros_like(x)
    for index, (block, attn_router, mlp_router) in enumerate(zip(
        model.blocks, model.attn_routers, model.mlp_routers
    )):
        for router, norm, branch in (
            (attn_router, block.norm1, block.attn),
            (mlp_router, block.norm2, block.mlp),
        ):
            if index < model.boundary:
                c = router(x)
                read = c[..., :1] * x + c[..., 1:] * m
                write_x, write_m = c[..., :1], c[..., 1:]
            else:
                c = router(x + m)
                # This is c_global / a with a = 1/sqrt(2). Writing is
                # a*c_global, hence half of these read coefficients.
                read_x = c[..., :1] - c[..., 1:]
                read_m = c[..., :1] + c[..., 1:]
                read = read_x * x + read_m * m
                write_x, write_m = read_x / 2, read_m / 2
            delta = branch(norm(read))
            x, m = x + write_x * delta, m + write_m * delta
    c = model.output_router(x + m)
    read = (c[..., :1] - c[..., 1:]) * x + (c[..., :1] + c[..., 1:]) * m
    logits = model.lm_head(model.norm_final(read))
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
    return logits, loss


class PhaseAdjointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(23)
        self.tokens = torch.randint(0, MODEL_KWARGS["vocab_size"], (2, 8))
        self.targets = torch.roll(self.tokens, shifts=-1, dims=1)

    def test_same_seed_exact_baseline_logits_backbone_and_input_gradients(self):
        baseline = make_model("baseline", dtype=torch.float32)
        captured = []

        def capture_input(module, args, output):
            output.retain_grad()
            captured.append(output)

        handle = baseline.token_embedding.register_forward_hook(capture_input)
        expected_logits, expected_loss = baseline(self.tokens, self.targets)
        expected_loss.backward()
        handle.remove()
        input_gradient = captured[0].grad.clone()
        parameters = dict(baseline.named_parameters())
        for method in METHODS[1:]:
            with self.subTest(method=method):
                model = make_model(method, dtype=torch.float32)
                for name, value in baseline.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                for router in all_routers(model):
                    self.assertEqual(torch.count_nonzero(router.bias).item(), 0)
                    self.assertEqual(torch.count_nonzero(router.weight).item(), 0)
                captured.clear()
                handle = model.token_embedding.register_forward_hook(capture_input)
                logits, loss = model(self.tokens, self.targets)
                loss.backward()
                handle.remove()
                torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
                torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
                # Equivalent backward graphs can sum contributions in a
                # different order; forward equality above remains bitwise.
                torch.testing.assert_close(captured[0].grad, input_gradient, rtol=2e-6, atol=1e-7)
                # Includes the total tied embedding/head gradient, not only
                # the embedding lookup's input-path contribution.
                actual_parameters = dict(model.named_parameters())
                for name, parameter in parameters.items():
                    torch.testing.assert_close(
                        actual_parameters[name].grad, parameter.grad, rtol=2e-6, atol=1e-7
                    )

    def test_float64_phase_input_and_total_embedding_gradients_match_baseline(self):
        baseline = make_model("baseline")
        captured = []

        def capture_input(module, args, output):
            output.retain_grad()
            captured.append(output)

        handle = baseline.token_embedding.register_forward_hook(capture_input)
        expected_logits, expected_loss = baseline(self.tokens, self.targets)
        expected_loss.backward()
        handle.remove()
        expected_input_gradient = captured[0].grad.clone()
        expected_parameters = dict(baseline.named_parameters())
        for method in ("phase-adjoint", "phase-adjoint-post-frozen", "boundary-skip",
                       "terminal-adjoint"):
            with self.subTest(method=method):
                model = make_model(method)
                captured.clear()
                handle = model.token_embedding.register_forward_hook(capture_input)
                logits, loss = model(self.tokens, self.targets)
                loss.backward()
                handle.remove()
                torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
                torch.testing.assert_close(
                    captured[0].grad, expected_input_gradient, rtol=2e-12, atol=1e-13
                )
                for name, parameter in model.named_parameters():
                    if name in expected_parameters:
                        torch.testing.assert_close(
                            parameter.grad, expected_parameters[name].grad,
                            rtol=2e-12, atol=1e-13,
                        )

    def test_phase_boundary_contract_and_default(self):
        self.assertEqual(make_model().boundary, 2)
        for boundary in (1, 3):
            model = make_model(boundary=boundary)
            self.assertEqual(model.boundary, boundary)
            self.assertEqual(model.carrier, "zero")
        for boundary in (0, 4, -1, True, 1.5):
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                make_model(boundary=boundary)
        with self.assertRaises(ValueError):
            make_model(num_layers=1)

    def test_phase_prewriters_have_ntp_gradients_zero_r4_and_frozen_control_do_not(self):
        live = make_model()
        frozen = make_model("phase-adjoint-frozen")
        zero_r4 = make_model("adjoint", carrier="zero")
        for model in (live, frozen, zero_r4):
            _, loss = model(self.tokens, self.targets)
            loss.backward()
        for routers in (live.attn_routers, live.mlp_routers):
            for router in routers[:live.boundary]:
                self.assertGreater(router.weight.grad.norm().item(), 1e-8)
                self.assertTrue(torch.isfinite(router.bias.grad))
        for routers in (frozen.attn_routers, frozen.mlp_routers):
            for router in routers[:frozen.boundary]:
                self.assertEqual(torch.count_nonzero(router.weight.grad).item(), 0)
                self.assertEqual(torch.count_nonzero(router.bias.grad).item(), 0)
        for router in all_routers(zero_r4):
            self.assertEqual(torch.count_nonzero(router.weight.grad).item(), 0)
            self.assertEqual(torch.count_nonzero(router.bias.grad).item(), 0)

    def test_gain_is_a_live_first_order_single_stream_control(self):
        model = make_model("gain")
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        self.assertEqual(model.n_streams, 1)
        for router in branch_routers(model):
            self.assertGreater(router.weight.grad.norm().item(), 1e-8)
            self.assertTrue(torch.isfinite(router.bias.grad))

    def test_terminal_each_body_writer_gradient_is_the_output_adjoint_dot_innovation(self):
        model = make_model("terminal-adjoint")
        entries = []
        for index, block in enumerate(model.blocks):
            entries.extend(((model.attn_routers[index], block.attn),
                            (model.mlp_routers[index], block.mlp)))
        inputs, deltas, final_inputs = {}, {}, []
        handles = []
        for index, (router, branch) in enumerate(entries):
            handles.append(router.register_forward_pre_hook(
                lambda module, args, index=index: inputs.__setitem__(index, args[0].detach())
            ))
            handles.append(branch.register_forward_hook(
                lambda module, args, output, index=index: deltas.__setitem__(index, output.detach())
            ))

        def capture_final_input(module, args):
            args[0].retain_grad()
            final_inputs.append(args[0])

        handles.append(model.norm_final.register_forward_pre_hook(capture_final_input))
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        for handle in handles:
            handle.remove()
        output_adjoint = final_inputs[0].grad
        for index, (router, _) in enumerate(entries):
            inner = (output_adjoint * deltas[index]).sum(dim=-1, keepdim=True)
            features = inputs[index] * torch.rsqrt(
                inputs[index].square().mean(dim=-1, keepdim=True) + router.eps
            )
            predicted_weight = (inner * features).sum(dim=(0, 1)) * router.scale
            with self.subTest(writer=index):
                self.assertGreater(router.weight.grad.norm().item(), 1e-8)
                torch.testing.assert_close(router.bias.grad, inner.sum(), rtol=1e-11, atol=1e-13)
                torch.testing.assert_close(router.weight.grad, predicted_weight, rtol=1e-11, atol=1e-13)
        # m_terminal=-h_terminal at initialization: exit routing is radial.
        predicted_exit_bias = -(output_adjoint * final_inputs[0].detach()).sum()
        torch.testing.assert_close(
            model.output_router.bias.grad, predicted_exit_bias, rtol=1e-10, atol=1e-13
        )

    def test_terminal_switch_occurs_after_all_body_updates_before_output_router(self):
        terminal = make_model("terminal-adjoint")
        activate_routers(terminal)
        ordinary_body = make_model("adjoint", carrier="zero")
        ordinary_body.load_state_dict(terminal.state_dict())
        output_inputs = []
        handle = terminal.output_router.register_forward_pre_hook(
            lambda module, args: output_inputs.append(args[0].detach())
        )
        states = terminal.get_stream_states(self.tokens)
        handle.remove()
        body_states = ordinary_body.get_stream_states(self.tokens)
        self.assertEqual(terminal.switch_location, "after-body")
        self.assertFalse(hasattr(terminal, "boundary"))
        self.assertEqual(len(states), 2 + 2 * terminal.num_layers)
        self.assertEqual(len(body_states), len(states) - 1)
        for actual, expected in zip(states[:-1], body_states):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(states[-1], torch.stack(phase_switch(*states[-2])), rtol=0, atol=0)
        torch.testing.assert_close(output_inputs[0], states[-1][0], rtol=0, atol=0)
        self.assertGreater((states[-1] - states[-2]).norm().item(), 1e-6)
        one_layer = make_model("terminal-adjoint", num_layers=1)
        self.assertEqual(len(one_layer.get_stream_states(self.tokens)), 4)

    def test_terminal_exit_signal_is_suppressed_by_exact_scale_invariant_rmsnorm(self):
        model = make_model("terminal-adjoint")
        model.norm_final.eps = 0.0
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        self.assertLess(model.output_router.weight.grad.norm().item(), 1e-12)
        self.assertLess(model.output_router.bias.grad.abs().item(), 1e-12)
        self.assertTrue(all(router.weight.grad.norm().item() > 1e-8 for router in branch_routers(model)))

    def test_terminal_nonzero_routes_support_finite_ntp_optimizer_steps(self):
        model = make_model("terminal-adjoint")
        activate_routers(model)
        before = model.attn_routers[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(self.tokens, self.targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ))
            optimizer.step()
        self.assertGreater((model.attn_routers[0].weight.detach() - before).norm().item(), 0)

    def test_strong_skip_and_post_frozen_controls_match_initial_prewriter_gradients(self):
        live = make_model()
        controls = [make_model("phase-adjoint-post-frozen"), make_model("boundary-skip")]
        for model in (live, *controls):
            _, loss = model(self.tokens, self.targets)
            loss.backward()
        for control in controls:
            for attribute in ("attn_routers", "mlp_routers"):
                actual_routers = getattr(control, attribute)
                for index, expected in enumerate(getattr(live, attribute)[:live.boundary]):
                    actual = actual_routers[index]
                    self.assertGreater(actual.weight.grad.norm().item(), 1e-8)
                    torch.testing.assert_close(actual.weight.grad, expected.weight.grad, rtol=0, atol=0)
                    torch.testing.assert_close(actual.bias.grad, expected.bias.grad, rtol=0, atol=0)
        post_frozen_parameters = dict(controls[0].named_parameters())
        for name, parameter in live.named_parameters():
            torch.testing.assert_close(
                post_frozen_parameters[name].grad, parameter.grad, rtol=0, atol=0
            )

    def test_strong_controls_have_distinct_functions_when_routes_activate(self):
        live = make_model()
        activate_routers(live)
        expected, _ = live(self.tokens)
        for method in ("phase-adjoint-post-frozen", "boundary-skip"):
            with self.subTest(method=method):
                model = make_model(method)
                model.load_state_dict(live.state_dict())
                actual, _, states = model(self.tokens, return_stream_states=True)
                self.assertGreater((actual - expected).norm().item(), 1e-7)
                switch_index = 1 + 2 * model.boundary
                self.assertGreater((states[switch_index - 1][1] - states[0][1]).norm().item(), 1e-6)
                for state in states[switch_index:]:
                    torch.testing.assert_close(state[1], states[switch_index][1], rtol=0, atol=0)

    def test_frozen_aux_freezes_branch_writes_but_still_builds_boundary_snapshot(self):
        model = make_model("phase-adjoint-frozen")
        activate_routers(model)
        states = model.get_stream_states(self.tokens)
        switch_index = 1 + 2 * model.boundary
        self.assertEqual(len(states), 2 + 2 * model.num_layers)
        for state in states[:switch_index]:
            torch.testing.assert_close(state[1], torch.zeros_like(state[1]), rtol=0, atol=0)
        torch.testing.assert_close(
            states[switch_index][1], -states[switch_index - 1][0], rtol=0, atol=0
        )
        for state in states[switch_index:]:
            torch.testing.assert_close(state[1], states[switch_index][1], rtol=0, atol=0)

    def test_zero_branches_preserve_physical_state_with_an_explicit_frame_change(self):
        for method in ("phase-adjoint", "phase-adjoint-frozen", "phase-adjoint-shear",
                       "phase-adjoint-post-frozen", "boundary-skip", "terminal-adjoint"):
            with self.subTest(method=method):
                model = make_model(method)
                activate_routers(model)
                with torch.no_grad():
                    for block in model.blocks:
                        block.attn.o_proj.weight.zero_()
                        block.mlp.proj.weight.zero_()
                states = model.get_stream_states(self.tokens)
                switch_index = (len(states) - 1 if method == "terminal-adjoint"
                                else 1 + 2 * model.boundary)
                for state in states[:switch_index]:
                    torch.testing.assert_close(state, states[0], rtol=0, atol=0)
                expected = torch.stack(phase_switch(*states[0]))
                self.assertGreater((expected - states[0]).norm().item(), 0)
                for state in states[switch_index:]:
                    torch.testing.assert_close(state, expected, rtol=0, atol=0)
                    global_state = torch.stack(((state[0] - state[1]) / 2,
                                                (state[0] + state[1]) / 2))
                    torch.testing.assert_close(global_state, states[0], rtol=0, atol=0)

    def test_global_reciprocal_program_and_frame_program_match_outputs_and_gradients(self):
        frame = make_model()
        activate_routers(frame)
        global_model = copy.deepcopy(frame)
        actual_logits, actual_loss = frame(self.tokens, self.targets)
        expected_logits, expected_loss = global_coordinate_forward(
            global_model, self.tokens, self.targets
        )
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-13, atol=1e-13)
        actual_loss.backward()
        expected_loss.backward()
        expected_parameters = dict(global_model.named_parameters())
        for name, parameter in frame.named_parameters():
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    parameter.grad, expected_parameters[name].grad,
                    rtol=1e-10, atol=1e-12,
                )

    def test_shear_control_has_unit_self_action_and_explicit_orthogonal_write(self):
        model = make_model("phase-adjoint-shear")
        router = model.attn_routers[0]
        with torch.no_grad():
            router.weight.normal_(std=0.3)
            router.bias.fill_(0.2)
            router.shear.weight.normal_(std=0.3)
            router.shear.bias.fill_(0.4)
        h = torch.randn(2, 8, 32, dtype=torch.float64)
        m = torch.randn_like(h)
        delta = torch.randn_like(h)
        c, shear = router(h), router.shear(h)
        r = torch.cat((-c[..., 1:], c[..., :1]), dim=-1)
        write = c + shear * r
        torch.testing.assert_close(
            (c * write).sum(-1), torch.ones(2, 8, dtype=torch.float64),
            rtol=0, atol=6e-16,
        )
        next_h, next_m = model._branch_step(
            h, m, router, torch.nn.Identity(), lambda value: delta
        )
        torch.testing.assert_close(
            c[..., :1] * next_h + c[..., 1:] * next_m,
            c[..., :1] * h + c[..., 1:] * m + delta,
            rtol=1e-13, atol=1e-13,
        )
        torch.testing.assert_close(
            r[..., :1] * next_h + r[..., 1:] * next_m
            - r[..., :1] * h - r[..., 1:] * m,
            shear * delta, rtol=1e-13, atol=1e-13,
        )
        with torch.no_grad():
            router.shear.weight.zero_()
            router.shear.bias.zero_()
        actual = model._branch_step(h, m, router, torch.nn.Identity(), lambda value: delta)
        expected = adjoint_step(h, m, delta, c)
        for actual_state, expected_state in zip(actual, expected):
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)

    def test_scalar_and_unit_routers_use_identical_gate_features(self):
        h = torch.randn(2, 8, 32, dtype=torch.float64)
        for routing in ("dynamic", "static"):
            scalar = ScalarResidualRouter(32, routing).double()
            unit = UnitAddressRouter(32, routing).double()
            with torch.no_grad():
                scalar.bias.fill_(-0.3)
                if scalar.weight is not None:
                    scalar.weight.normal_(std=0.4)
            unit.load_state_dict(scalar.state_dict())
            c = unit(h)
            torch.testing.assert_close(scalar(h), c[..., 1:] / c[..., :1], rtol=1e-14, atol=1e-14)

    def test_nonzero_routes_preserve_future_token_causality_and_batch_isolation(self):
        for method in ("gain", "adjoint-shear", "phase-adjoint",
                       "phase-adjoint-frozen", "phase-adjoint-shear",
                       "phase-adjoint-post-frozen", "boundary-skip", "terminal-adjoint"):
            with self.subTest(method=method):
                model = make_model(method).eval()
                activate_routers(model)
                changed_future = self.tokens.clone()
                changed_future[:, 4:] = (changed_future[:, 4:] + 1) % model.vocab_size
                original, _ = model(self.tokens)
                changed, _ = model(changed_future)
                torch.testing.assert_close(original[:, :4], changed[:, :4], rtol=0, atol=0)
                self.assertGreater((original[:, 4:] - changed[:, 4:]).norm().item(), 1e-6)
                changed_neighbor = self.tokens.clone()
                changed_neighbor[1] = (changed_neighbor[1] + 7) % model.vocab_size
                replaced, _ = model(changed_neighbor)
                alone, _ = model(self.tokens[:1])
                torch.testing.assert_close(original[0], replaced[0], rtol=0, atol=0)
                torch.testing.assert_close(original[:1], alone, rtol=1e-12, atol=1e-13)

    def test_bfloat16_execution_keeps_state_dtype_and_all_gradients_finite(self):
        for method in ("gain", "adjoint-shear", "phase-adjoint",
                       "phase-adjoint-frozen", "phase-adjoint-shear",
                       "phase-adjoint-post-frozen", "boundary-skip", "terminal-adjoint"):
            with self.subTest(method=method):
                model = make_model(method, dtype=torch.bfloat16)
                activate_routers(model)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    logits, loss, states = model(
                        self.tokens, self.targets, return_stream_states=True
                    )
                    address = model.attn_routers[0](states[0][0])
                self.assertEqual(address.dtype, torch.float32)
                self.assertEqual(logits.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(loss))
                for state in states:
                    self.assertEqual(state.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(state).all())
                loss.backward()
                self.assertTrue(all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ))

    def test_one_ntp_step_and_state_dict_roundtrip(self):
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _, loss = model(self.tokens, self.targets)
        loss.backward()
        optimizer.step()
        self.assertGreater(model.attn_routers[0].weight.norm().item(), 0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = make_model(boundary=model.boundary)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        expected, expected_loss = model(self.tokens, self.targets)
        actual, actual_loss = restored(self.tokens, self.targets)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)

    def test_factory_control_identity_parameter_counts_and_lazy_attnres(self):
        baseline = make_model("baseline")
        backbone_count = baseline.count_parameters()
        scalar_parameters = MODEL_KWARGS["d_model"] + 1
        branches = 2 * MODEL_KWARGS["num_layers"]
        expected_extra = {
            "gain": branches * scalar_parameters,
            "phase-adjoint": (branches + 1) * scalar_parameters,
            "phase-adjoint-frozen": (branches + 1) * scalar_parameters,
            "phase-adjoint-post-frozen": (branches + 1) * scalar_parameters,
            "boundary-skip": (branches + 1) * scalar_parameters,
            "terminal-adjoint": (branches + 1) * scalar_parameters,
            "phase-adjoint-shear": (2 * branches + 1) * scalar_parameters,
        }
        for method, count in expected_extra.items():
            model = make_model(method)
            self.assertEqual(model.count_parameters() - backbone_count, count)
        self.assertIsInstance(make_model("gain"), ResidualGainTransformer)
        self.assertIsInstance(make_model("phase-adjoint"), PhaseAdjointHCTransformer)
        self.assertIsInstance(make_model("boundary-skip"), BoundarySkipTransformer)
        self.assertIsInstance(make_model("terminal-adjoint"), TerminalAdjointHCTransformer)
        self.assertIsInstance(make_model("adjoint-shear"), ControlledAdjointHCTransformer)
        for method, kwargs in (
            ("phase-adjoint", {"freeze_aux_updates": True}),
            ("phase-adjoint-frozen", {"freeze_aux_updates": False}),
            ("phase-adjoint", {"carrier": "signed"}),
            ("adjoint-shear", {"shear_control": False}),
            ("gain", {"shear_control": True}),
            ("phase-adjoint-post-frozen", {"freeze_post_aux_updates": False}),
            ("boundary-skip", {"freeze_post_aux_updates": False}),
            ("terminal-adjoint", {"carrier": "signed"}),
            ("terminal-adjoint", {"freeze_aux_updates": True}),
            ("unknown", {}),
        ):
            with self.subTest(method=method, kwargs=kwargs), self.assertRaises(ValueError):
                make_model(method, **kwargs)
        module = types.ModuleType("lm.block_attnres")
        module.BlockAttnResTransformer = BaselineTransformer
        with patch.dict(sys.modules, {"lm.block_attnres": module}):
            lazy_model = create_residual_model("block-attnres", **MODEL_KWARGS)
        self.assertIsInstance(lazy_model, BaselineTransformer)


if __name__ == "__main__":
    unittest.main()
