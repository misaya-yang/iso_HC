"""Source-list contracts for the Block AttnRes mechanism comparator.

These small CPU checks establish mechanics, gradients and isolation. They do
not reproduce the original training recipe or measure language-model quality.
"""

import copy
import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lm.block_attnres import AttnResReadout, BlockAttnResTransformer
from lm.models import BaselineTransformer


MODEL_KWARGS = dict(
    vocab_size=31,
    d_model=16,
    num_layers=3,
    num_heads=4,
    context_length=8,
    mlp_ratio=2,
    dropout=0.0,
    use_flash=False,
)


def readouts(model):
    return [*model.attn_readouts, *model.mlp_readouts, model.output_readout]


def make_model(block_size=2):
    torch.manual_seed(1729)
    return BlockAttnResTransformer(**MODEL_KWARGS, block_size=block_size).double()


def explicit_read(sources, readout):
    keys = [
        value / torch.sqrt(value.square().mean(-1, keepdim=True) + readout.key_norm.eps)
        * readout.key_norm.weight
        for value in sources
    ]
    scores = torch.stack([(key * readout.query).sum(-1) for key in keys])
    exponentials = torch.exp(scores - scores.max(dim=0, keepdim=True).values)
    weights = exponentials / exponentials.sum(dim=0, keepdim=True)
    return sum(weight.unsqueeze(-1) * value for weight, value in zip(weights, sources))


def explicit_forward(model, input_ids, targets):
    """Rebuild each source list from raw deltas instead of a mutable partial."""
    position = torch.arange(input_ids.shape[1], device=input_ids.device)
    embedding = model.token_embedding(input_ids) + model.pos_embedding(position)
    deltas = []
    group_size = 2 * model.block_size

    def sources():
        result = [embedding]
        for start in range(0, len(deltas), group_size):
            result.append(torch.stack(deltas[start:start + group_size]).sum(dim=0))
        return result

    for block, attn_readout, mlp_readout in zip(
        model.blocks, model.attn_readouts, model.mlp_readouts
    ):
        read = explicit_read(sources(), attn_readout)
        deltas.append(block.attn(block.norm1(read)))
        read = explicit_read(sources(), mlp_readout)
        deltas.append(block.mlp(block.norm2(read)))
    read = explicit_read(sources(), model.output_readout)
    logits = model.lm_head(model.norm_final(read))
    loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size), targets.reshape(-1), ignore_index=-100
    )
    return logits, loss


class ConstantBranch(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, state):
        return torch.full_like(state, self.value)


class BlockAttnResTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(23)
        self.tokens = torch.randint(0, MODEL_KWARGS["vocab_size"], (2, 8))
        self.targets = torch.roll(self.tokens, shifts=-1, dims=1)

    def test_same_seed_backbone_and_rng_with_zero_queries_and_learned_key_norms(self):
        torch.manual_seed(1729)
        baseline = BaselineTransformer(**MODEL_KWARGS).double()
        expected_rng = torch.get_rng_state()
        model = make_model()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, value in baseline.state_dict().items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        for readout in readouts(model):
            self.assertEqual(torch.count_nonzero(readout.query).item(), 0)
            torch.testing.assert_close(
                readout.key_norm.weight, torch.ones_like(readout.key_norm.weight),
                rtol=0, atol=0,
            )
            self.assertTrue(readout.key_norm.weight.requires_grad)
        self.assertEqual(BlockAttnResTransformer(**MODEL_KWARGS).block_size, 4)

    def test_zero_query_reads_mean_and_softmax_normalizes_over_sources(self):
        readout = AttnResReadout(5).double()
        sources = [torch.randn(2, 7, 5, dtype=torch.float64) for _ in range(3)]
        with torch.no_grad():
            readout.key_norm.weight.copy_(torch.arange(1, 6, dtype=torch.float64))
        read, weights = readout(sources, return_weights=True)
        torch.testing.assert_close(read, torch.stack(sources).mean(0))
        self.assertEqual(weights.shape, (3, 2, 7))
        torch.testing.assert_close(weights, torch.full_like(weights, 1 / 3))
        with torch.no_grad():
            readout.query.copy_(torch.tensor([0.2, -0.4, 0.1, 0.5, -0.3]))
        read, weights = readout(sources, return_weights=True)
        torch.testing.assert_close(weights.sum(0), torch.ones(2, 7, dtype=torch.float64))
        torch.testing.assert_close(read, explicit_read(sources, readout), rtol=1e-13, atol=1e-13)
        self.assertGreater((weights - 1 / 3).abs().max().item(), 0.01)

    def test_explicit_raw_delta_reference_matches_full_forward_and_backward(self):
        for block_size in (1, 2, 4):
            with self.subTest(block_size=block_size):
                model = make_model(block_size)
                with torch.no_grad():
                    for readout in readouts(model):
                        readout.query.normal_(std=0.4)
                        readout.key_norm.weight.uniform_(0.6, 1.4)
                reference = copy.deepcopy(model)
                logits, loss = model(self.tokens, self.targets)
                expected_logits, expected_loss = explicit_forward(
                    reference, self.tokens, self.targets
                )
                torch.testing.assert_close(logits, expected_logits, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)
                loss.backward()
                expected_loss.backward()
                expected_parameters = dict(reference.named_parameters())
                for name, parameter in model.named_parameters():
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        torch.testing.assert_close(
                            parameter.grad, expected_parameters[name].grad,
                            rtol=1e-10, atol=1e-12,
                        )
                self.assertGreater(model.output_readout.query.grad.norm().item(), 1e-8)
                self.assertGreater(model.output_readout.key_norm.weight.grad.norm().item(), 1e-8)

    def test_completed_blocks_and_partial_are_not_duplicated_or_skipped(self):
        model = make_model(block_size=2)
        for layer, block in enumerate(model.blocks):
            block.attn = ConstantBranch(2 * layer + 1)
            block.mlp = ConstantBranch(2 * layer + 2)
        captured = []
        hook = model.output_readout.register_forward_pre_hook(
            lambda module, args: captured.append(list(args[0]))
        )
        try:
            logits, _ = model(self.tokens)
        finally:
            hook.remove()
        embedding = model.token_embedding(self.tokens) + model.pos_embedding(torch.arange(8))
        sources = captured[0]
        self.assertEqual(len(sources), 3)
        torch.testing.assert_close(sources[0], embedding, rtol=0, atol=0)
        torch.testing.assert_close(sources[1], torch.full_like(embedding, 10), rtol=0, atol=0)
        torch.testing.assert_close(sources[2], torch.full_like(embedding, 11), rtol=0, atol=0)
        expected_read = (embedding + 21) / 3
        expected_logits = model.lm_head(model.norm_final(expected_read))
        torch.testing.assert_close(logits, expected_logits, rtol=1e-13, atol=1e-13)

    def test_source_counts_follow_block_boundaries_and_not_token_length(self):
        for block_size in (1, 2, 4):
            model = make_model(block_size)
            expected = []
            for layer in range(MODEL_KWARGS["num_layers"]):
                completed = 1 + layer // block_size
                expected.extend([completed + int(layer % block_size != 0), completed + 1])
            expected.append(1 + math.ceil(MODEL_KWARGS["num_layers"] / block_size))
            for length in (3, 8):
                with self.subTest(block_size=block_size, length=length):
                    captured = []

                    def record(module, args):
                        sources = args[0]
                        captured.append(len(sources))
                        for source in sources:
                            self.assertEqual(source.shape, (2, length, MODEL_KWARGS["d_model"]))

                    hooks = [readout.register_forward_pre_hook(record) for readout in readouts(model)]
                    try:
                        model(self.tokens[:, :length])
                    finally:
                        for hook in hooks:
                            hook.remove()
                    self.assertEqual(captured, expected)

    def test_nonzero_queries_preserve_token_causality_and_batch_isolation(self):
        model = make_model().eval()
        with torch.no_grad():
            for readout in readouts(model):
                readout.query.normal_(std=0.5)
                readout.key_norm.weight.uniform_(0.7, 1.3)
        changed_future = self.tokens.clone()
        changed_future[:, 4:] = (changed_future[:, 4:] + 1) % MODEL_KWARGS["vocab_size"]
        logits, _ = model(self.tokens)
        changed, _ = model(changed_future)
        torch.testing.assert_close(logits[:, :4], changed[:, :4], rtol=0, atol=0)
        self.assertGreater((logits[:, 4:] - changed[:, 4:]).norm().item(), 1e-6)
        changed_neighbor = self.tokens.clone()
        changed_neighbor[1] = (changed_neighbor[1] + 7) % MODEL_KWARGS["vocab_size"]
        changed, _ = model(changed_neighbor)
        alone, _ = model(self.tokens[:1])
        torch.testing.assert_close(logits[0], changed[0], rtol=0, atol=0)
        torch.testing.assert_close(logits[:1], alone, rtol=1e-13, atol=1e-13)

    def test_half_storage_has_finite_fp32_key_statistics_and_full_backward(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                readout = AttnResReadout(4).to(dtype=dtype)
                with torch.no_grad():
                    readout.query.copy_(torch.tensor([0.2, -0.4, 0.1, 0.5]))
                sources = [
                    (1024 * torch.randn(2, 3, 4)).to(dtype=dtype).requires_grad_()
                    for _ in range(3)
                ]
                read, weights = readout(sources, return_weights=True)
                self.assertEqual(read.dtype, dtype)
                self.assertEqual(weights.dtype, torch.float32)
                self.assertTrue(torch.isfinite(read).all())
                self.assertTrue(torch.isfinite(weights).all())
                torch.testing.assert_close(weights.sum(0), torch.ones(2, 3), rtol=1e-6, atol=1e-6)
                read.float().mean().backward()
                for value in [*sources, readout.query, readout.key_norm.weight]:
                    self.assertTrue(torch.isfinite(value.grad).all())

                # The existing backbone's RMSNorm backward is not stable with
                # pure fp16 parameters. Exercise fp16 through fp32-parameter
                # autocast, and also verify a fully bf16-stored model.
                storage_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
                model = make_model().to(dtype=storage_dtype)
                with torch.no_grad():
                    for destination in readouts(model):
                        destination.query.normal_(std=0.3)
                with torch.autocast("cpu", dtype=dtype, enabled=(dtype == torch.float16)):
                    logits, loss = model(self.tokens, self.targets)
                self.assertEqual(logits.dtype, dtype)
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_block_size_is_rejected(self):
        for block_size in (0, -1, 1.5, True):
            with self.subTest(block_size=block_size):
                with self.assertRaises(ValueError):
                    BlockAttnResTransformer(**MODEL_KWARGS, block_size=block_size)


if __name__ == "__main__":
    unittest.main()
