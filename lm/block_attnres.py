"""Block Attention Residuals on the existing Transformer backbone.

Mechanism comparator for https://arxiv.org/pdf/2603.15031, not its production
kernel or complete training recipe. Each destination has a learned pseudo-query
and key RMSNorm. Zero queries read the source mean; finite-epsilon branch norms
do not make this an exact floating-point baseline initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import BaselineTransformer, RMSNorm


class AttnResReadout(nn.Module):
    """Softmax over depth sources independently for each sample and token."""

    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_dim))
        self.key_norm = RMSNorm(hidden_dim, eps=eps)

    def forward(self, sources, return_weights=False):
        values = torch.stack(sources, dim=0)  # sources, batch, tokens, features
        dtype = (
            torch.float32
            if values.dtype in (torch.float16, torch.bfloat16)
            else values.dtype
        )
        # Accumulate key statistics, logits and weighted values in fp32 for
        # half storage. Keep float64 for contract checks and return state dtype.
        with torch.autocast(device_type=values.device.type, enabled=False):
            work = values.to(dtype=dtype)
            keys = work * torch.rsqrt(
                work.square().mean(dim=-1, keepdim=True) + self.key_norm.eps
            )
            keys = keys * self.key_norm.weight.to(dtype=dtype)
            scores = (keys * self.query.to(dtype=dtype)).sum(dim=-1)
            weights = torch.softmax(scores, dim=0)
            result = (weights.unsqueeze(-1) * work).sum(dim=0)
        result = result.to(dtype=values.dtype)
        if return_weights:
            return result, weights
        return result


class BlockAttnResTransformer(BaselineTransformer):
    """Embedding, completed block sums and one mutable partial block as sources.

    ``block_size`` counts complete Transformer blocks, each containing attention
    and MLP destinations. Branch outputs accumulate only into the current block
    sum. At a block boundary that sum becomes one immutable history source and
    the partial state is reset; no ordinary residual skip is added to the read.

    Baseline backbone modules, state-dict keys and constructor RNG consumption
    are preserved. Additional queries are zero and key-norm weights are one.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        context_length,
        mlp_ratio=4,
        dropout=0.0,
        use_flash=True,
        head_mixing_type=None,
        head_mixing_kwargs=None,
        block_size=4,
    ):
        if (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            context_length=context_length,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_flash=use_flash,
            head_mixing_type=head_mixing_type,
            head_mixing_kwargs=head_mixing_kwargs,
        )
        self.block_size = block_size
        self.attn_readouts = nn.ModuleList([
            AttnResReadout(d_model) for _ in range(num_layers)
        ])
        self.mlp_readouts = nn.ModuleList([
            AttnResReadout(d_model) for _ in range(num_layers)
        ])
        self.output_readout = AttnResReadout(d_model)

    def forward(self, input_ids, targets=None):
        _, length = input_ids.shape
        assert length <= self.context_length
        position = torch.arange(length, device=input_ids.device)
        embedding = self.token_embedding(input_ids) + self.pos_embedding(position)
        completed = [embedding]
        partial = None

        for layer, (block, attn_readout, mlp_readout) in enumerate(zip(
            self.blocks, self.attn_readouts, self.mlp_readouts
        )):
            sources = completed if partial is None else completed + [partial]
            read = attn_readout(sources)
            delta = block.attn(block.norm1(read))
            partial = delta if partial is None else partial + delta

            read = mlp_readout(completed + [partial])
            partial = partial + block.mlp(block.norm2(read))

            if (layer + 1) % self.block_size == 0:
                completed.append(partial)
                partial = None

        sources = completed if partial is None else completed + [partial]
        read = self.output_readout(sources)
        logits = self.lm_head(self.norm_final(read))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100
            )
        return logits, loss
