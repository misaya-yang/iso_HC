"""Two-stream residual candidate with one unit address for reading and writing.

For each attention or MLP sublayer, c = (1, tanh(g(x))) / ||(1, tanh(g(x)))||,
z = c[0] * x + c[1] * m, and (x, m) += c * F(Norm(z)). The untouched
state path is the identity; this is not a whole-network stability guarantee.
Zero router parameters reproduce the baseline function at initialization.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import BaselineTransformer


class UnitAddressRouter(nn.Module):
    """A scalar linear gate and a unit two-stream address for every token.

    Address arithmetic uses float32 for reduced-precision input and preserves
    float64 input. There is no learned normalization or auxiliary objective.
    ``static`` retains a learned scalar bias but removes input dependence.
    """

    def __init__(self, hidden_dim, routing="dynamic", eps=1e-6):
        super().__init__()
        if routing not in ("dynamic", "static"):
            raise ValueError(f"Unknown routing: {routing}")
        self.routing = routing
        self.eps = eps
        self.scale = 1.0 / math.sqrt(hidden_dim)
        self.bias = nn.Parameter(torch.zeros(()))
        if routing == "dynamic":
            self.weight = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        # Autocast must not silently reduce address arithmetic back to bf16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.weight is not None:
                state = x.to(dtype=dtype)
                normed = state * torch.rsqrt(
                    state.square().mean(dim=-1, keepdim=True) + self.eps
                )
                score = (normed * self.weight.to(dtype=dtype)).sum(dim=-1, keepdim=True)
                score = score * self.scale + self.bias.to(dtype=dtype)
            else:
                score = self.bias.to(dtype=dtype).expand(*x.shape[:-1], 1)
            t = torch.tanh(score)
            c0 = torch.rsqrt(1.0 + t.square())
            return torch.cat((c0, t * c0), dim=-1)


def adjoint_read(x, m, c):
    """Read the addressed state; ``c`` has shape ``(..., 2)``.

    Cast the computed address to the state dtype for state arithmetic, so a
    fully bf16 model does not silently promote its whole residual workspace.
    """
    address = c.to(dtype=x.dtype)
    return address[..., :1] * x + address[..., 1:] * m


def adjoint_step(x, m, delta, c):
    """Write one branch update with the same address used by adjoint_read."""
    address = c.to(dtype=x.dtype)
    return x + address[..., :1] * delta, m + address[..., 1:] * delta


class AdjointHCTransformer(BaselineTransformer):
    """A complete LM implementation of the two-stream residual candidate.

    Baseline modules and state-dict keys are preserved. New zero-initialized
    routers consume no random draws, so an identical seed produces identical
    baseline parameters. The signed carrier starts as D * (token + position),
    with a fixed balanced alternating-sign feature mask and no new embedding.

    ``zero``/``copy`` carriers and ``freeze_aux_updates`` are mechanism controls.
    Freezing auxiliary writes deliberately breaks the tied read/write rule.
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
        routing="dynamic",
        carrier="signed",
        freeze_aux_updates=False,
    ):
        if carrier not in ("signed", "zero", "copy"):
            raise ValueError(f"Unknown carrier: {carrier}")
        if carrier == "signed" and d_model % 2:
            raise ValueError("A balanced signed carrier requires an even d_model")
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
        self.n_streams = 2
        self.routing = routing
        self.carrier = carrier
        self.freeze_aux_updates = freeze_aux_updates
        mask = torch.ones(d_model)
        if carrier == "signed":
            mask[1::2] = -1
        elif carrier == "zero":
            mask.zero_()
        self.register_buffer("carrier_mask", mask)
        self.attn_routers = nn.ModuleList([
            UnitAddressRouter(d_model, routing) for _ in range(num_layers)
        ])
        self.mlp_routers = nn.ModuleList([
            UnitAddressRouter(d_model, routing) for _ in range(num_layers)
        ])
        self.output_router = UnitAddressRouter(d_model, routing)

    def _branch_step(self, x, m, router, norm, branch):
        c = router(x)
        delta = branch(norm(adjoint_read(x, m, c)))
        if self.freeze_aux_updates:
            return x + c[..., :1].to(dtype=x.dtype) * delta, m
        return adjoint_step(x, m, delta, c)

    def forward(self, input_ids, targets=None, return_stream_states=False):
        _, length = input_ids.shape
        assert length <= self.context_length
        position = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.pos_embedding(position)
        m = x * self.carrier_mask
        states = [torch.stack((x, m))] if return_stream_states else None

        for block, attn_router, mlp_router in zip(
            self.blocks, self.attn_routers, self.mlp_routers
        ):
            x, m = self._branch_step(x, m, attn_router, block.norm1, block.attn)
            if states is not None:
                states.append(torch.stack((x, m)))
            x, m = self._branch_step(x, m, mlp_router, block.norm2, block.mlp)
            if states is not None:
                states.append(torch.stack((x, m)))

        z = adjoint_read(x, m, self.output_router(x))
        logits = self.lm_head(self.norm_final(z))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100
            )
        if return_stream_states:
            return logits, loss, states
        return logits, loss

    @torch.no_grad()
    def get_stream_states(self, input_ids):
        """Diagnostic-only snapshots; ordinary forward retains no state list."""
        _, _, states = self.forward(input_ids, return_stream_states=True)
        return states
