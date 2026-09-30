"""Phase adjoint candidate and explicitly identified residual controls.

The phase model starts with a zero auxiliary stream and performs the fixed
coordinate change (h, m) -> (h + m, m - h) before one Transformer block. Each
branch in either frame uses the existing unit-address tied read/write operator.
Zero routers reproduce the baseline exactly, while a pre-boundary auxiliary
write can affect the post-boundary primary stream to first order.

The switch is a scaled rotation, not an identity carry in a single coordinate
frame. These contracts imply neither whole-network stability nor LM quality.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adjoint import AdjointHCTransformer, adjoint_read, adjoint_step
from .models import BaselineTransformer


class ScalarResidualRouter(nn.Module):
    """tanh scalar gate with the same features and initialization as R4.

    The gain and shear controls do not need the additional unit-address
    normalization. Reduced-precision gate arithmetic still uses float32.
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

    def forward(self, h):
        dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            if self.weight is None:
                score = self.bias.to(dtype=dtype).expand(*h.shape[:-1], 1)
            else:
                state = h.to(dtype=dtype)
                normed = state * torch.rsqrt(
                    state.square().mean(dim=-1, keepdim=True) + self.eps
                )
                score = (normed * self.weight.to(dtype=dtype)).sum(dim=-1, keepdim=True)
                score = score * self.scale + self.bias.to(dtype=dtype)
            return torch.tanh(score)


def phase_switch(h, m):
    """Change coordinates once; both outputs use the original inputs."""
    return h + m, m - h


class ControlledAdjointHCTransformer(AdjointHCTransformer):
    """R4 with an optional explicit shear control b = c + s * (-c1, c0).

    Shear gates are zero-initialized and have the same input features as the
    read routers. Their write vector satisfies c^T b = 1 in real arithmetic,
    but deliberately breaks strict adjoint alignment. Frozen auxiliary writes
    also deliberately break the main rule. Neither control is hidden in R4.
    """

    def __init__(self, *args, shear_control=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shear_control = shear_control
        if shear_control:
            for router in [*self.attn_routers, *self.mlp_routers]:
                router.shear = ScalarResidualRouter(self.d_model, self.routing)

    def _branch_step(self, h, m, router, norm, branch, freeze_aux_updates=None):
        frozen = self.freeze_aux_updates if freeze_aux_updates is None else freeze_aux_updates
        c = router(h)
        delta = branch(norm(adjoint_read(h, m, c)))
        if not self.shear_control:
            if frozen:
                return h + c[..., :1].to(dtype=h.dtype) * delta, m
            return adjoint_step(h, m, delta, c)
        shear = router.shear(h)
        perpendicular = torch.cat((-c[..., 1:], c[..., :1]), dim=-1)
        write = (c + shear * perpendicular).to(dtype=h.dtype)
        next_h = h + write[..., :1] * delta
        next_m = m if frozen else m + write[..., 1:] * delta
        return next_h, next_m


class PhaseAdjointHCTransformer(ControlledAdjointHCTransformer):
    """Zero-carrier two-stream model with one fixed phase coordinate switch.

    ``boundary`` is the number of completed Transformer blocks before the
    switch, defaults to num_layers // 2, and must satisfy 1 <= boundary < L.
    There is no learned scale or independent write gate in the main model.

    In global coordinates the post-switch read is
    (c0 - c1) * x + (c0 + c1) * m, with reciprocal half-scale writing. The frame
    program avoids irrational initialization constants and preserves exact
    baseline floating-point evaluation when routers are zero.

    Diagnostic snapshots include the initial state, every branch update, and
    one additional state immediately after the coordinate switch. Frozen-aux
    controls still execute that switch; only branch auxiliary writes freeze.
    ``freeze_post_aux_updates`` preserves all pre-boundary writes and freezes
    auxiliary branch writes only in the post-boundary frame.
    """

    def __init__(self, *args, boundary=None, freeze_post_aux_updates=False, **kwargs):
        super().__init__(*args, carrier="zero", **kwargs)
        if boundary is None:
            boundary = self.num_layers // 2
        if (
            not isinstance(boundary, int)
            or isinstance(boundary, bool)
            or not 1 <= boundary < self.num_layers
        ):
            raise ValueError("boundary must be an integer with 1 <= boundary < num_layers")
        self.boundary = boundary
        self.freeze_post_aux_updates = freeze_post_aux_updates

    def _pre_branch_step(self, h, m, router, norm, branch):
        return self._branch_step(h, m, router, norm, branch)

    def _post_branch_step(self, h, m, router, norm, branch):
        return self._branch_step(
            h, m, router, norm, branch,
            freeze_aux_updates=self.freeze_aux_updates or self.freeze_post_aux_updates,
        )

    def forward(self, input_ids, targets=None, return_stream_states=False):
        _, length = input_ids.shape
        assert length <= self.context_length
        position = torch.arange(length, device=input_ids.device)
        h = self.token_embedding(input_ids) + self.pos_embedding(position)
        m = torch.zeros_like(h)
        states = [torch.stack((h, m))] if return_stream_states else None

        for index, (block, attn_router, mlp_router) in enumerate(zip(
            self.blocks, self.attn_routers, self.mlp_routers
        )):
            if index == self.boundary:
                h, m = phase_switch(h, m)
                if states is not None:
                    states.append(torch.stack((h, m)))
            branch_step = self._pre_branch_step if index < self.boundary else self._post_branch_step
            h, m = branch_step(h, m, attn_router, block.norm1, block.attn)
            if states is not None:
                states.append(torch.stack((h, m)))
            h, m = branch_step(h, m, mlp_router, block.norm2, block.mlp)
            if states is not None:
                states.append(torch.stack((h, m)))

        z = adjoint_read(h, m, self.output_router(h))
        logits = self.lm_head(self.norm_final(z))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100
            )
        if return_stream_states:
            return logits, loss, states
        return logits, loss


class BoundarySkipTransformer(PhaseAdjointHCTransformer):
    """Strong HC/skip control with matched first-order pre-boundary writers.

    Before the boundary: delta = F(Norm(h)), h += delta, m += tanh(g(h))*delta.
    The same fixed coordinate switch then exposes this history to the ordinary
    unit reader. Post-boundary auxiliary writes are frozen. This control keeps
    the standard primary pre-boundary computation rather than tied c0 scaling
    or a memory-dependent branch read; it is not the proposed main method.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("freeze_aux_updates", False) or kwargs.get("shear_control", False):
            raise ValueError("boundary-skip requires live pre-writes and no shear")
        if not kwargs.pop("freeze_post_aux_updates", True):
            raise ValueError("boundary-skip requires frozen post-boundary auxiliary writes")
        super().__init__(*args, freeze_post_aux_updates=True, **kwargs)

    def _pre_branch_step(self, h, m, router, norm, branch):
        c = router(h)
        # c0 >= 1/sqrt(2). Recover the scalar before casting to state dtype.
        scalar = (c[..., 1:] / c[..., :1]).to(dtype=h.dtype)
        delta = branch(norm(h))
        return h + delta, m + scalar * delta


class TerminalAdjointHCTransformer(AdjointHCTransformer):
    """Necessary control: one fixed switch after every body branch completes.

    The body has zero auxiliary input and ordinary unit tied read/write. The
    terminal (h, m) -> (h + m, m - h) switch precedes the same unit output
    router used by R4/R5. It is explicit and independent of phase boundaries.
    At zero routers each body writer has a first-order history path to output.

    Initially the terminal auxiliary state is -h, so the output router changes
    the read radially and RMSNorm can suppress its initial learning signal.
    This is a control for delayed history exposure, not a quality conclusion.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.pop("carrier", "zero") != "zero":
            raise ValueError("terminal-adjoint requires a zero carrier")
        if kwargs.get("freeze_aux_updates", False):
            raise ValueError("terminal-adjoint requires live tied auxiliary writes")
        super().__init__(*args, carrier="zero", **kwargs)
        self.switch_location = "after-body"

    def forward(self, input_ids, targets=None, return_stream_states=False):
        _, length = input_ids.shape
        assert length <= self.context_length
        position = torch.arange(length, device=input_ids.device)
        h = self.token_embedding(input_ids) + self.pos_embedding(position)
        m = torch.zeros_like(h)
        states = [torch.stack((h, m))] if return_stream_states else None

        for block, attn_router, mlp_router in zip(
            self.blocks, self.attn_routers, self.mlp_routers
        ):
            h, m = self._branch_step(h, m, attn_router, block.norm1, block.attn)
            if states is not None:
                states.append(torch.stack((h, m)))
            h, m = self._branch_step(h, m, mlp_router, block.norm2, block.mlp)
            if states is not None:
                states.append(torch.stack((h, m)))

        # A terminal switch is outside the block loop; boundary=L would never
        # reach the internal phase switch and is not an implementation here.
        h, m = phase_switch(h, m)
        if states is not None:
            states.append(torch.stack((h, m)))
        z = adjoint_read(h, m, self.output_router(h))
        logits = self.lm_head(self.norm_final(z))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100
            )
        if return_stream_states:
            return logits, loss, states
        return logits, loss


class ResidualGainTransformer(BaselineTransformer):
    """Single-stream gain control h += (1 + tanh(g(h))) * F(Norm(h))."""

    def __init__(self, *args, routing="dynamic", **kwargs):
        super().__init__(*args, **kwargs)
        self.n_streams = 1
        self.routing = routing
        self.attn_routers = nn.ModuleList([
            ScalarResidualRouter(self.d_model, routing) for _ in self.blocks
        ])
        self.mlp_routers = nn.ModuleList([
            ScalarResidualRouter(self.d_model, routing) for _ in self.blocks
        ])

    def _branch_step(self, h, router, norm, branch):
        gain = (1 + router(h)).to(dtype=h.dtype)
        return h + gain * branch(norm(h))

    def forward(self, input_ids, targets=None, return_stream_states=False):
        _, length = input_ids.shape
        assert length <= self.context_length
        position = torch.arange(length, device=input_ids.device)
        h = self.token_embedding(input_ids) + self.pos_embedding(position)
        states = [h.unsqueeze(0)] if return_stream_states else None
        for block, attn_router, mlp_router in zip(
            self.blocks, self.attn_routers, self.mlp_routers
        ):
            h = self._branch_step(h, attn_router, block.norm1, block.attn)
            if states is not None:
                states.append(h.unsqueeze(0))
            h = self._branch_step(h, mlp_router, block.norm2, block.mlp)
            if states is not None:
                states.append(h.unsqueeze(0))
        logits = self.lm_head(self.norm_final(h))
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
        return self.forward(input_ids, return_stream_states=True)[2]


def create_residual_model(method, **model_kwargs):
    """Build scoped experiment arms without the historical runner/factory.

    All arms accept BaselineTransformer constructor arguments. ``routing`` may
    be shared across arms; ``boundary`` applies only to phase models. Method
    names fix control flags, and conflicting flags raise instead of silently
    changing the identity of an experiment. R4 arms default to signed carrier.
    Block AttnRes is imported only when requested from lm.block_attnres.
    """
    kwargs = dict(model_kwargs)
    boundary = kwargs.pop("boundary", None)
    if method in ("baseline", "gain", "block-attnres"):
        for flag in ("freeze_aux_updates", "shear_control", "freeze_post_aux_updates"):
            if kwargs.pop(flag, False):
                raise ValueError(f"{flag} is not a control for {method}")
        if method == "gain":
            return ResidualGainTransformer(**kwargs)
        kwargs.pop("routing", None)
        if method == "block-attnres":
            from .block_attnres import BlockAttnResTransformer

            return BlockAttnResTransformer(**kwargs)
        model = BaselineTransformer(**kwargs)
        model.n_streams = 1
        return model

    controls = {
        "adjoint": (False, False, False),
        "adjoint-frozen": (True, False, False),
        "adjoint-shear": (False, True, False),
        "phase-adjoint": (False, False, False),
        "phase-adjoint-frozen": (True, False, False),
        "phase-adjoint-shear": (False, True, False),
        "phase-adjoint-post-frozen": (False, False, True),
        "boundary-skip": (False, False, True),
        "terminal-adjoint": (False, False, False),
    }
    if method not in controls:
        raise ValueError(f"Unknown residual method: {method}")
    frozen, shear, post_frozen = controls[method]
    for flag, expected in (("freeze_aux_updates", frozen), ("shear_control", shear),
                           ("freeze_post_aux_updates", post_frozen)):
        supplied = kwargs.pop(flag, expected)
        if supplied != expected:
            raise ValueError(f"{method} requires {flag}={expected}")
    if method == "terminal-adjoint":
        return TerminalAdjointHCTransformer(**kwargs)
    if method.startswith("phase-") or method == "boundary-skip":
        if kwargs.pop("carrier", "zero") != "zero":
            raise ValueError("phase models require a zero carrier")
        model_class = BoundarySkipTransformer if method == "boundary-skip" else PhaseAdjointHCTransformer
        return model_class(
            boundary=boundary, freeze_aux_updates=frozen, shear_control=shear,
            freeze_post_aux_updates=post_frozen, **kwargs
        )
    if shear:
        return ControlledAdjointHCTransformer(shear_control=True, **kwargs)
    return AdjointHCTransformer(freeze_aux_updates=frozen, **kwargs)
