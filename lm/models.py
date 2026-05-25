r"""125M-scale Transformer models: baseline and HC variants.

HCTransformer (single-block multi-stream HC):
  X_0 = token_embed + pos_embed + stream_embed
  For each layer l:
    z_l = (1/s) a_l^T X_l                    # readout
    delta_l = Block_l(z_l)                   # non-residual transformer block
    H_l = Mixing_l()                         # stream mixing matrix
    X_{l+1} = H_l @ X_l + b_l \otimes delta_l  # mixing + injection
  z_L = (1/s) a_out^T X_L                  # final readout
  logits = LMHead(Norm(z_L))

BaselineTransformer (standard pre-norm):
  x_0 = token_embed + pos_embed
  For each layer l:
    x_{l+1} = Block_l(x_l)                  # full pre-norm block with residual
  logits = LMHead(Norm(x_L))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixing import create_mixing
from .headmix import HeadOutputMixing


class RMSNorm(nn.Module):
    """RMSNorm as used in modern LLMs."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout=0.0,
        use_flash=True,
        head_mixing_type=None,
        head_mixing_kwargs=None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.use_flash = use_flash

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.last_headmix_stats = None

        self.head_mixer = None
        if head_mixing_type not in (None, "none", "baseline"):
            self.head_mixer = HeadOutputMixing(
                num_heads,
                mixing_type=head_mixing_type,
                **(head_mixing_kwargs or {}),
            )

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            if not hasattr(self, '_causal_mask') or self._causal_mask.shape[-1] < T:
                self._causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(self._causal_mask[:T, :T], float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous()  # (B, T, heads, head_dim)
        if self.head_mixer is not None:
            out, self.last_headmix_stats = self.head_mixer(out, return_stats=True)
        else:
            self.last_headmix_stats = None

        out = out.reshape(B, T, C)
        out = self.o_proj(out)
        return out


class MLP(nn.Module):
    """Feed-forward network with GELU activation (GPT-2 style)."""

    def __init__(self, hidden_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.mlp_dim = int(hidden_dim * mlp_ratio)
        self.fc = nn.Linear(hidden_dim, self.mlp_dim, bias=False)
        self.proj = nn.Linear(self.mlp_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with residual connections."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_ratio=4,
        dropout=0.0,
        use_flash=True,
        head_mixing_type=None,
        head_mixing_kwargs=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(
            hidden_dim,
            num_heads,
            dropout,
            use_flash=use_flash,
            head_mixing_type=head_mixing_type,
            head_mixing_kwargs=head_mixing_kwargs,
        )
        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDeltaBlock(nn.Module):
    """Pre-norm Transformer block WITHOUT residual connections.

    Returns delta = Attn(Norm(x)) + MLP(Norm(x + Attn(Norm(x))))
    This is the "update" part that HC framework injects back into streams.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_ratio=4,
        dropout=0.0,
        use_flash=True,
        head_mixing_type=None,
        head_mixing_kwargs=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(
            hidden_dim,
            num_heads,
            dropout,
            use_flash=use_flash,
            head_mixing_type=head_mixing_type,
            head_mixing_kwargs=head_mixing_kwargs,
        )
        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        mlp_out = self.mlp(self.norm2(x + attn_out))
        return attn_out + mlp_out  # delta only, no residual


class BaselineTransformer(nn.Module):
    """Standard GPT-2 style Transformer baseline."""

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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_length, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                mlp_ratio,
                dropout,
                use_flash=use_flash,
                head_mixing_type=head_mixing_type,
                head_mixing_kwargs=head_mixing_kwargs,
            )
            for _ in range(num_layers)
        ])

        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.context_length

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.norm_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_headmix_diagnostics(self):
        stats = {}
        for layer, block in enumerate(self.blocks):
            if block.attn.last_headmix_stats is not None:
                for key, value in block.attn.last_headmix_stats.items():
                    stats[f"layer{layer}/headmix_{key}"] = value
        return stats


class HCTransformer(nn.Module):
    r"""Single-block multi-stream HC Transformer.

    Maintains s residual streams: X_l \in R^{s \times B \times T \times d}
    Each layer:
      z_l = (1/s) a_l^T X_l          # learned readout
      delta_l = Block_l(z_l)          # non-residual transformer block
      X_{l+1} = H_l @ X_l + b_l \otimes delta_l  # mixing + injection

    Args:
        vocab_size: vocabulary size
        d_model: dimension per stream
        num_layers: number of layers
        num_heads: number of attention heads
        n_streams: number of residual streams (s)
        context_length: maximum sequence length
        mixing_type: 'identity', 'unconstrained', 'orthogonal', 'isohc', 'mhc'
        mlp_ratio: MLP hidden dim ratio
        dropout: dropout rate
        lambda_a: readout perturbation scale (init)
        lambda_b: injection perturbation scale (init)
        ns_steps: Newton-Schulz steps for IsoHC
        sinkhorn_iters: Sinkhorn iterations for mHC
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 n_streams, context_length, mixing_type='isohc',
                 mlp_ratio=4, dropout=0.0,
                 lambda_a=0.01, lambda_b=0.01,
                 ns_steps=5, sinkhorn_iters=10,
                 use_flash=True,
                 head_mixing_type=None,
                 head_mixing_kwargs=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_streams = n_streams
        self.context_length = context_length
        self.mixing_type = mixing_type
        self.lambda_a_init = lambda_a
        self.lambda_b_init = lambda_b

        # Embeddings: shared token + pos, plus stream-specific embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_length, d_model)
        self.stream_embed = nn.Parameter(
            torch.randn(n_streams, 1, d_model) * 0.02
        )

        # Transformer blocks: operate on readout z (B, T, d)
        self.blocks = nn.ModuleList([
            TransformerDeltaBlock(
                d_model,
                num_heads,
                mlp_ratio,
                dropout,
                use_flash=use_flash,
                head_mixing_type=head_mixing_type,
                head_mixing_kwargs=head_mixing_kwargs,
            )
            for _ in range(num_layers)
        ])

        # Stream mixing: one per layer
        self.mixings = nn.ModuleList([
            create_mixing(n_streams, mixing_type, ns_steps=ns_steps,
                          sinkhorn_iters=sinkhorn_iters)
            for _ in range(num_layers)
        ])

        # Readout vectors: a_l = 1 + lambda_a * P_perp * u_l
        self.readout_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(n_streams))
            for _ in range(num_layers)
        ])
        self.readout_lambda = nn.Parameter(torch.tensor(lambda_a))

        # Injection vectors: b_l = 1 + lambda_b * P_perp * v_l
        self.injection_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(n_streams))
            for _ in range(num_layers)
        ])
        self.injection_lambda = nn.Parameter(torch.tensor(lambda_b))

        # Final readout
        self.readout_final = nn.Parameter(torch.zeros(n_streams))
        self.readout_final_lambda = nn.Parameter(torch.tensor(lambda_a))

        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_readout(self, weights, lambda_param):
        """a = 1 + lambda * P_perp * w where P_perp = I - (1/s) 1 1^T."""
        ones = torch.ones(self.n_streams, device=weights.device, dtype=weights.dtype)
        P_perp_w = weights - weights.mean()  # project away mean direction
        return ones + lambda_param * P_perp_w

    def _make_injection(self, weights, lambda_param):
        """b = 1 + lambda * P_perp * v, with constraint 1^T b = s (satisfied automatically)."""
        return self._make_readout(weights, lambda_param)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.context_length

        # Embeddings: (B, T, d) -> (s, B, T, d)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb  # (B, T, d)
        X = x.unsqueeze(0) + self.stream_embed.unsqueeze(1)  # (s, B, T, d)

        # Forward through layers
        for l, (block, mixing) in enumerate(zip(self.blocks, self.mixings)):
            # Readout
            a = self._make_readout(self.readout_weights[l], self.readout_lambda)
            z = torch.einsum('s,sbtd->btd', a, X) / self.n_streams  # (B, T, d)

            # Block (non-residual)
            delta = block(z)  # (B, T, d)

            # Mixing + injection
            b = self._make_injection(self.injection_weights[l], self.injection_lambda)
            H = mixing()  # (s, s)

            # X_new = H @ X + b \otimes delta
            mixed = torch.einsum('ij,jbtd->ibtd', H, X)  # (s, B, T, d)
            injected = b.view(self.n_streams, 1, 1, 1) * delta.unsqueeze(0)  # (s, B, T, d)
            X = mixed + injected

        # Final readout
        a_final = self._make_readout(self.readout_final, self.readout_final_lambda)
        z_final = torch.einsum('s,sbtd->btd', a_final, X) / self.n_streams

        z_final = self.norm_final(z_final)
        logits = self.lm_head(z_final)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def get_diagnostics(self):
        """Collect diagnostics from all layers."""
        diags = []
        for mixing in self.mixings:
            diags.append(mixing.get_diagnostics())
        return diags

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_headmix_diagnostics(self):
        stats = {}
        for layer, block in enumerate(self.blocks):
            if block.attn.last_headmix_stats is not None:
                for key, value in block.attn.last_headmix_stats.items():
                    stats[f"layer{layer}/headmix_{key}"] = value
        return stats

    def get_stream_states(self, input_ids):
        """Get stream states X_l at each layer for analysis.

        Returns list of tensors: [X_0, X_1, ..., X_L] where X_l is (s, B, T, d).
        """
        B, T = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        X = x.unsqueeze(0) + self.stream_embed.unsqueeze(1)

        states = [X.detach().clone()]

        for l, (block, mixing) in enumerate(zip(self.blocks, self.mixings)):
            a = self._make_readout(self.readout_weights[l], self.readout_lambda)
            z = torch.einsum('s,sbtd->btd', a, X) / self.n_streams
            delta = block(z)
            b = self._make_injection(self.injection_weights[l], self.injection_lambda)
            H = mixing()
            mixed = torch.einsum('ij,jbtd->ibtd', H, X)
            injected = b.view(self.n_streams, 1, 1, 1) * delta.unsqueeze(0)
            X = mixed + injected
            states.append(X.detach().clone())

        return states


class TwoBranchHCTransformer(nn.Module):
    """Two-branch HC Transformer with separate attention and MLP transport.

    This is the LM architecture used for the next IsoHC experiments.  Each GPT
    block contributes two residual updates, and each update has its own stream
    mixer:

      X_{l+1/2} = H_l^attn X_l + b_l^attn * Attn(LN(a_l^attn X_l / n))
      X_{l+1}   = H_l^mlp  X_{l+1/2} + b_l^mlp * MLP(LN(a_l^mlp X_{l+1/2}/n))

    The stream mean therefore carries the standard residual path, while the
    1_perp stream subspace remains available for IsoHC/mHC comparison.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        n_streams,
        context_length,
        mixing_type="isohc",
        mlp_ratio=4,
        dropout=0.0,
        lambda_a=0.01,
        lambda_b=0.01,
        ns_steps=5,
        svd_fallback=True,
        sinkhorn_iters=10,
        use_flash=True,
        head_mixing_type=None,
        head_mixing_kwargs=None,
        stream_noise_scale=1e-3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_streams = n_streams
        self.context_length = context_length
        self.mixing_type = mixing_type

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_length, d_model)

        stream_embed = torch.randn(n_streams, 1, d_model) * stream_noise_scale
        stream_embed = stream_embed - stream_embed.mean(dim=0, keepdim=True)
        self.stream_embed = nn.Parameter(stream_embed)

        self.attn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.attns = nn.ModuleList([
            CausalSelfAttention(
                d_model,
                num_heads,
                dropout,
                use_flash=use_flash,
                head_mixing_type=head_mixing_type,
                head_mixing_kwargs=head_mixing_kwargs,
            )
            for _ in range(num_layers)
        ])
        self.mlp_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([
            MLP(d_model, mlp_ratio, dropout) for _ in range(num_layers)
        ])

        mixing_kwargs = {
            "ns_steps": ns_steps,
            "svd_fallback": svd_fallback,
            "sinkhorn_iters": sinkhorn_iters,
        }
        self.attn_mixings = nn.ModuleList([
            create_mixing(n_streams, mixing_type, **mixing_kwargs)
            for _ in range(num_layers)
        ])
        self.mlp_mixings = nn.ModuleList([
            create_mixing(n_streams, mixing_type, **mixing_kwargs)
            for _ in range(num_layers)
        ])

        self.attn_readout_weights = nn.ParameterList([
            nn.Parameter(self._zero_mean_random(n_streams))
            for _ in range(num_layers)
        ])
        self.attn_injection_weights = nn.ParameterList([
            nn.Parameter(self._zero_mean_random(n_streams))
            for _ in range(num_layers)
        ])
        self.mlp_readout_weights = nn.ParameterList([
            nn.Parameter(self._zero_mean_random(n_streams))
            for _ in range(num_layers)
        ])
        self.mlp_injection_weights = nn.ParameterList([
            nn.Parameter(self._zero_mean_random(n_streams))
            for _ in range(num_layers)
        ])
        self.readout_lambda = nn.Parameter(torch.tensor(lambda_a))
        self.injection_lambda = nn.Parameter(torch.tensor(lambda_b))
        self.readout_final = nn.Parameter(self._zero_mean_random(n_streams))
        self.readout_final_lambda = nn.Parameter(torch.tensor(lambda_a))

        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _zero_mean_random(n_streams):
        x = torch.randn(n_streams)
        return x - x.mean()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_stream_vector(self, weights, lambda_param):
        ones = torch.ones(self.n_streams, device=weights.device, dtype=weights.dtype)
        return ones + lambda_param * (weights - weights.mean())

    def _transport(self, X, delta, mixing, injection_weights, mixing_override=None):
        H = mixing() if mixing_override is None else mixing_override
        b = self._make_stream_vector(injection_weights, self.injection_lambda)
        mixed = torch.einsum("ij,jbtd->ibtd", H, X)
        injected = b.view(self.n_streams, 1, 1, 1) * delta.unsqueeze(0)
        return mixed + injected

    def _apply_stream_intervention(self, X, state_index, stream_intervention):
        if stream_intervention is None:
            return X
        target = stream_intervention.get("state_index")
        if isinstance(target, (list, tuple, set)):
            active = state_index in target
        else:
            active = state_index == target
        if not active:
            return X

        mean = X.mean(dim=0, keepdim=True)
        mode = stream_intervention.get("mode", "mean_only")
        if mode == "mean_only":
            return mean.expand_as(X)
        if mode == "scale_perp":
            scale = float(stream_intervention.get("scale", 0.0))
            return mean + scale * (X - mean)
        raise ValueError(f"Unknown stream intervention mode: {mode}")

    def _resolve_mixing_override(self, branch, layer, mixing_overrides, device, dtype):
        if mixing_overrides is None:
            return None
        value = None
        if isinstance(mixing_overrides, str):
            value = mixing_overrides
        elif isinstance(mixing_overrides, dict):
            value = mixing_overrides.get((branch, layer))
            if value is None:
                value = mixing_overrides.get(f"{branch}:{layer}")
            if value is None:
                value = mixing_overrides.get(branch)
        else:
            value = mixing_overrides

        if value is None:
            return None
        if isinstance(value, str):
            if value != "identity":
                raise ValueError(f"Unknown mixing override: {value}")
            return torch.eye(self.n_streams, device=device, dtype=dtype)
        return value.to(device=device, dtype=dtype)

    @staticmethod
    def _record_stream_state(states, X, retain_stream_grads):
        if retain_stream_grads and X.requires_grad:
            X.retain_grad()
        states.append(X)

    def forward(
        self,
        input_ids,
        targets=None,
        stream_intervention=None,
        mixing_overrides=None,
        return_stream_states=False,
        retain_stream_grads=False,
    ):
        B, T = input_ids.shape
        assert T <= self.context_length

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=input_ids.device))
        X = (tok_emb + pos_emb).unsqueeze(0) + self.stream_embed.unsqueeze(1)
        X = self._apply_stream_intervention(X, 0, stream_intervention)
        states = []
        if return_stream_states:
            self._record_stream_state(states, X, retain_stream_grads)
        state_index = 0

        for l in range(self.num_layers):
            a_attn = self._make_stream_vector(
                self.attn_readout_weights[l], self.readout_lambda
            )
            z_attn = torch.einsum("s,sbtd->btd", a_attn, X) / self.n_streams
            delta_attn = self.attns[l](self.attn_norms[l](z_attn))
            attn_override = self._resolve_mixing_override(
                "attn", l, mixing_overrides, X.device, X.dtype
            )
            X = self._transport(
                X,
                delta_attn,
                self.attn_mixings[l],
                self.attn_injection_weights[l],
                mixing_override=attn_override,
            )
            state_index += 1
            X = self._apply_stream_intervention(X, state_index, stream_intervention)
            if return_stream_states:
                self._record_stream_state(states, X, retain_stream_grads)

            a_mlp = self._make_stream_vector(
                self.mlp_readout_weights[l], self.readout_lambda
            )
            z_mlp = torch.einsum("s,sbtd->btd", a_mlp, X) / self.n_streams
            delta_mlp = self.mlps[l](self.mlp_norms[l](z_mlp))
            mlp_override = self._resolve_mixing_override(
                "mlp", l, mixing_overrides, X.device, X.dtype
            )
            X = self._transport(
                X,
                delta_mlp,
                self.mlp_mixings[l],
                self.mlp_injection_weights[l],
                mixing_override=mlp_override,
            )
            state_index += 1
            X = self._apply_stream_intervention(X, state_index, stream_intervention)
            if return_stream_states:
                self._record_stream_state(states, X, retain_stream_grads)

        a_final = self._make_stream_vector(self.readout_final, self.readout_final_lambda)
        z_final = torch.einsum("s,sbtd->btd", a_final, X) / self.n_streams
        z_final = self.norm_final(z_final)
        logits = self.lm_head(z_final)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        if return_stream_states:
            return logits, loss, states
        return logits, loss

    def get_diagnostics(self):
        diags = []
        for branch, mixings in (("attn", self.attn_mixings), ("mlp", self.mlp_mixings)):
            for layer, mixing in enumerate(mixings):
                diag = mixing.get_diagnostics()
                diag["branch"] = branch
                diag["layer"] = layer
                diags.append(diag)
        return diags

    def get_headmix_diagnostics(self):
        stats = {}
        for layer, attn in enumerate(self.attns):
            if attn.last_headmix_stats is not None:
                for key, value in attn.last_headmix_stats.items():
                    stats[f"layer{layer}/headmix_{key}"] = value
        return stats

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def get_named_mixing_matrices(self):
        matrices = []
        index = 0
        for layer in range(self.num_layers):
            matrices.append({
                "index": index,
                "branch": "attn",
                "layer": layer,
                "H": self.attn_mixings[layer]().detach().clone(),
            })
            index += 1
            matrices.append({
                "index": index,
                "branch": "mlp",
                "layer": layer,
                "H": self.mlp_mixings[layer]().detach().clone(),
            })
            index += 1
        return matrices

    @torch.no_grad()
    def get_stream_states(self, input_ids):
        _, _, states = self.forward(input_ids, return_stream_states=True)
        return [state.detach().clone() for state in states]
