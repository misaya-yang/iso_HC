"""Tiny Transformer models with IsoHC and baseline residual connections."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    RMSNorm, CausalSelfAttention, MLP,
    IsoHCResidualMixing, UnconstrainedHCResidualMixing, MHCLiteResidualMixing
)


class IsoHCTransformerBlock(nn.Module):
    """
    Transformer block with IsoHC residual mixing.

    The IsoHC mixing replaces the identity in the attention residual:
      y = H * x + Attention(Norm(x))
      out = y + MLP(Norm(y))

    where H is projected to M_iso at each forward pass.
    """

    def __init__(self, hidden_dim, num_heads, n_streams, mlp_ratio=4, dropout=0.0, ns_steps=5):
        super().__init__()
        assert hidden_dim % n_streams == 0
        self.hidden_dim = hidden_dim
        self.n_streams = n_streams
        self.d_per_stream = hidden_dim // n_streams

        self.norm1 = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.hc_mixing = IsoHCResidualMixing(n_streams, ns_steps=ns_steps, init_identity=True)

        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        Returns: (batch, seq_len, hidden_dim)
        """
        B, T, C = x.shape

        # Attention path with IsoHC residual mixing
        x_streams = x.reshape(B, T, self.n_streams, self.d_per_stream)
        mixed = self.hc_mixing(x_streams)
        attn_out = self.attn(self.norm1(x))
        attn_streams = attn_out.reshape(B, T, self.n_streams, self.d_per_stream)
        y = (mixed + attn_streams).reshape(B, T, C)

        # MLP path (standard residual)
        out = y + self.mlp(self.norm2(y))

        return out

    def get_hc_diagnostics(self):
        """Return IsoHC projection diagnostics."""
        return self.hc_mixing.get_diagnostics()


class BaselineTransformerBlock(nn.Module):
    """Standard pre-norm Transformer block."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UnconstrainedHCTransformerBlock(nn.Module):
    """Transformer block with unconstrained HC residual mixing."""

    def __init__(self, hidden_dim, num_heads, n_streams, mlp_ratio=4, dropout=0.0):
        super().__init__()
        assert hidden_dim % n_streams == 0
        self.hidden_dim = hidden_dim
        self.n_streams = n_streams
        self.d_per_stream = hidden_dim // n_streams

        self.norm1 = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.hc_mixing = UnconstrainedHCResidualMixing(n_streams, init_scale=0.01)

        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(self, x):
        B, T, C = x.shape
        x_streams = x.reshape(B, T, self.n_streams, self.d_per_stream)
        mixed = self.hc_mixing(x_streams)
        attn_out = self.attn(self.norm1(x))
        attn_streams = attn_out.reshape(B, T, self.n_streams, self.d_per_stream)
        y = (mixed + attn_streams).reshape(B, T, C)
        out = y + self.mlp(self.norm2(y))
        return out


class IsoHCTransformer(nn.Module):
    """Tiny Transformer with IsoHC residual mixing."""

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 n_streams, context_length, mlp_ratio=4, dropout=0.0,
                 ns_steps=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_streams = n_streams
        self.context_length = context_length
        self.ns_steps = ns_steps

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(context_length, hidden_dim)

        self.blocks = nn.ModuleList([
            IsoHCTransformerBlock(
                hidden_dim, num_heads, n_streams,
                mlp_ratio=mlp_ratio, dropout=dropout, ns_steps=ns_steps
            )
            for _ in range(num_layers)
        ])

        self.norm_final = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

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
        """
        input_ids: (batch, seq_len) int tensor
        targets: optional (batch, seq_len) int tensor for loss computation

        Returns:
          logits: (batch, seq_len, vocab_size)
          loss: scalar if targets provided, else None
        """
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

    def get_diagnostics(self):
        """Collect IsoHC projection diagnostics from all layers."""
        diags = []
        for block in self.blocks:
            if hasattr(block, 'get_hc_diagnostics'):
                diags.append(block.get_hc_diagnostics())
        return diags

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class UnconstrainedHCTransformer(nn.Module):
    """Tiny Transformer with unconstrained HC residual mixing."""

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 n_streams, context_length, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_streams = n_streams
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(context_length, hidden_dim)

        self.blocks = nn.ModuleList([
            UnconstrainedHCTransformerBlock(
                hidden_dim, num_heads, n_streams,
                mlp_ratio=mlp_ratio, dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm_final = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
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


class BaselineTransformer(nn.Module):
    """Standard pre-norm Transformer (baseline)."""

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 context_length, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(context_length, hidden_dim)

        self.blocks = nn.ModuleList([
            BaselineTransformerBlock(
                hidden_dim, num_heads,
                mlp_ratio=mlp_ratio, dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm_final = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
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
