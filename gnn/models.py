"""GNN models: GCN, ResGCN, IsoStream-GCN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import IsoNodeProjection


class GCNLayer(nn.Module):
    """Single GCN layer: Y = σ(S X W)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, S):
        """
        x: (N, d_in)
        S: (N, N) normalized adjacency
        Returns: (N, d_out)
        """
        x = S @ x  # graph propagation
        x = self.linear(x)  # feature transform
        return x


class GCN(nn.Module):
    """Standard multi-layer GCN."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x, S):
        """
        x: (N, d_in) node features
        S: (N, N) normalized adjacency
        Returns: (N, out_dim) logits
        """
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, S)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output_proj(x)

    def get_hidden_states(self, x, S):
        """Get hidden states at each layer for oversmoothing analysis."""
        states = []
        x = self.input_proj(x)
        x = F.relu(x)
        states.append(x.detach().clone())

        for layer in self.layers:
            x = layer(x, S)
            x = F.relu(x)
            states.append(x.detach().clone())

        return states


class ResGCN(nn.Module):
    """Deep residual GCN: X_{l+1} = X_l + β * σ(S X_l W_l)."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, beta=0.5, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x, S):
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x_new = layer(x, S)
            x_new = F.relu(x_new)
            x = x + self.beta * x_new
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output_proj(x)

    def get_hidden_states(self, x, S):
        """Get hidden states at each layer."""
        states = []
        x = self.input_proj(x)
        x = F.relu(x)
        states.append(x.detach().clone())

        for layer in self.layers:
            x_new = layer(x, S)
            x_new = F.relu(x_new)
            x = x + self.beta * x_new
            states.append(x.detach().clone())

        return states


class IsoStreamGCN(nn.Module):
    """
    IsoStream-GCN v2: multiple streams with isometric mixing.

    Fixes from v1:
      1. Deterministic stream diversity via learned stream_embed
      2. Dropout only on message branch (not residual state)
      3. Concat readout (not mean) so stream-zero subspace survives

    Maintains s streams: X_l ∈ R^{s × N × d}
    Each layer:
      Y_{l,s} = σ(S · Dropout(X_{l,s}) · W_l)   ← message branch
      X_{l+1} = H_l @ X_l + β · Y_l              ← update
    where H_l ∈ M_{1_s} (orthogonal, preserves 1 vector)

    Final readout: Z = concat over streams → Linear
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 n_streams=4, beta=0.5, dropout=0.5, ns_steps=5):
        super().__init__()
        self.num_layers = num_layers
        self.n_streams = n_streams
        self.beta = beta
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # Shared input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)

        # Stream-specific embedding for deterministic diversity.
        # Each stream starts from a different direction in hidden space.
        self.stream_embed = nn.Parameter(
            torch.randn(n_streams, 1, hidden_dim) * 0.02
        )

        # GCN layers: one per layer, shared across streams
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Iso mixing matrices: one per layer
        # H ∈ R^{s×s}, H^T H = I, H @ 1 = 1
        self.iso_mixings = nn.ModuleList([
            IsoStreamMixing(n_streams, ns_steps=ns_steps)
            for _ in range(num_layers)
        ])

        # Output: concat streams → linear projection
        self.output_proj = nn.Linear(n_streams * hidden_dim, out_dim, bias=False)

    def forward(self, x, S):
        """
        x: (N, d_in)
        S: (N, N) normalized adjacency
        Returns: (N, out_dim) logits
        """
        N = x.shape[0]

        # Project input
        x = self.input_proj(x)  # (N, hidden_dim)
        x = F.relu(x)

        # Expand to streams with deterministic diversity
        # Each stream = base + stream-specific bias
        x_streams = x.unsqueeze(0) + self.stream_embed  # (s, N, hidden_dim)

        for gcn_layer, iso_mix in zip(self.gcn_layers, self.iso_mixings):
            # Apply GCN to each stream (dropout only on message input)
            y_streams = []
            for s in range(self.n_streams):
                x_s = x_streams[s]  # (N, hidden_dim)
                # Dropout only on message branch input, not residual state
                x_s_drop = F.dropout(x_s, p=self.dropout, training=self.training)
                y = gcn_layer(x_s_drop, S)  # (N, hidden_dim)
                y = F.relu(y)
                y_streams.append(y)
            y_streams = torch.stack(y_streams, dim=0)  # (s, N, hidden_dim)

            # Iso mixing: X_new = H @ X + β * Y
            H = iso_mix()  # (s, s)
            mixed = torch.einsum('ij,jnd->ind', H, x_streams)  # (s, N, hidden_dim)
            x_streams = mixed + self.beta * y_streams

        # Concat readout: preserve all stream information
        x = x_streams.permute(1, 0, 2).reshape(N, self.n_streams * self.hidden_dim)
        return self.output_proj(x)

    def get_hidden_states(self, x, S):
        """Get hidden states (concat of all streams) at each layer."""
        states = []
        N = x.shape[0]

        x = self.input_proj(x)
        x = F.relu(x)
        x_streams = x.unsqueeze(0) + self.stream_embed
        # Store mean for consistency with other models' state shape
        states.append(x_streams.mean(dim=0).detach().clone())

        for gcn_layer, iso_mix in zip(self.gcn_layers, self.iso_mixings):
            y_streams = []
            for s in range(self.n_streams):
                x_s = x_streams[s]
                x_s_drop = F.dropout(x_s, p=self.dropout, training=self.training)
                y = gcn_layer(x_s_drop, S)
                y = F.relu(y)
                y_streams.append(y)
            y_streams = torch.stack(y_streams, dim=0)

            H = iso_mix()
            mixed = torch.einsum('ij,jnd->ind', H, x_streams)
            x_streams = mixed + self.beta * y_streams
            states.append(x_streams.mean(dim=0).detach().clone())

        return states

    def get_iso_diagnostics(self):
        """Get IsoStream projection diagnostics."""
        diags = []
        for mix in self.iso_mixings:
            diags.append(mix.get_diagnostics())
        return diags


class IsoStreamMixing(nn.Module):
    """
    Learnable isometric mixing for stream dimension.

    Projects H_raw ∈ R^{s×s} to M_1 = {H : H^T H = I, H @ 1 = 1}.
    """

    def __init__(self, n_streams, ns_steps=5):
        super().__init__()
        self.n_streams = n_streams
        self.ns_steps = ns_steps

        # Raw parameter: near identity
        H_init = torch.eye(n_streams, dtype=torch.float32)
        H_init += torch.randn(n_streams, n_streams, dtype=torch.float32) * 0.01
        self.H_raw = nn.Parameter(H_init)

        # Precompute U for 1-vector (same as IsoHC)
        from isohc.projection import construct_orthogonal_complement
        U = construct_orthogonal_complement(n_streams, device='cpu', dtype=torch.float32)
        self.register_buffer('U', U)

    def forward(self):
        """Return projected H."""
        from isohc.projection import iso_ns_project
        return iso_ns_project(self.H_raw, U=self.U, steps=self.ns_steps, use_svd=False)

    def get_diagnostics(self):
        """Return projection diagnostics."""
        H = self.forward()
        n = self.n_streams
        ones = torch.ones(n, 1, device=H.device, dtype=torch.float32)
        I = torch.eye(n, device=H.device, dtype=torch.float32)
        orth_error = torch.norm(H.T @ H - I, p='fro').item()
        fix_error = torch.norm(H @ ones - ones, p=2).item()

        # Energy preservation
        X_test = torch.randn(n, 128, device=H.device, dtype=torch.float32)
        energy_ratio = (torch.norm(H @ X_test, p='fro') / torch.norm(X_test, p='fro')).item()

        return {
            'orth_error': orth_error,
            'fix_error': fix_error,
            'energy_ratio': energy_ratio,
        }


class IsoResGCN(nn.Module):
    """
    IsoRes-GCN: hybrid diffusion + isometry architecture.

    Combines three paths at each layer:
      X_{l+1} = Norm(X_l + β·σ(SX_lW_l) + γ·Q_lX_l)

    where:
      - X_l               : identity memory (residual)
      - β·σ(SX_lW_l)      : graph diffusion / aggregation (GCN path)
      - γ·Q_lX_l          : isometric anti-collapse path (Iso path)

    Q_l satisfies Q_l^T Q_l = I and Q_l v = v, where v = sqrt(d_tilde).
    γ should be small (e.g., 0.05–0.2); the Iso path is a stabiliser,
    not the main propagation.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 n_nodes, v,
                 beta=0.5, gamma=0.1,
                 dropout=0.5, use_norm=True,
                 ns_steps=5, use_svd=False):
        super().__init__()
        self.num_layers = num_layers
        self.beta = beta
        self.gamma = gamma
        self.dropout = dropout
        self.use_norm = use_norm

        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)

        # GCN layers: W_l for feature transform
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers)
        ])

        # Isometric operators: one per layer, Q_l preserves v
        self.iso_operators = nn.ModuleList([
            IsoNodeProjection(n_nodes, v, ns_steps=ns_steps,
                              init_identity=True, use_svd=use_svd)
            for _ in range(num_layers)
        ])

        if use_norm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])

        self.output_proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x, S):
        """
        x: (N, d_in)  node features
        S: (N, N)     normalized adjacency
        Returns: (N, out_dim) logits
        """
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, (gcn_layer, iso_op) in enumerate(zip(self.gcn_layers, self.iso_operators)):
            # GCN path: β·σ(S X W)
            gcn_out = gcn_layer(x)      # (N, hidden_dim)  ← XW
            gcn_out = S @ gcn_out       # (N, hidden_dim)  ← SXW
            gcn_out = F.relu(gcn_out)

            # Iso path: γ·Q X
            Q = iso_op()                # (N, N)
            iso_out = Q @ x             # (N, hidden_dim)

            # Hybrid update
            x = x + self.beta * gcn_out + self.gamma * iso_out

            if self.use_norm:
                x = self.norms[i](x)

            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output_proj(x)

    def get_hidden_states(self, x, S):
        """Get hidden states at each layer for oversmoothing analysis."""
        states = []
        x = self.input_proj(x)
        x = F.relu(x)
        states.append(x.detach().clone())

        for i, (gcn_layer, iso_op) in enumerate(zip(self.gcn_layers, self.iso_operators)):
            gcn_out = gcn_layer(x)
            gcn_out = S @ gcn_out
            gcn_out = F.relu(gcn_out)

            Q = iso_op()
            iso_out = Q @ x

            x = x + self.beta * gcn_out + self.gamma * iso_out

            if self.use_norm:
                x = self.norms[i](x)

            states.append(x.detach().clone())

        return states

    def get_iso_diagnostics(self):
        """Get isometric operator diagnostics."""
        diags = []
        for op in self.iso_operators:
            diags.append(op.get_diagnostics())
        return diags
