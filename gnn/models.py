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
    IsoStream-GCN: multiple streams with isometric mixing.

    Maintains s streams: X_l ∈ R^{s × N × d}
    Each layer: Y_{l,s} = σ(S X_{l,s} W_l)
    Then: X_{l+1} = H_l X_l + β Y_l
    where H_l ∈ M_{1_s} (orthogonal, preserves 1 vector)

    Final readout: Z = mean over streams
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 n_streams=4, beta=0.5, dropout=0.5, ns_steps=5):
        super().__init__()
        self.num_layers = num_layers
        self.n_streams = n_streams
        self.beta = beta
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # Input projection: shared across streams
        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)

        # GCN layers: one per layer, shared across streams
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Iso mixing matrices: one per layer
        # H ∈ R^{s×s}, H^T H = I, H @ 1 = 1
        # We use the same Iso projection as IsoHC
        self.iso_mixings = nn.ModuleList([
            IsoStreamMixing(n_streams, ns_steps=ns_steps)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x, S):
        """
        x: (N, d_in)
        S: (N, N) normalized adjacency
        Returns: (N, out_dim) logits
        """
        N = x.shape[0]

        # Project input and expand to streams
        x = self.input_proj(x)  # (N, hidden_dim)
        x = F.relu(x)

        # Expand to streams: (s, N, hidden_dim)
        x_streams = x.unsqueeze(0).repeat(self.n_streams, 1, 1)
        x_streams = F.dropout(x_streams, p=self.dropout, training=self.training)

        for gcn_layer, iso_mix in zip(self.gcn_layers, self.iso_mixings):
            # Apply GCN to each stream
            y_streams = []
            for s in range(self.n_streams):
                y = gcn_layer(x_streams[s], S)  # (N, hidden_dim)
                y = F.relu(y)
                y_streams.append(y)
            y_streams = torch.stack(y_streams, dim=0)  # (s, N, hidden_dim)

            # Iso mixing: X_new = H @ X + β * Y
            H = iso_mix()  # (s, s)
            # H @ X: for each node and feature dim, mix across streams
            mixed = torch.einsum('ij,jnd->ind', H, x_streams)  # (s, N, hidden_dim)
            x_streams = mixed + self.beta * y_streams

            x_streams = F.dropout(x_streams, p=self.dropout, training=self.training)

        # Stream mean readout
        x = x_streams.mean(dim=0)  # (N, hidden_dim)
        return self.output_proj(x)

    def get_hidden_states(self, x, S):
        """Get hidden states (stream-averaged) at each layer."""
        states = []
        N = x.shape[0]

        x = self.input_proj(x)
        x = F.relu(x)
        x_streams = x.unsqueeze(0).repeat(self.n_streams, 1, 1)
        states.append(x_streams.mean(dim=0).detach().clone())

        for gcn_layer, iso_mix in zip(self.gcn_layers, self.iso_mixings):
            y_streams = []
            for s in range(self.n_streams):
                y = gcn_layer(x_streams[s], S)
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
