"""Quick verification script: test forward/backward without GPU.

Runs a single step of forward/backward for each model type
to catch shape mismatches and logic bugs.

Usage:
    python experiments/lm_verify.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from lm.models import BaselineTransformer, HCTransformer


def test_model(name, model, batch_size=2, seq_len=32, device='cpu'):
    """Test a single forward/backward step."""
    print(f"\nTesting {name}...")
    model = model.to(device)
    model.train()

    # Dummy data
    x = torch.randint(0, 100, (batch_size, seq_len), device=device)
    y = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Forward
    logits, loss = model(x, y)
    print(f"  logits shape: {logits.shape}")
    print(f"  loss: {loss.item():.4f}")
    assert logits.shape == (batch_size, seq_len, model.vocab_size), "Logits shape mismatch"
    assert torch.isfinite(loss), "Loss is not finite"

    # Backward
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  grad_norm: {grad_norm:.4f}")
    assert grad_norm > 0, "No gradients"

    # Diagnostics
    if hasattr(model, 'get_diagnostics'):
        diags = model.get_diagnostics()
        print(f"  diagnostics: {len(diags)} layers")
        if diags:
            print(f"    sample: {diags[0]}")

    # Stream states
    if hasattr(model, 'get_stream_states'):
        states = model.get_stream_states(x)
        print(f"  stream states: {len(states)} layers, shape {states[0].shape}")

    print(f"  PASS")
    return True


def main():
    print("=" * 60)
    print("LM Model Verification")
    print("=" * 60)

    vocab_size = 100
    d_model = 64
    num_layers = 2
    num_heads = 2
    n_streams = 4
    context_length = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Baseline
    baseline = BaselineTransformer(
        vocab_size, d_model, num_layers, num_heads, context_length
    )
    test_model("BaselineTransformer", baseline, device=device)

    # Unconstrained HC
    unconstrained = HCTransformer(
        vocab_size, d_model, num_layers, num_heads, n_streams, context_length,
        mixing_type='unconstrained'
    )
    test_model("HCTransformer (unconstrained)", unconstrained, device=device)

    # Orthogonal HC
    orthogonal = HCTransformer(
        vocab_size, d_model, num_layers, num_heads, n_streams, context_length,
        mixing_type='orthogonal'
    )
    test_model("HCTransformer (orthogonal)", orthogonal, device=device)

    # IsoHC
    isohc = HCTransformer(
        vocab_size, d_model, num_layers, num_heads, n_streams, context_length,
        mixing_type='isohc', ns_steps=3
    )
    test_model("HCTransformer (isohc)", isohc, device=device)

    # mHC
    mhc = HCTransformer(
        vocab_size, d_model, num_layers, num_heads, n_streams, context_length,
        mixing_type='mhc', sinkhorn_iters=5
    )
    test_model("HCTransformer (mhc)", mhc, device=device)

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
