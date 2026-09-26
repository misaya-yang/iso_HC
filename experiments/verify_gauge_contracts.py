"""Offline CPU checks of static HC gauge contracts; no training or downloads.

Run from the repository root: python3 experiments/verify_gauge_contracts.py
Numerical examples validate stated contracts, not novelty or trained-model gains.
"""

import argparse
import copy
import json
import math
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isohc.projection import construct_orthogonal_complement
from lm.models import TwoBranchHCTransformer
from lm.train import cosine_lr_schedule

DTYPE = torch.float64


def max_error(x, y):
    return float((x - y).abs().max())


def equality(name, x, y, tolerance=1e-10, **details):
    error = max_error(x, y)
    return {"name": name, "max_error": error, "tolerance": tolerance,
            "passed": error <= tolerance, **details}


def static_gauge(H):
    """X_k = G_k Z_k, with G_0=I and G_{k+1}=H_k G_k."""
    G = [torch.eye(H[0].shape[0], dtype=DTYPE)]
    for h in H:
        G.append(h @ G[-1])
    return G


def kernel(H, a, b):
    """K[i,j] = a_i^T H_{i-1} ... H_{j+1} b_j / n, for j<i."""
    n, depth = H[0].shape[0], len(H)
    K = torch.zeros(depth + 1, depth, dtype=DTYPE)
    for j in range(depth):
        state = b[j]
        for i in range(j + 1, depth + 1):
            K[i, j] = a[i] @ state / n
            if i < depth:
                state = H[i] @ state
    return K


def toy_forward(H, a, b, entry, inputs, weights):
    X = entry[:, None] * inputs[None, :]
    for k, h in enumerate(H):
        delta = torch.tanh((a[k] @ X / len(entry)) @ weights[k])
        X = h @ X + b[k][:, None] * delta[None, :]
    return a[-1] @ X / len(entry)


def algebra_checks():
    n, depth, width = 4, 6, 5
    eye = torch.eye(n, dtype=DTYPE)
    ones = torch.ones(n, dtype=DTYPE)
    H = [eye + 0.15 * torch.randn(n, n, dtype=DTYPE) for _ in range(depth)]
    a = [torch.randn(n, dtype=DTYPE) for _ in range(depth + 1)]
    b = [torch.randn(n, dtype=DTYPE) for _ in range(depth)]
    entry, inputs = torch.randn(n, dtype=DTYPE), torch.randn(width, dtype=DTYPE)
    weights = [torch.randn(width, width, dtype=DTYPE) / math.sqrt(width)
               for _ in range(depth)]
    G = static_gauge(H)
    ag = [g.T @ v for g, v in zip(G, a)]
    bg = [torch.linalg.solve(G[k + 1], b[k]) for k in range(depth)]
    identities = [eye] * depth
    checks = [equality(
        "invertible_static_transport_free_vectors",
        toy_forward(H, a, b, entry, inputs, weights),
        toy_forward(identities, ag, bg, entry, inputs, weights),
        scope="Nonlinear algebraic recurrence; unrestricted read/write/exit vectors.",
        max_cumulative_condition=float(max(torch.linalg.cond(g) for g in G))),
        equality("static_gauge_preserves_depth_kernel", kernel(H, a, b),
                 kernel(identities, ag, bg))]

    # An arbitrary G_0 requires transforming the input boundary as well.
    gauges = [eye + 0.1 * torch.randn(n, n, dtype=DTYPE) for _ in range(depth + 1)]
    Hg = [torch.linalg.solve(gauges[k + 1], H[k] @ gauges[k]) for k in range(depth)]
    ag = [g.T @ v for g, v in zip(gauges, a)]
    bg = [torch.linalg.solve(gauges[k + 1], b[k]) for k in range(depth)]
    checks.append(equality("arbitrary_gauge_transforms_input_and_exit_boundaries",
        toy_forward(H, a, b, entry, inputs, weights),
        toy_forward(Hg, ag, bg, torch.linalg.solve(gauges[0], entry), inputs, weights)))
    checks.append(equality("arbitrary_gauge_preserves_depth_kernel", kernel(H, a, b),
                           kernel(Hg, ag, bg)))

    # A fixed shared input plus fixed mean exit is not closed under full O(n).
    q = torch.diag(torch.tensor([1.0, -1.0], dtype=DTYPE))
    e = torch.ones(2, dtype=DTYPE)
    original, untransformed_exit = e @ q @ e / 2, e @ e / 2
    checks.append(equality("orthogonal_exit_boundary_transform", original,
                           (q.T @ e) @ e / 2,
                           unchanged_exit_gap=float(abs(original - untransformed_exit)),
                           transformed_exit_sum=float((q.T @ e).sum()),
                           required_sum_in_repo_class=2,
                           scope="Exit transform leaves the fixed-sum parameter class."))
    checks[-1]["passed"] &= abs(float(original - untransformed_exit)) > 0.5

    # Singular H cannot be converted to I by invertible state coordinates.
    singular = torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE))
    left = torch.tensor([[1., .2], [.1, 1.]], dtype=DTYPE)
    right = torch.tensor([[1., -.1], [.3, 1.]], dtype=DTYPE)
    transformed = torch.linalg.solve(left, singular @ right)
    rank = int(torch.linalg.matrix_rank(transformed))
    checks.append({"name": "singular_transport_rank_obstruction", "passed": rank == 1,
                   "transport_rank": rank, "identity_rank": 2,
                   "max_error": float(torch.linalg.svdvals(transformed)[-1]),
                   "scope": "State-coordinate obstruction; not a claim that every singular model defines a new function class."})

    U = construct_orthogonal_complement(n, dtype=DTYPE)
    P0 = torch.outer(ones, ones) / n
    mean_H = [P0 + U @ torch.linalg.qr(torch.randn(n - 1, n - 1, dtype=DTYPE))[0] @ U.T
              for _ in range(depth)]
    u = [U @ torch.randn(n - 1, dtype=DTYPE) for _ in range(depth + 1)]
    v = [U @ torch.randn(n - 1, dtype=DTYPE) for _ in range(depth)]
    base = kernel(mean_H, [ones] * (depth + 1), [ones] * depth)
    corrections = []
    for scale in (0.01, 0.1, 0.5):
        corrections.append((kernel(mean_H, [ones + scale * x for x in u],
                                   [ones + scale * x for x in v]) - base) / scale ** 2)
    checks.append(equality("mean_preserving_kernel_lambda_squared", corrections[0], corrections[2],
                           tolerance=1e-10, lambda_values=[0.01, 0.1, 0.5],
                           intermediate_scale_max_error=max_error(corrections[0], corrections[1]),
                           scope="Fixed zero-mean vectors and exact mean-preserving H; says nothing by itself about loss changes or initial stream offsets."))

    # No mean-complement exchange is needed for more than one depth-state mode.
    u2 = torch.tensor([1.0, -1.0], dtype=DTYPE) / math.sqrt(2)
    rows = [e, e + math.sqrt(2) * u2]
    cols = [e, e + math.sqrt(2) * u2]
    cross_cut = torch.stack([torch.stack([r @ c / 2 for c in cols]) for r in rows])
    checks.append(equality("mean_preserving_identity_has_rank_two_cross_cut_kernel",
        cross_cut, torch.tensor([[1., 1.], [1., 2.]], dtype=DTYPE),
        rank=int(torch.linalg.matrix_rank(cross_cut)),
        scope="Two past writes and two future reads, H=I; fixed-sum read/write vectors."))
    checks[-1]["passed"] &= int(torch.linalg.matrix_rank(cross_cut)) == 2

    # Complement can leave and return; projection after every step destroys it.
    Q = torch.stack([e / math.sqrt(2), u2], dim=1)
    exchange = Q @ torch.tensor([[0., -1.], [1., 0.]], dtype=DTYPE) @ Q.T
    full = u2 @ exchange @ exchange @ u2
    projected = (u2 @ exchange @ u2) ** 2
    checks.append(equality("projection_between_steps_loses_exchange_return_path",
        full, torch.tensor(-1., dtype=DTYPE),
        product_of_projected_steps=float(projected),
        projection_after_full_product=float(full),
        discrepancy=float(abs(full - projected))))
    checks[-1]["passed"] &= abs(float(full - projected)) > 0.9
    return checks


@torch.no_grad()
def model_check(exact_projection):
    model = TwoBranchHCTransformer(vocab_size=23, d_model=16, num_layers=3,
        num_heads=2, n_streams=4, context_length=7, mixing_type="isohc",
        mlp_ratio=2, lambda_a=0.5, lambda_b=0.5, use_flash=False).double().eval()
    branches = [(kind, layer) for layer in range(model.num_layers) for kind in ("attn", "mlp")]
    for kind, layer in branches:
        mixing = getattr(model, kind + "_mixings")[layer]
        mixing.H_raw.copy_(torch.eye(4, dtype=DTYPE) + 0.3 * torch.randn(4, 4, dtype=DTYPE))
        if exact_projection:
            # Explicitly distinguish theorem precision from the default cached fp32 basis.
            mixing.U = construct_orthogonal_complement(4, dtype=DTYPE)
            mixing.use_svd = True
    H = [getattr(model, kind + "_mixings")[layer]() for kind, layer in branches]
    G = static_gauge(H)
    gauged = copy.deepcopy(model)
    closure_errors, norm_errors = [], []
    for k, (kind, layer) in enumerate(branches):
        for role, scale, transform in (
            ("readout", model.readout_lambda, lambda x: G[k].T @ x),
            ("injection", model.injection_lambda, lambda x: torch.linalg.solve(G[k + 1], x)),
        ):
            source = getattr(model, kind + "_" + role + "_weights")[layer]
            target = getattr(gauged, kind + "_" + role + "_weights")[layer]
            vector = model._make_stream_vector(source, scale)
            transformed = transform(vector)
            closure_errors.append(abs(float(transformed.sum() - 4)))
            norm_errors.append(abs(float(transformed.square().sum() - vector.square().sum())))
            target.copy_((transformed - 1) / scale)
    final = model._make_stream_vector(model.readout_final, model.readout_final_lambda)
    gauged.readout_final.copy_((G[-1].T @ final - 1) / model.readout_final_lambda)
    ids = torch.randint(0, 23, (2, 7))
    original = model(ids)[0]
    result = gauged(ids, mixing_overrides="identity")[0]
    raw_identity = model(ids, mixing_overrides="identity")[0]
    return equality("repo_two_branch_logits_" + ("exact_projection" if exact_projection else "default_projection"),
        original, result, tolerance=1e-10 if exact_projection else 2e-6,
        ungauged_identity_max_error=max_error(original, raw_identity),
        max_transport_orthogonality_error=max(float(torch.linalg.norm(h.T @ h - torch.eye(4))) for h in H),
        max_transport_mean_error=max(float(torch.linalg.norm(h @ torch.ones(4, dtype=DTYPE) - 1)) for h in H),
        max_read_write_fixed_sum_error=max(closure_errors),
        max_read_write_squared_norm_change=max(norm_errors),
        scope="Actual TwoBranchHCTransformer; float64 forward, nonzero read/write coupling; no training.")


def weight_decay_replay():
    """Zero-gradient symmetric-init counterfactual, not recovered parameter history."""
    paths = [
        "docs/0605_alldoc/0525_results_raw/0525_fe_fair_deep48_p33013_20m/fe-deep-48l-512_mhc_seed0/run_summary.json",
        "docs/0605_alldoc/0524_results_raw/0524_deep_stress_512/deep-stress-512_mhc_seed0/run_summary.json",
    ]
    results = []
    for relative in paths:
        summary = json.loads((ROOT / relative).read_text())
        cfg = summary["config"]
        tokens_per_step = cfg["batch_size"] * cfg["context_length"] * cfg.get("grad_accum_steps", 1)
        total_steps = math.ceil(cfg["total_tokens"] / tokens_per_step)
        warmup = cfg["warmup_tokens"] // tokens_per_step
        shrink = 1.0
        # train_epoch restarts its local step counter every epoch.
        for epoch in summary["train_metrics"]:
            for step in range(epoch["steps"]):
                lr = cosine_lr_schedule(step, warmup, total_steps, cfg["max_lr"], cfg["min_lr"])
                shrink *= 1 - lr * cfg["weight_decay"]
        diag_bias, n = 4.0, cfg["n_streams"]
        exponent = math.exp(diag_bias * shrink)
        predicted = (exponent - 1) / (exponent + n - 1)
        measured = summary["posthoc"]["h_diagnostics"]["sv_mean_1perp_mean"]
        results.append({"source": relative, "layers": cfg["num_layers"],
            "optimizer_steps": sum(x["steps"] for x in summary["train_metrics"]),
            "init_diag_bias": diag_bias, "logit_shrink_factor": shrink,
            "predicted_single_step_sigma": predicted, "measured_single_step_sigma": measured,
            "absolute_difference": abs(predicted - measured),
            "relative_difference": abs(predicted - measured) / measured,
            "predicted_composite_sigma": predicted ** (2 * cfg["num_layers"]),
            "measured_row_sum_error_mean": summary["posthoc"]["h_diagnostics"]["row_sum_err_mean"],
            "scope": "Symmetric 4I initialization; omits random init noise and task gradients. Close agreement does not prove gradients vanish. Composite is prediction only, not matched to another run."})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output", type=Path, default=ROOT / "results/theory_audit_20260925/gauge_contracts.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checks = algebra_checks() + [model_check(False), model_check(True)]
    report = {"seed": args.seed, "precision": str(DTYPE), "device": "cpu",
              "python": platform.python_version(), "torch": torch.__version__,
              "training_performed": False, "checks": checks,
              "weight_decay_counterfactuals": weight_decay_replay(),
              "passed": all(check["passed"] for check in checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['max_error']:.3e}")
    for replay in report["weight_decay_counterfactuals"]:
        print(f"WD-only {replay['layers']}L: predicted {replay['predicted_single_step_sigma']:.8f}, measured {replay['measured_single_step_sigma']:.8f}")
    print(f"Receipt: {args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
