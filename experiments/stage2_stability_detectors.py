"""Stability-boundary detectors for IsoHC-style residual mixing.

This script focuses on diagnostics that matter for the paper:

1. Gradient norm versus depth.
2. Energy retention on the mean-zero subspace.
3. Stream diversity / collapse proxies.
4. Stability boundary under depth, dtype, and initialization scale.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isohc import iso_ns_project


METHODS = ("isohc", "mhc-lite", "unconstrained", "gcn-diffusion")
DTYPES = ("fp64", "fp32", "bf16", "bf16_fp32_mix")


def summarize(xs):
    return {
        "mean": float(sum(xs) / len(xs)),
        "std": float(torch.tensor(xs, dtype=torch.float64).std(unbiased=False).item()) if xs else 0.0,
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def complete_graph_diffusion(n, alpha, device, dtype):
    ones = torch.ones(n, n, device=device, dtype=dtype) / n
    eye = torch.eye(n, device=device, dtype=dtype)
    return (1.0 - alpha) * eye + alpha * ones


def mhc_lite_matrix(n, device, dtype, scale):
    m = torch.eye(n, device=device, dtype=dtype) + torch.rand(n, n, device=device, dtype=dtype) * scale
    m = m.abs() + 1e-8
    for _ in range(30):
        m = m / m.sum(dim=1, keepdim=True)
        m = m / m.sum(dim=0, keepdim=True)
    return m


def make_h(n, method, device, dtype, ns_steps, scale, alpha):
    if method == "isohc":
        raw_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        raw = torch.eye(n, device=device, dtype=raw_dtype) + torch.randn(n, n, device=device, dtype=raw_dtype) * scale
        return iso_ns_project(raw, steps=ns_steps).to(dtype)
    if method == "mhc-lite":
        return mhc_lite_matrix(n, device, dtype, scale)
    if method == "unconstrained":
        return torch.eye(n, device=device, dtype=dtype) + torch.randn(n, n, device=device, dtype=dtype) * scale
    if method == "gcn-diffusion":
        return complete_graph_diffusion(n, alpha, device, dtype)
    raise ValueError(method)


def dtype_config(name):
    if name == "fp64":
        return torch.float64, torch.float64
    if name == "fp32":
        return torch.float32, torch.float32
    if name == "bf16":
        return torch.bfloat16, torch.bfloat16
    if name == "bf16_fp32_mix":
        return torch.bfloat16, torch.float32
    raise ValueError(name)


def mean_zero(x):
    return x - x.mean(dim=0, keepdim=True)


def offdiag_cosine_abs_mean(x):
    # x: (streams, features). High value means streams collapsed / aligned.
    x = x.float()
    x = x - x.mean(dim=1, keepdim=True)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    c = x @ x.T
    n = c.shape[0]
    mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
    return c[mask].abs().mean().item()


def run_one(method, n, depth, feature_dim, trials, dtype_name, device, ns_steps, scale, alpha):
    state_dtype, matmul_dtype = dtype_config(dtype_name)
    energy_raw, energy_zero, grad_ratio, mean_error, diversity = [], [], [], [], []
    sigma_min, sigma_max, left_fix, right_fix = [], [], [], []
    ones = torch.ones(n, 1, device=device, dtype=torch.float32)

    for _ in range(trials):
        hs = [make_h(n, method, device, torch.float32 if matmul_dtype != torch.float64 else torch.float64, ns_steps, scale, alpha) for _ in range(depth)]

        for h in hs:
            h32 = h.float()
            sv = torch.linalg.svdvals(h32)
            sigma_min.append(sv.min().item())
            sigma_max.append(sv.max().item())
            left_fix.append(torch.linalg.norm(h32.T @ ones - ones).item())
            right_fix.append(torch.linalg.norm(h32 @ ones - ones).item())

        x0 = torch.randn(n, feature_dim, device=device, dtype=state_dtype)
        x0z = mean_zero(x0.float()).to(state_dtype)

        x = x0
        xz = x0z
        for h in hs:
            h_run = h.to(matmul_dtype)
            if dtype_name == "bf16_fp32_mix":
                x = h_run.float() @ x.float()
                xz = h_run.float() @ xz.float()
            else:
                x = h_run @ x.to(matmul_dtype)
                xz = h_run @ xz.to(matmul_dtype)

        energy_raw.append((torch.linalg.norm(x.float()) / torch.linalg.norm(x0.float())).item())
        energy_zero.append((torch.linalg.norm(xz.float()) / torch.linalg.norm(x0z.float())).item())
        mean_error.append(torch.linalg.norm((ones.T @ x.float() - ones.T @ x0.float()) / n).item())
        diversity.append(offdiag_cosine_abs_mean(x.float()))

        g_last = torch.randn(n, feature_dim, device=device, dtype=state_dtype)
        g = g_last
        for h in reversed(hs):
            h_run = h.to(matmul_dtype)
            if dtype_name == "bf16_fp32_mix":
                g = h_run.float().T @ g.float()
            else:
                g = h_run.T @ g.to(matmul_dtype)
        grad_ratio.append((torch.linalg.norm(g.float()) / torch.linalg.norm(g_last.float())).item())

    return {
        "energy_raw": summarize(energy_raw),
        "energy_mean_zero": summarize(energy_zero),
        "grad_ratio": summarize(grad_ratio),
        "mean_error": summarize(mean_error),
        "stream_abs_cosine": summarize(diversity),
        "sigma_min": summarize(sigma_min),
        "sigma_max": summarize(sigma_max),
        "left_fix_error": summarize(left_fix),
        "right_fix_error": summarize(right_fix),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["isohc", "mhc-lite", "unconstrained", "gcn-diffusion"], choices=METHODS)
    parser.add_argument("--streams", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--depths", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "bf16", "bf16_fp32_mix"], choices=DTYPES)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--gcn-alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/stage2_stability_detectors")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "results": {},
    }

    print("Stability detectors")
    print("methods", args.methods)
    print("streams", args.streams, "depths", args.depths, "dtypes", args.dtypes)

    start = time.time()
    for dtype_name in args.dtypes:
        result["results"][dtype_name] = {}
        for method in args.methods:
            result["results"][dtype_name][method] = {}
            print(f"\n=== dtype={dtype_name} method={method} ===", flush=True)
            for n in args.streams:
                result["results"][dtype_name][method][str(n)] = {}
                for depth in args.depths:
                    t0 = time.time()
                    r = run_one(method, n, depth, args.feature_dim, args.trials, dtype_name, args.device, args.ns_steps, args.scale, args.gcn_alpha)
                    result["results"][dtype_name][method][str(n)][str(depth)] = r
                    print(
                        f"dtype={dtype_name:13s} method={method:13s} n={n:2d} L={depth:4d} "
                        f"grad={r['grad_ratio']['mean']:.4f} "
                        f"zeroE={r['energy_mean_zero']['mean']:.4f} "
                        f"cos={r['stream_abs_cosine']['mean']:.4f} "
                        f"meanMax={r['mean_error']['max']:.2e} "
                        f"smin={r['sigma_min']['min']:.4f} smax={r['sigma_max']['max']:.4f} "
                        f"time={time.time() - t0:.1f}s",
                        flush=True,
                    )

    json_path = output_dir / "stability_detectors.json"
    json_path.write_text(json.dumps(result, indent=2))
    md = [
        "# Stability Detectors",
        "",
        f"Timestamp: {result['timestamp']}",
        "",
        "| dtype | method | n | L | grad mean | mean-zero energy | stream abs cosine | mean max | sigma min | sigma max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dtype_name in args.dtypes:
        for method in args.methods:
            for n in args.streams:
                for depth in args.depths:
                    r = result["results"][dtype_name][method][str(n)][str(depth)]
                    md.append(
                        f"| {dtype_name} | {method} | {n} | {depth} | "
                        f"{r['grad_ratio']['mean']:.4f} | {r['energy_mean_zero']['mean']:.4f} | "
                        f"{r['stream_abs_cosine']['mean']:.4f} | {r['mean_error']['max']:.2e} | "
                        f"{r['sigma_min']['min']:.4f} | {r['sigma_max']['max']:.4f} |"
                    )
    md_path = output_dir / "stability_detectors.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nSaved {json_path} and {md_path}")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
