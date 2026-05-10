"""Precision/depth diagnostics for IsoHC residual-only propagation."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isohc import iso_ns_project


VARIANTS = ("fp64", "fp32", "bf16_all", "bf16_fp32_mix")


def summarize(xs):
    return {
        "mean": float(sum(xs) / len(xs)),
        "max": float(max(xs)),
        "min": float(min(xs)),
    }


def make_h_list(n, depth, variant, device, ns_steps):
    raw_dtype = torch.float64 if variant == "fp64" else torch.float32
    h_compute_dtype = torch.float64 if variant == "fp64" else torch.float32
    hs = []
    for _ in range(depth):
        raw = torch.randn(n, n, device=device, dtype=raw_dtype) * 0.1
        h = iso_ns_project(raw, steps=ns_steps).to(h_compute_dtype)
        hs.append(h)
    return hs


def run_one(n, depth, variant, feature_dim, trials, ns_steps, device):
    if variant == "fp64":
        state_dtype = torch.float64
        matmul_dtype = torch.float64
    elif variant == "fp32":
        state_dtype = torch.float32
        matmul_dtype = torch.float32
    elif variant == "bf16_all":
        state_dtype = torch.bfloat16
        matmul_dtype = torch.bfloat16
    elif variant == "bf16_fp32_mix":
        state_dtype = torch.bfloat16
        matmul_dtype = torch.float32
    else:
        raise ValueError(variant)

    energy, grad, mean_err = [], [], []
    right_fix, left_fix, orth, smin, smax = [], [], [], [], []
    ones32 = torch.ones(n, 1, device=device, dtype=torch.float32)

    for _ in range(trials):
        hs = make_h_list(n, depth, variant, device, ns_steps)
        for h in hs:
            h32 = h.float()
            eye = torch.eye(n, device=device, dtype=torch.float32)
            orth.append(torch.linalg.norm(h32.T @ h32 - eye).item())
            right_fix.append(torch.linalg.norm(h32 @ ones32 - ones32).item())
            left_fix.append(torch.linalg.norm(h32.T @ ones32 - ones32).item())
            sv = torch.linalg.svdvals(h32)
            smin.append(sv.min().item())
            smax.append(sv.max().item())

        x0 = torch.randn(n, feature_dim, device=device, dtype=state_dtype)
        x = x0
        for h in hs:
            if variant == "bf16_fp32_mix":
                x = h.float() @ x.float()
            else:
                x = h.to(matmul_dtype) @ x.to(matmul_dtype)
        energy.append((torch.linalg.norm(x.float()) / torch.linalg.norm(x0.float())).item())
        mean_err.append(torch.linalg.norm((ones32.T @ x.float() - ones32.T @ x0.float()) / n).item())

        g_last = torch.randn(n, feature_dim, device=device, dtype=state_dtype)
        g = g_last
        for h in reversed(hs):
            if variant == "bf16_fp32_mix":
                g = h.float().T @ g.float()
            else:
                g = h.to(matmul_dtype).T @ g.to(matmul_dtype)
        grad.append((torch.linalg.norm(g.float()) / torch.linalg.norm(g_last.float())).item())

    return {
        "energy": summarize(energy),
        "grad": summarize(grad),
        "mean_error": summarize(mean_err),
        "right_fix_error": summarize(right_fix),
        "left_fix_error": summarize(left_fix),
        "orth_error": summarize(orth),
        "sigma_min": summarize(smin),
        "sigma_max": summarize(smax),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--depths", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--variants", type=str, nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="results/stage2_precision")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device
    print("Precision/depth suite")
    print("device", device_name)
    print("streams", args.streams, "depths", args.depths, "trials", args.trials)
    print("variants", args.variants)

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device_name,
        "streams": args.streams,
        "depths": args.depths,
        "feature_dim": args.feature_dim,
        "trials": args.trials,
        "ns_steps": args.ns_steps,
        "variants": args.variants,
        "results": {},
    }

    start_all = time.time()
    for variant in args.variants:
        results["results"][variant] = {}
        print(f"\n=== {variant} ===", flush=True)
        for n in args.streams:
            results["results"][variant][str(n)] = {}
            for depth in args.depths:
                t0 = time.time()
                r = run_one(n, depth, variant, args.feature_dim, args.trials, args.ns_steps, args.device)
                results["results"][variant][str(n)][str(depth)] = r
                elapsed = time.time() - t0
                print(
                    f"variant={variant:14s} n={n} L={depth:3d} "
                    f"energy={r['energy']['mean']:.6f} grad={r['grad']['mean']:.6f} "
                    f"mean_max={r['mean_error']['max']:.3e} "
                    f"left_max={r['left_fix_error']['max']:.3e} "
                    f"right_max={r['right_fix_error']['max']:.3e} "
                    f"orth_max={r['orth_error']['max']:.3e} time={elapsed:.1f}s",
                    flush=True,
                )

    json_path = output_dir / "precision_depth_suite.json"
    json_path.write_text(json.dumps(results, indent=2))

    md = [
        "# Precision Depth Suite",
        "",
        f"Timestamp: {results['timestamp']}",
        f"Device: {results['device']}",
        "",
        "| variant | n | L | energy mean | grad mean | mean max | left fix max | right fix max | orth max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in args.variants:
        for n in args.streams:
            for depth in args.depths:
                r = results["results"][variant][str(n)][str(depth)]
                md.append(
                    f"| {variant} | {n} | {depth} | {r['energy']['mean']:.6f} | "
                    f"{r['grad']['mean']:.6f} | {r['mean_error']['max']:.3e} | "
                    f"{r['left_fix_error']['max']:.3e} | {r['right_fix_error']['max']:.3e} | "
                    f"{r['orth_error']['max']:.3e} |"
                )
    md_path = output_dir / "precision_depth_suite.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"\nSaved {json_path} and {md_path}")
    print(f"Total time: {time.time() - start_all:.1f}s")


if __name__ == "__main__":
    main()
