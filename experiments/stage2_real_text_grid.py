"""Run a small real-text grid and save a compact summary."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd, log_path):
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def load_result(output_dir, model, layers):
    path = Path(output_dir) / f"real_text_{model}_L{layers}.json"
    data = json.loads(path.read_text())
    metrics = data["metrics"]
    return {
        "model": model,
        "layers": layers,
        "passed": data["passed"],
        "params": data["params"],
        "final_train_loss": metrics[-1]["loss"],
        "val_loss": data["val_loss"],
        "max_grad_norm": max(m["grad_norm"] for m in metrics),
        "total_time_s": data["total_time_s"],
        "max_orth_error": max((m.get("max_orth_error", 0.0) for m in metrics)),
        "max_fix_error": max((m.get("max_fix_error", 0.0) for m in metrics)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--models", type=str, nargs="+", default=["baseline", "isohc"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/stage2_real_text_grid")
    parser.add_argument("--data-path", type=str, default="data/tinyshakespeare/input.txt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "runs": [],
    }

    for model in args.models:
        for layers in args.layers:
            log_path = output_dir / f"{model}_L{layers}.log"
            cmd = [
                sys.executable,
                "experiments/stage2_real_text_smoke.py",
                "--model",
                model,
                "--layers",
                str(layers),
                "--hidden-dim",
                str(args.hidden_dim),
                "--heads",
                str(args.heads),
                "--streams",
                str(args.streams),
                "--context-length",
                str(args.context_length),
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
                "--precision",
                args.precision,
                "--learning-rate",
                str(args.learning_rate),
                "--warmup-steps",
                str(args.warmup_steps),
                "--device",
                args.device,
                "--output-dir",
                str(output_dir),
                "--data-path",
                args.data_path,
            ]
            print("=" * 70, flush=True)
            print(f"Running {model} L{layers}", flush=True)
            print(" ".join(cmd), flush=True)
            status = run(cmd, log_path)
            print(f"status={status} log={log_path}", flush=True)
            if status != 0:
                summary["runs"].append({"model": model, "layers": layers, "status": status, "log": str(log_path)})
                break
            row = load_result(output_dir, model, layers)
            row["status"] = status
            row["log"] = str(log_path)
            summary["runs"].append(row)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        else:
            continue
        break

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Real Text Grid Summary",
        "",
        f"Timestamp: {summary['timestamp']}",
        "",
        "| model | layers | passed | params | final train loss | val loss | max grad | time s | max orth | max fix |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["runs"]:
        lines.append(
            f"| {r['model']} | {r['layers']} | {r.get('passed', False)} | {r.get('params', 0):,} | "
            f"{r.get('final_train_loss', 0):.4f} | {r.get('val_loss', 0):.4f} | "
            f"{r.get('max_grad_norm', 0):.4f} | {r.get('total_time_s', 0):.1f} | "
            f"{r.get('max_orth_error', 0):.2e} | {r.get('max_fix_error', 0):.2e} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Saved summary to {output_dir / 'summary.md'}")
    return 0 if all(r.get("status", 1) == 0 for r in summary["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
