"""Causal-control suites for the static Birkhoff/IsoHC comparison."""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.lm_5090_next_runs import build_preset_configs, run_single
from lm.mixing import MHCMixing
from lm.transport_analysis import collect_transport_report


DEPTH_PRESETS = {
    24: {"d_model": 704, "num_heads": 8, "parameters": 178_516_487,
         "num_transports": 48},
    48: {"d_model": 512, "num_heads": 8, "parameters": 177_041_159,
         "num_transports": 96},
    72: {"d_model": 416, "num_heads": 8, "parameters": 170_703_431,
         "num_transports": 144},
    96: {"d_model": 368, "num_heads": 8, "parameters": 174_765_479,
         "num_transports": 192},
    128: {"d_model": 320, "num_heads": 8, "parameters": 173_618_055,
          "num_transports": 256},
}


def symmetric_birkhoff_gain(n_streams, diag_bias, temperature):
    if n_streams < 2:
        raise ValueError("n_streams must be at least 2")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    diagonal = math.exp(diag_bias / temperature)
    return (diagonal - 1.0) / (diagonal + n_streams - 1.0)


def identity_hc_parameter_count(
    vocab_size, context_length, n_streams, num_layers, d_model
):
    per_layer = 12 * d_model * d_model + 2 * d_model + 4 * n_streams
    shared = (
        vocab_size * d_model
        + context_length * d_model
        + n_streams * d_model
        + d_model
        + n_streams
        + 3
    )
    return num_layers * per_layer + shared


def build_depth_summary(summaries, eps=1e-300):
    rows = []
    for summary in summaries:
        config = summary["config"]
        row = {
            "success": bool(summary.get("success", False)),
            "method": config["method"],
            "experiment_variant": config["experiment_variant"],
            "num_layers": int(config["num_layers"]),
            "d_model": int(config["d_model"]),
            "target_parameters": config.get("target_parameters"),
        }
        if not row["success"]:
            row["error"] = summary.get("error")
            rows.append(row)
            continue
        transport = summary["final_transport"]
        final = transport["final"]
        steps = transport["steps"]
        row.update({
            "num_transports": int(transport["num_transports"]),
            "log_composite_sv_min": math.log(max(
                float(final["composite_sv_min"]), eps
            )),
            "log_composite_sv_max": math.log(max(
                float(final["composite_sv_max"]), eps
            )),
            "sum_step_log_sv_min": sum(
                math.log(max(float(step["sv_min"]), eps))
                for step in steps
            ),
            "sum_step_log_sv_max": sum(
                math.log(max(float(step["sv_max"]), eps))
                for step in steps
            ),
        })
        rows.append(row)
    return {"metric_schema_version": 2, "rows": rows}


@torch.no_grad()
def build_geometry_report(n_streams, num_transports, seed):
    rows = []
    grid = itertools.product(
        (2.0, 4.0, 6.0, 8.0),
        (0.5, 1.0, 2.0),
        (0.0, 0.01),
        (5, 10, 20),
    )
    for diag_bias, temperature, noise_std, sinkhorn_iters in grid:
        torch.manual_seed(seed)
        matrices = []
        for index in range(num_transports):
            H = MHCMixing(
                n_streams,
                sinkhorn_iters=sinkhorn_iters,
                temperature=temperature,
                diag_bias=diag_bias,
                noise_std=noise_std,
            )()
            matrices.append({
                "index": index,
                "branch": "transport",
                "layer": index,
                "H": H,
            })
        analytic = (
            symmetric_birkhoff_gain(n_streams, diag_bias, temperature)
            if noise_std == 0.0 else None
        )
        rows.append({
            "diag_bias": diag_bias,
            "temperature": temperature,
            "noise_std": noise_std,
            "sinkhorn_iters": sinkhorn_iters,
            "n_streams": n_streams,
            "num_transports": num_transports,
            "analytic_single_step_gain": analytic,
            "analytic_composite_gain": (
                analytic ** num_transports if analytic is not None else None
            ),
            "measured": collect_transport_report(matrices),
        })
    return {"metric_schema_version": 2, "seed": seed, "rows": rows}


def _birkhoff_kwargs(diag_bias, identity_blend=1.0):
    return {
        "diag_bias": float(diag_bias),
        "temperature": 1.0,
        "noise_std": 0.0,
        "sinkhorn_iters": 10,
        "identity_blend": float(identity_blend),
    }


def build_suite_configs(
    suite, output_dir, dataset, total_tokens, batch_size, seed, use_compile
):
    if suite not in {"p0-smoke", "p0-train", "p0-depth"}:
        raise ValueError(f"Unknown training suite: {suite}")
    configs = []

    def add(
        preset, method, variant, mixing_kwargs=None,
        freeze_mixing=False, structural=None,
    ):
        config = build_preset_configs(
            preset=preset,
            methods=[method],
            output_dir=output_dir,
            total_tokens=total_tokens,
            batch_size=batch_size,
            seed=seed,
            dataset=dataset,
            use_compile=use_compile,
        )[0]
        config.update({
            "experiment_variant": variant,
            "mixing_kwargs": dict(mixing_kwargs or {}),
            "lambda_a": 0.01,
            "lambda_b": 0.01,
            "freeze_mixing": bool(freeze_mixing),
            "metric_schema_version": 2,
            "save_dir": str(Path(output_dir) / f"{variant}_seed{seed}"),
        })
        if structural:
            config.update(structural)
        if suite == "p0-smoke":
            config["diagnostics_every"] = 1
        configs.append(config)

    if suite == "p0-smoke":
        gain = symmetric_birkhoff_gain(4, 4.0, 1.0)
        add("run0", "identity-hc", "identity_hc")
        add("run0", "isohc", "isohc")
        add("run0", "static-birkhoff-hc", "birkhoff_d4_trainable",
            _birkhoff_kwargs(4.0))
        add("run0", "static-birkhoff-hc", "birkhoff_d4_frozen",
            _birkhoff_kwargs(4.0), freeze_mixing=True)
        add("run0", "static-birkhoff-hc", "birkhoff_d4_blend05",
            _birkhoff_kwargs(4.0, identity_blend=0.5))
        add("run0", "scaled-isohc", "scaled_isohc_match_d4",
            {"complement_scale": gain})
        return configs

    if suite == "p0-train":
        add("fe-deep-48l-512", "identity-hc", "identity_hc")
        add("fe-deep-48l-512", "isohc", "isohc")
        for bias in (2.0, 4.0, 6.0, 8.0):
            label = f"d{int(bias)}"
            add("fe-deep-48l-512", "static-birkhoff-hc",
                f"birkhoff_{label}_trainable", _birkhoff_kwargs(bias))
            add("fe-deep-48l-512", "static-birkhoff-hc",
                f"birkhoff_{label}_frozen", _birkhoff_kwargs(bias),
                freeze_mixing=True)
            add("fe-deep-48l-512", "scaled-isohc",
                f"scaled_isohc_match_{label}", {
                    "complement_scale": symmetric_birkhoff_gain(
                        4, bias, 1.0
                    )
                })
        return configs

    for layers, depth in DEPTH_PRESETS.items():
        structural = {
            "num_layers": layers,
            "d_model": depth["d_model"],
            "num_heads": depth["num_heads"],
            "target_parameters": depth["parameters"],
            "expected_num_transports": depth["num_transports"],
        }
        prefix = f"depth{layers}"
        gain = symmetric_birkhoff_gain(4, 4.0, 1.0)
        add("fe-deep-48l-512", "identity-hc", f"{prefix}_identity_hc",
            structural=structural)
        add("fe-deep-48l-512", "isohc", f"{prefix}_isohc",
            structural=structural)
        add("fe-deep-48l-512", "static-birkhoff-hc",
            f"{prefix}_birkhoff_d4_trainable", _birkhoff_kwargs(4.0),
            structural=structural)
        add("fe-deep-48l-512", "static-birkhoff-hc",
            f"{prefix}_birkhoff_d4_frozen", _birkhoff_kwargs(4.0),
            freeze_mixing=True, structural=structural)
        add("fe-deep-48l-512", "scaled-isohc",
            f"{prefix}_scaled_isohc_match_d4",
            {"complement_scale": gain}, structural=structural)
    return configs


def ensure_run_is_new(run_dir):
    summary = Path(run_dir) / "run_summary.json"
    if summary.exists():
        raise FileExistsError(f"Completed run already exists: {summary}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")
    return path


def run_posthoc(run_dirs, args):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("analyze_lm_mechanisms.py")),
        "--run_dirs",
        *map(str, run_dirs),
        "--output_dir",
        str(Path(args.output_dir) / "posthoc"),
        "--dataset",
        args.dataset,
        "--eval_batches",
        "4",
        "--intervention_stride",
        "8",
        "--intervention_scales",
        "0", "0.25", "0.5", "0.75", "1",
        "--num_workers",
        "0",
    ]
    if args.val_cache_path:
        command.extend(["--val_cache_path", args.val_cache_path])
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HC causal-control experiment suites"
    )
    parser.add_argument(
        "--suite",
        choices=["geometry", "p0-smoke", "p0-train", "p0-depth"],
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="random")
    parser.add_argument("--total_tokens", type=int, default=262_144)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_streams", type=int, default=4)
    parser.add_argument("--num_transports", type=int, default=96)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--auto_batch", action="store_true")
    parser.add_argument("--memory_target_gb", type=float, default=30.0)
    parser.add_argument("--train_cache_path")
    parser.add_argument("--val_cache_path")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--skip_posthoc", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.suite == "geometry":
        path = output_dir / "geometry_summary.json"
        if path.exists():
            raise FileExistsError(
                f"Completed geometry report already exists: {path}"
            )
        write_json(path, build_geometry_report(
            args.n_streams, args.num_transports, args.seed
        ))
        print(f"Wrote {path}")
        return

    if args.suite in {"p0-train", "p0-depth"} and not torch.cuda.is_available():
        raise RuntimeError(
            f"{args.suite} requires CUDA; use p0-smoke for CPU checks"
        )
    summary_path = output_dir / "suite_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Completed suite summary already exists: {summary_path}"
        )

    configs = build_suite_configs(
        args.suite,
        args.output_dir,
        args.dataset,
        args.total_tokens,
        args.batch_size,
        args.seed,
        not args.no_compile,
    )
    summaries = []
    run_dirs = []
    for config in configs:
        config.update({
            "train_cache_path": args.train_cache_path,
            "val_cache_path": args.val_cache_path,
            "vocab_size": args.vocab_size,
        })
        ensure_run_is_new(config["save_dir"])
        summaries.append(run_single(
            config,
            auto_batch=args.auto_batch,
            memory_target_gb=args.memory_target_gb,
        ))
        run_dirs.append(Path(config["save_dir"]))

    write_json(summary_path, summaries)
    print(f"Wrote {summary_path}")
    if args.suite == "p0-depth":
        path = output_dir / "depth_summary.json"
        write_json(path, build_depth_summary(summaries))
        print(f"Wrote {path}")
    if not args.skip_posthoc:
        run_posthoc(run_dirs, args)


if __name__ == "__main__":
    main()
