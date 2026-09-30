"""CUDA-only baseline/phase initialization audit on an explicit real token cache.

Run each backend/precision setting in a separate process. This checks zero-update
functions and gradients, not a quality advantage after optimization. Inductor
runs also retain an eager reference for the same unmodified model parameters.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.residual_lm_diagnostic import (  # noqa: E402
    BlockOrder, PackedTokens, build_model, digest_file, scale_residual_initialization,
    source_provenance,
)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def differences(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    a, b = left.reshape(-1).numpy(), right.reshape(-1).numpy()
    maximum, squares, reference, changed = 0.0, 0.0, 0.0, 0
    for start in range(0, a.size, 1024 * 1024):
        x = a[start:start + 1024 * 1024].astype(np.float64)
        y = b[start:start + 1024 * 1024].astype(np.float64)
        delta = x - y
        maximum = max(maximum, float(np.max(np.abs(delta))))
        squares += float(np.sum(delta * delta))
        reference += float(np.sum(x * x))
        changed += int(np.count_nonzero(delta))
    return {
        "elements": int(a.size), "changed_elements": changed, "exactly_equal": changed == 0,
        "max_abs": maximum, "rms": math.sqrt(squares / max(a.size, 1)),
        "relative_l2": math.sqrt(squares / max(reference, 1e-300)),
        "difference_squared_sum": squares, "reference_squared_sum": reference,
    }


def named_differences(left, right, names):
    rows = {name: differences(left[name], right[name]) for name in names}
    squares = sum(row["difference_squared_sum"] for row in rows.values())
    reference = sum(row["reference_squared_sum"] for row in rows.values())
    return {
        "tensors": len(rows), "exactly_equal": all(row["exactly_equal"] for row in rows.values()),
        "max_abs": max(row["max_abs"] for row in rows.values()),
        "relative_l2": math.sqrt(squares / max(reference, 1e-300)),
        "changed_tensors": {name: row for name, row in rows.items() if not row["exactly_equal"]},
    }


def capture(raw_model, callable_model, x, y, training):
    callable_model.train(training)
    raw_model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = callable_model(x, y)
    backward_autocast_enabled = torch.is_autocast_enabled("cuda")
    if loss is None or not bool(torch.isfinite(loss)):
        raise FloatingPointError("CUDA forward returned a missing/nonfinite loss")
    if training:
        loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    gradients = {
        name: parameter.grad.detach().float().cpu()
        for name, parameter in raw_model.named_parameters() if parameter.grad is not None
    }
    if any(not bool(torch.isfinite(grad).all()) for grad in gradients.values()):
        raise FloatingPointError("Nonfinite initialization gradient")
    cpu_logits = logits.detach().float().cpu()
    router_squares = sum(float(grad.double().square().sum()) for name, grad in gradients.items() if "router" in name)
    result = {
        "loss": float(loss.detach()), "logits_dtype": str(logits.dtype), "loss_dtype": str(loss.dtype),
        "logits_shape": list(logits.shape), "logits_sha256_as_float32": tensor_sha(cpu_logits),
        "router_gradient_l2": math.sqrt(router_squares), "seconds_including_lazy_compile": seconds,
        "backward_cuda_autocast_enabled_at_call": backward_autocast_enabled if training else None,
    }
    raw_model.zero_grad(set_to_none=True)
    return {"metadata": result, "logits": cpu_logits, "gradients": gradients}


def compare(left, right, shared_names, gradients=True):
    result = {
        "loss_left": left["metadata"]["loss"], "loss_right": right["metadata"]["loss"],
        "loss_right_minus_left": right["metadata"]["loss"] - left["metadata"]["loss"],
        "logits": differences(left["logits"], right["logits"]),
    }
    if gradients:
        missing = [name for name in shared_names if name not in left["gradients"] or name not in right["gradients"]]
        if missing:
            raise ValueError(f"Missing shared backbone gradients: {missing}")
        result["shared_backbone_gradients"] = named_differences(left["gradients"], right["gradients"], shared_names)
    return result


def run(args, report):
    import torch._inductor.config as inductor_config
    import torch._functorch.config as functorch_config

    try:
        installed_casts = getattr(inductor_config, "emulate_precision_casts")
        supported = True
    except AttributeError:
        installed_casts, supported = None, False
    report["precision_configuration"] = {
        "emulate_precision_casts_supported": supported, "installed_default": installed_casts,
        "requested": args.emulate_precision_casts, "config_file": inductor_config.__file__,
        "config_file_sha256": digest_file(inductor_config.__file__),
    }
    if args.emulate_precision_casts == "true":
        if not supported:
            report.update(status="unsupported_precision_option", gpu_results_produced=False)
            return
        inductor_config.emulate_precision_casts = True
    report["precision_configuration"]["effective"] = getattr(inductor_config, "emulate_precision_casts", None)
    source_lines = Path(functorch_config.__file__).read_text().splitlines()
    excerpts = sorted({index for position, line in enumerate(source_lines) if "backward_pass_autocast" in line
                       for index in range(max(0, position - 8), min(len(source_lines), position + 9))})
    try:
        installed_backward = functorch_config.backward_pass_autocast
        backward_supported = True
    except AttributeError:
        installed_backward, backward_supported = None, False
    report["backward_autocast_configuration"] = {
        "config_name": "torch._functorch.config.backward_pass_autocast", "supported": backward_supported,
        "installed_default": installed_backward, "requested": args.backward_autocast,
        "config_file": functorch_config.__file__, "config_file_sha256": digest_file(functorch_config.__file__),
        "source_excerpt": [{"line": index + 1, "text": source_lines[index]} for index in excerpts],
    }
    if args.backward_autocast == "off":
        if not backward_supported:
            report.update(status="unsupported_backward_autocast_option", gpu_results_produced=False)
            return
        functorch_config.backward_pass_autocast = "off"
    report["backward_autocast_configuration"]["effective"] = getattr(functorch_config, "backward_pass_autocast", None)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Actual CUDA with BF16 support is required; CPU results are not substituted")
    torch.use_deterministic_algorithms(args.deterministic)
    report["environment"] = {
        "python": sys.version, "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "capability": list(torch.cuda.get_device_capability()), "cpu_threads": torch.get_num_threads(),
        "amp_dtype": "torch.bfloat16", "parameter_dtype": "torch.float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }
    train = PackedTokens(args.train_cache, args.vocab_size, args.context_length)
    data = PackedTokens(args.val_cache, args.vocab_size, args.context_length) if args.val_cache else train
    ids = torch.arange(args.batch_size) if args.val_cache else BlockOrder(train.blocks, args.data_seed).take(args.batch_size)
    if args.batch_size > data.blocks:
        raise ValueError("Diagnostic batch exceeds cache block count")
    x_cpu, y_cpu = data.batch(ids)
    report["data"] = {
        "train_cache": train.metadata, "diagnostic_cache": data.metadata, "block_ids": ids.tolist(),
        "batch_source": "fixed_first_validation_blocks" if args.val_cache else "first_shuffled_training_blocks",
        "inputs_sha256": tensor_sha(x_cpu), "targets_sha256": tensor_sha(y_cpu),
    }
    config = dict(vocab_size=args.vocab_size, width=args.width, layers=args.layers, heads=args.heads,
                  context_length=args.context_length, mlp_ratio=4., dropout=0., boundary=args.boundary)
    report["model_config"] = {**config, "model_seed": args.model_seed, "data_seed": args.data_seed,
                              "residual_init_scale": args.residual_init_scale, "backend": args.backend}
    models, parameters = {}, {}
    for method in ("baseline", "phase-adjoint"):
        torch.manual_seed(args.model_seed)
        model = build_model(dict(config, method=method))
        scale_residual_initialization(model, args.residual_init_scale)
        models[method] = model.cuda()
        parameters[method] = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()}
    shared = list(parameters["baseline"])
    report["shared_backbone_initial_parameters"] = named_differences(parameters["baseline"], parameters["phase-adjoint"], shared)
    report["model_parameter_dtypes"] = {
        method: sorted({str(parameter.dtype) for parameter in model.parameters()}) for method, model in models.items()
    }
    report["provenance"] = source_provenance(models["baseline"])
    report["provenance"]["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = digest_file(__file__)
    x, y = x_cpu.cuda(), y_cpu.cuda()
    observations, comparisons = {}, {}
    report["observations"], report["comparisons"] = observations, comparisons
    results = {}
    for method, model in models.items():
        eager = [capture(model, model, x, y, True) for _ in range(args.repeats)]
        eager_eval = capture(model, model, x, y, False)
        results[(method, "eager_train")] = eager[0]
        results[(method, "eager_eval")] = eager_eval
        observations[method] = {"eager_train": [row["metadata"] for row in eager], "eager_eval": eager_eval["metadata"]}
        if args.repeats > 1:
            comparisons[f"{method}/eager_repeat_noise_floor"] = compare(eager[0], eager[1], shared)
        comparisons[f"{method}/eager_train_vs_eval"] = compare(eager[0], eager_eval, shared, gradients=False)
        if args.backend == "inductor":
            compiled = torch.compile(model, backend="inductor", mode="default")
            compiled_train = [capture(model, compiled, x, y, True) for _ in range(args.repeats)]
            compiled_eval = capture(model, compiled, x, y, False)
            results[(method, "inductor_train")] = compiled_train[0]
            results[(method, "inductor_eval")] = compiled_eval
            observations[method].update(inductor_train=[row["metadata"] for row in compiled_train], inductor_eval=compiled_eval["metadata"])
            comparisons[f"{method}/eager_vs_inductor_train"] = compare(eager[0], compiled_train[0], shared)
            comparisons[f"{method}/eager_vs_inductor_eval"] = compare(eager_eval, compiled_eval, shared, gradients=False)
            if args.repeats > 1:
                comparisons[f"{method}/inductor_repeat_noise_floor"] = compare(compiled_train[0], compiled_train[1], shared)
    for mode in ("eager_train", "eager_eval", "inductor_train", "inductor_eval"):
        if ("baseline", mode) in results:
            comparisons[f"baseline_vs_phase/{mode}"] = compare(results[("baseline", mode)], results[("phase-adjoint", mode)], shared, gradients=mode.endswith("train"))
    report.update(status="complete", gpu_results_produced=True, optimizer_updates=0,
                  scope="Real CUDA zero-update initialization/gradient audit; no quality ranking or post-update equivalence claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache")
    parser.add_argument("--backend", choices=("eager", "inductor"), required=True)
    parser.add_argument("--emulate-precision-casts", choices=("default", "true"), default="default")
    parser.add_argument("--backward-autocast", choices=("default", "off"), default="default")
    for name, default in (("vocab-size", 50257), ("context-length", 512), ("layers", 24),
                          ("width", 256), ("heads", 4), ("batch-size", 2), ("model-seed", 419),
                          ("data-seed", 20260929), ("repeats", 2)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--boundary", type=int)
    parser.add_argument("--residual-init-scale", type=float)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats not in (1, 2) or args.layers < 2:
        parser.error("Positive batch, >=2 layers and repeats=1 or 2 are required")
    args.boundary = args.layers // 2 if args.boundary is None else args.boundary
    args.residual_init_scale = 1 / math.sqrt(2 * args.layers) if args.residual_init_scale is None else args.residual_init_scale
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "diagnostic_script_sha256": digest_file(__file__), "gpu_results_produced": False}
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "status": report["status"],
                          "precision_configuration": report.get("precision_configuration"),
                          "backward_autocast_configuration": report.get("backward_autocast_configuration"),
                          "comparisons": {key: {"loss_delta": value["loss_right_minus_left"],
                                                "logits_max_abs": value["logits"]["max_abs"]}
                                          for key, value in report.get("comparisons", {}).items()}}, indent=2))


if __name__ == "__main__":
    main()
