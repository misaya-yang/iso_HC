"""Reproducible CPU integration probe for the adjoint residual candidate.

This is synthetic training-path and local CPU cost evidence, not evidence of
language-model quality, GPU performance, scaling, or superiority to HC baselines.
No dataset download, accelerator, or remote service is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.adjoint import AdjointHCTransformer  # noqa: E402
from lm.models import BaselineTransformer  # noqa: E402


ARMS = {
    "baseline": None,
    "dynamic_signed": {"routing": "dynamic", "carrier": "signed"},
    "static_signed": {"routing": "static", "carrier": "signed"},
    "dynamic_signed_frozen_aux": {
        "routing": "dynamic", "carrier": "signed", "freeze_aux_updates": True,
    },
}


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def router_parameters(model):
    return {
        name: param for name, param in model.named_parameters()
        if "router" in name
    }


def l2(tensors):
    return math.sqrt(sum(float(tensor.detach().double().square().sum()) for tensor in tensors))


def shared_models(shape, seed, arms=ARMS):
    torch.manual_seed(seed)
    baseline = BaselineTransformer(**shape).cpu()
    reference = baseline.state_dict()
    models = {}
    equality = {}
    for name, kwargs in arms.items():
        if kwargs is None:
            model = baseline
        else:
            torch.manual_seed(seed)
            model = AdjointHCTransformer(**shape, **kwargs).cpu()
        # Check constructor RNG identity directly; do not repair it by loading
        # baseline weights before checking the claimed same-seed initialization.
        equality[name] = all(torch.equal(model.state_dict()[key], value) for key, value in reference.items())
        if not equality[name]:
            raise AssertionError(f"Backbone differs at initialization: {name}")
        models[name] = model
    return models, equality


def markov_batches(*, vocab_size, batch_size, context_length, batches, sample_seed, permutation_seed):
    """A fixed first-order source: p(next=permutation[current])=.8, else uniform.

    It intentionally tests ordinary token cross-entropy learning, not long-range
    memory. Train/validation share the source kernel and use independent draws.
    """
    permutation_generator = torch.Generator().manual_seed(permutation_seed)
    permutation = torch.randperm(vocab_size, generator=permutation_generator)
    generator = torch.Generator().manual_seed(sample_seed)
    tokens = torch.empty(batches, batch_size, context_length + 1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(vocab_size, (batches, batch_size), generator=generator)
    for index in range(context_length):
        preferred = permutation[tokens[:, :, index]]
        noise = torch.randint(vocab_size, preferred.shape, generator=generator)
        follow = torch.rand(preferred.shape, generator=generator) < 0.8
        tokens[:, :, index + 1] = torch.where(follow, preferred, noise)
    return tokens[:, :, :-1].contiguous(), tokens[:, :, 1:].contiguous(), permutation


@torch.no_grad()
def validation_loss(model, batches):
    was_training = model.training
    model.eval()
    values = [float(model(x, y)[1]) for x, y in zip(*batches)]
    model.train(was_training)
    return statistics.mean(values)


@torch.no_grad()
def auxiliary_update(model, x):
    """Compare final auxiliary state with the live initial carrier on one batch."""
    if isinstance(model, AdjointHCTransformer):
        states = model.get_stream_states(x)
        initial, final = states[0], states[-1]
        difference = final[1:] - initial[1:]
        return {
            "final_minus_initial_aux_rms": float(difference.square().mean().sqrt()),
            "initial_aux_rms": float(initial[1:].square().mean().sqrt()),
            "final_aux_rms": float(final[1:].square().mean().sqrt()),
        }
    return None


def training_probe():
    shape = dict(vocab_size=32, d_model=64, num_layers=2, num_heads=4,
                 context_length=32, mlp_ratio=4, dropout=0.0, use_flash=True)
    steps, batch_size = 80, 4
    models, backbone_equal = shared_models(shape, seed=419)
    train_x, train_y, permutation = markov_batches(
        vocab_size=32, batch_size=batch_size, context_length=32, batches=steps,
        sample_seed=2718, permutation_seed=3141,
    )
    val_x, val_y, _ = markov_batches(
        vocab_size=32, batch_size=batch_size, context_length=32, batches=16,
        sample_seed=1618, permutation_seed=3141,
    )
    with torch.no_grad():
        baseline_logits = models["baseline"](train_x[0])[0]
    records = {}
    for name, model in models.items():
        model.train()
        initial_parameters = {key: value.detach().clone() for key, value in router_parameters(model).items()}
        with torch.no_grad():
            init_error = float((model(train_x[0])[0] - baseline_logits).abs().max())
        initial_aux = auxiliary_update(model, train_x[0])
        curve = [{"step": 0, "validation_loss": validation_loss(model, (val_x, val_y)),
                  "router_parameter_l2_change": 0.0}]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
        training_losses, gradient_norms = [], []
        started = time.perf_counter()
        for step, (x, y) in enumerate(zip(train_x, train_y), start=1):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {name} loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite {name} gradient at step {step}")
            optimizer.step()
            training_losses.append(float(loss.detach()))
            gradient_norms.append(float(norm))
            if step % 20 == 0:
                curve.append({
                    "step": step,
                    "validation_loss": validation_loss(model, (val_x, val_y)),
                    "router_parameter_l2_change": l2(
                        param - initial_parameters[key] for key, param in router_parameters(model).items()
                    ),
                })
        elapsed = time.perf_counter() - started
        records[name] = {
            "parameters": sum(param.numel() for param in model.parameters()),
            "backbone_exactly_shared_at_init": backbone_equal[name],
            "initial_logits_max_abs_error_vs_baseline": init_error,
            "initial_auxiliary": initial_aux,
            "final_auxiliary": auxiliary_update(model, train_x[0]),
            "curve": curve,
            "train_loss_per_step": training_losses,
            "gradient_l2_before_clipping_per_step": gradient_norms,
            "first_train_loss": training_losses[0],
            "last_train_loss": training_losses[-1],
            "first_validation_loss": curve[0]["validation_loss"],
            "last_validation_loss": curve[-1]["validation_loss"],
            "final_router_parameter_l2_change": curve[-1]["router_parameter_l2_change"],
            "elapsed_seconds_including_periodic_validation": elapsed,
        }
    return {
        "shape": shape, "model_seed": 419, "steps": steps, "batch_size": batch_size,
        "tokens_per_step": batch_size * 32, "training_tokens_per_arm": steps * batch_size * 32,
        "validation_tokens_per_evaluation": 16 * batch_size * 32,
        "optimizer": {"name": "AdamW", "lr": 0.001, "betas": [0.9, 0.999],
                      "weight_decay": 0.01, "max_gradient_norm": 1.0},
        "data": {"type": "first_order_permutation_markov", "preferred_transition_probability": 0.8,
                 "random_transition_probability": 0.2, "train_seed": 2718, "validation_seed": 1618,
                 "permutation_seed": 3141, "permutation": permutation.tolist(),
                 "same_precomputed_train_batches_all_arms": True,
                 "train_inputs_sha256": tensor_digest(train_x),
                 "train_targets_sha256": tensor_digest(train_y),
                 "validation_inputs_sha256": tensor_digest(val_x),
                 "validation_targets_sha256": tensor_digest(val_y),
                 "scope": "Independent samples of the same source; not a long-range or natural-language task."},
        "arms": records,
    }


def gradient_probe():
    shape = dict(vocab_size=32, d_model=64, num_layers=2, num_heads=4,
                 context_length=32, mlp_ratio=4, dropout=0.0, use_flash=True)
    models, shared = shared_models(shape, 419, {
        "baseline": None,
        "dynamic_signed": {"routing": "dynamic", "carrier": "signed"},
        "dynamic_zero": {"routing": "dynamic", "carrier": "zero"},
    })
    x, y, _ = markov_batches(vocab_size=32, batch_size=4, context_length=32,
                             batches=1, sample_seed=2718, permutation_seed=3141)
    result = {}
    for name, model in models.items():
        logits, loss = model(x[0], y[0])
        loss.backward()
        norms = {key: float(param.grad.norm()) if param.grad is not None else None
                 for key, param in router_parameters(model).items()}
        result[name] = {
            "backbone_exactly_shared_at_init": shared[name], "loss": float(loss.detach()),
            "router_gradient_l2": l2(param.grad for param in router_parameters(model).values() if param.grad is not None),
            "router_gradient_l2_by_parameter": norms,
        }
    return {"scope": "Real full-model NTP backward at exact baseline initialization", "arms": result}


def percentiles(samples):
    values = torch.tensor(samples, dtype=torch.float64)
    return {"median_ms": statistics.median(samples),
            "p10_ms": float(torch.quantile(values, 0.1)),
            "p90_ms": float(torch.quantile(values, 0.9)), "samples_ms": samples}


def cpu_benchmark(*, width, layers, length, vocab_size, repeats):
    shape = dict(vocab_size=vocab_size, d_model=width, num_layers=layers, num_heads=4,
                 context_length=length, mlp_ratio=4, dropout=0.0, use_flash=True)
    models, shared = shared_models(shape, 419, {
        "baseline": None, "dynamic_signed": {"routing": "dynamic", "carrier": "signed"},
    })
    x, y, _ = markov_batches(vocab_size=vocab_size, batch_size=2, context_length=length,
                             batches=1, sample_seed=123, permutation_seed=3141)

    def iteration(model):
        model.zero_grad(set_to_none=True)
        model(x[0], y[0])[1].backward()

    for model in models.values():
        model.train()
        for _ in range(3):
            iteration(model)
    measurements = {name: [] for name in models}
    names = list(models)
    for repeat in range(repeats):
        for name in names if repeat % 2 == 0 else reversed(names):
            started = time.perf_counter()
            iteration(models[name])
            measurements[name].append(1000 * (time.perf_counter() - started))
    return {
        "scope": "Whole-model forward+cross_entropy+backward+zero_grad; no optimizer, transfer, or data loading. Local CPU only.",
        "shape": shape, "batch_size": 2, "dtype": "float32", "warmup_iterations": 3,
        "measured_iterations": repeats, "interleaved_alternating_arm_order": True,
        "arms": {name: {**percentiles(values), "parameters": sum(p.numel() for p in models[name].parameters()),
                         "backbone_exactly_shared_at_init": shared[name]} for name, values in measurements.items()},
        "candidate_over_baseline_median_ratio": statistics.median(measurements["dynamic_signed"]) / statistics.median(measurements["baseline"]),
        "limits": "No GPU, fused-kernel, large-model, wall-clock matched training, or activation-memory conclusion follows.",
    }


def compile_probe():
    shape = dict(vocab_size=32, d_model=64, num_layers=2, num_heads=4,
                 context_length=32, mlp_ratio=4, dropout=0.0, use_flash=True)
    models, _ = shared_models(shape, 419, {
        "baseline": None, "dynamic_signed": {"routing": "dynamic", "carrier": "signed"},
    })
    x, y, _ = markov_batches(vocab_size=32, batch_size=4, context_length=32,
                             batches=1, sample_seed=123, permutation_seed=3141)
    model = models["dynamic_signed"]
    eager_logits, eager_loss = model(x[0], y[0])
    eager_loss.backward()
    eager_grads = {key: param.grad.detach().clone() for key, param in model.named_parameters() if param.grad is not None}
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    try:
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        logits, loss = compiled(x[0], y[0])
        loss.backward()
        return {"attempted": True, "backend": "aot_eager", "fullgraph": True, "succeeded": True,
                "logits_max_abs_difference": float((eager_logits - logits).detach().abs().max()),
                "gradient_max_abs_difference": max(float((param.grad - eager_grads[key]).abs().max())
                                                   for key, param in model.named_parameters() if param.grad is not None),
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "Single fixed shape fullgraph forward+backward; not fusion or compiled throughput."}
    except Exception as error:
        return {"attempted": True, "backend": "aot_eager", "fullgraph": True, "succeeded": False,
                "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/adjoint_hc_20260926/probe.json")
    parser.add_argument("--compile-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    report = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": True, "integration_training_only": True,
        "natural_language_training_performed": False, "gpu_used": False, "data_downloaded": False,
        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                        "processor": platform.processor(), "device": "cpu", "dtype": "float32",
                        "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), ROOT / "lm/adjoint.py", ROOT / "lm/models.py"]},
        "gradient_probe": gradient_probe(),
        "training": training_probe(),
        "cpu_benchmark": {
            "selection": "Two predefined shapes; retain both to expose width dependence, not select a favorable result.",
            "small": cpu_benchmark(width=128, layers=2, length=64, vocab_size=32, repeats=20),
            "wide": cpu_benchmark(width=512, layers=4, length=128, vocab_size=128, repeats=8),
        },
        "compile_check": compile_probe() if args.compile_check else {"attempted": False},
        "conclusions_permitted": ["Exact baseline initialization on this shape", "Initial full-model routing gradient",
                                  "Finite ordinary NTP optimization on a synthetic source", "Unfused local CPU execution cost"],
        "conclusions_not_permitted": ["Natural-language quality advantage", "SOTA", "Mature HC comparison",
                                      "Large-model stability", "GPU cost or end-to-end deployment efficiency"],
    }
    training = report["training"]["arms"]
    gradients = report["gradient_probe"]["arms"]
    report["integration_checks"] = {
        "same_seed_backbone_equality": all(arm["backbone_exactly_shared_at_init"] for arm in training.values()),
        "exact_initial_logits": all(arm["initial_logits_max_abs_error_vs_baseline"] == 0 for arm in training.values()),
        "signed_carrier_opens_route_gradient": gradients["dynamic_signed"]["router_gradient_l2"] > 0,
        "zero_carrier_has_dead_initial_route_gradient": gradients["dynamic_zero"]["router_gradient_l2"] == 0,
        "all_arms_learn_synthetic_source": all(arm["last_validation_loss"] < arm["first_validation_loss"] for arm in training.values()),
        "all_candidate_routers_move": all(arm["final_router_parameter_l2_change"] > 0 for name, arm in training.items() if name != "baseline"),
        "dynamic_auxiliary_writes_activate": training["dynamic_signed"]["initial_auxiliary"]["final_minus_initial_aux_rms"] == 0
            and training["dynamic_signed"]["final_auxiliary"]["final_minus_initial_aux_rms"] > 0,
        "static_auxiliary_writes_activate": training["static_signed"]["initial_auxiliary"]["final_minus_initial_aux_rms"] == 0
            and training["static_signed"]["final_auxiliary"]["final_minus_initial_aux_rms"] > 0,
        "frozen_auxiliary_control_stays_frozen": training["dynamic_signed_frozen_aux"]["final_auxiliary"]["final_minus_initial_aux_rms"] == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output),
                      "gradient_l2": {key: value["router_gradient_l2"] for key, value in report["gradient_probe"]["arms"].items()},
                      "final_validation_loss": {key: value["last_validation_loss"] for key, value in report["training"]["arms"].items()},
                      "cpu_median_ratios": {name: report["cpu_benchmark"][name]["candidate_over_baseline_median_ratio"]
                                            for name in ("small", "wide")},
                      "compile_check": report["compile_check"]}, indent=2))


if __name__ == "__main__":
    main()
