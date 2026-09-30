"""Local-cache residual LM diagnostics with reproducible optimizer-boundary resume.

No tokenizer, download, MPS fallback, or natural-language claim is implicit.
``--updates`` fixes the LR schedule; ``--stop-after`` interrupts that schedule.
GPU profiling is an exclusive mode and includes the complete optimizer step.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm.adjoint import AdjointHCTransformer  # noqa: E402
from lm.models import BaselineTransformer, CausalSelfAttention, MLP  # noqa: E402

METHODS = (
    "baseline", "gain", "adjoint", "adjoint-frozen", "adjoint-shear",
    "phase-adjoint", "phase-adjoint-frozen", "phase-adjoint-shear", "phase-adjoint-post-frozen",
    "terminal-adjoint", "boundary-skip", "block-attnres",
)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PackedTokens:
    """Nonoverlapping input blocks; every target position is scored once per pass."""

    def __init__(self, path, vocab_size, context_length, manifest_path=None):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local token cache does not exist: {path}")
        if path.suffix != ".pt":
            raise ValueError("Token caches must be explicit .pt tensors")
        candidates = [Path(manifest_path)] if manifest_path else [
            path.with_suffix(".manifest.json"), path.with_name(path.name + ".manifest.json"),
            path.parent / "manifest.json",
        ]
        manifest_path = next((p.expanduser().resolve() for p in candidates if p.expanduser().is_file()), None)
        if manifest_path is None:
            raise FileNotFoundError(f"Required token-cache manifest is missing beside {path}")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("Cache manifest must be a nonempty JSON object")
        if manifest.get("status") != "complete":
            raise ValueError("Cache manifest must have status=complete")
        if "outputs" in manifest:
            matches = [(split, entry) for split, entry in manifest["outputs"].items()
                       if Path(entry.get("path", "")).name == path.name]
            if len(matches) != 1:
                raise ValueError("Cache must match exactly one manifest output entry")
            split, entry = matches[0]
            declared_sha = entry.get("file_sha256")
        elif "synthetic" in str(manifest.get("source", "")).lower():
            entry = manifest.get("files", {}).get(path.name)
            if entry is None:
                raise ValueError("Synthetic cache is absent from its manifest")
            split, declared_sha = manifest.get("split"), entry.get("sha256")
        else:
            raise ValueError("Real-data manifest must bind the cache through outputs")
        try:
            tokens = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            tokens = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
            raise ValueError("Cache must contain one 1D token tensor; row caches are not accepted")
        if tokens.is_floating_point() or tokens.is_complex() or tokens.dtype == torch.bool:
            raise ValueError("Token IDs must have an integer dtype")
        if context_length < 1 or vocab_size < 2 or tokens.numel() <= context_length:
            raise ValueError("Positive context, vocab >= 2, and at least T+1 tokens are required")
        for chunk in tokens.split(1024 * 1024):
            ids = chunk.long()
            if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
                raise ValueError(f"Token ID outside declared vocabulary [0, {vocab_size})")
        self.tokens = tokens
        self.context_length = context_length
        self.blocks = (tokens.numel() - 1) // context_length
        declared_vocab = manifest.get("vocab_size", manifest.get("tokenizer", {}).get("vocab_size"))
        if declared_vocab is not None and declared_vocab != vocab_size:
            raise ValueError("Manifest vocabulary does not match the explicit vocabulary")
        cache_sha = digest_file(path)
        if declared_sha != cache_sha:
            raise ValueError("Cache SHA does not match its manifest")
        if "outputs" in manifest and (entry.get("tokens") != tokens.numel() or entry.get("dtype") != str(tokens.dtype)):
            raise ValueError("Cache token count or dtype does not match its manifest")
        self.metadata = {
            "path": str(path), "sha256": cache_sha, "dtype": str(tokens.dtype),
            "tokens": tokens.numel(), "blocks": self.blocks, "context_length": context_length,
            "unused_tail_targets": tokens.numel() - 1 - self.blocks * context_length,
            "vocab_size": vocab_size,
            "manifest_path": str(manifest_path), "manifest_sha256": digest_file(manifest_path),
            "source": manifest.get("source", manifest.get("dataset")), "split": split,
            "source_revision": manifest.get("source_revision"), "tokenizer": manifest.get("tokenizer"),
        }

    def batch(self, block_ids):
        ids = torch.as_tensor(block_ids, dtype=torch.long)
        positions = ids[:, None] * self.context_length + torch.arange(self.context_length + 1)
        chunk = self.tokens[positions].long()
        return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


class BlockOrder:
    """An independent per-epoch RNG; cursor counts consumed blocks, not prefetch."""

    def __init__(self, blocks, seed, cursor=0):
        self.blocks, self.seed, self.cursor = blocks, seed, cursor
        self.epoch = -1
        self.permutation = None

    def take(self, count):
        pieces = []
        while count:
            epoch, offset = divmod(self.cursor, self.blocks)
            if epoch != self.epoch:
                generator = torch.Generator().manual_seed((self.seed + epoch) % (2**63 - 1))
                self.permutation = torch.randperm(self.blocks, generator=generator)
                self.epoch = epoch
            length = min(count, self.blocks - offset)
            pieces.append(self.permutation[offset:offset + length])
            self.cursor += length
            count -= length
        return torch.cat(pieces)


def build_model(config):
    kwargs = dict(
        vocab_size=config["vocab_size"], d_model=config["width"], num_layers=config["layers"],
        num_heads=config["heads"], context_length=config["context_length"],
        mlp_ratio=config["mlp_ratio"], dropout=config["dropout"], use_flash=True,
    )
    method = config["method"]
    if method.startswith("phase-adjoint") or method == "boundary-skip":
        kwargs["boundary"] = config["boundary"]
    if method == "block-attnres":
        kwargs["block_size"] = config["block_size"]
    try:
        from lm.phase_adjoint import create_residual_model
    except ModuleNotFoundError as error:
        if error.name != "lm.phase_adjoint":
            raise
        # The established candidate remains testable while the new registry is built.
        if method == "baseline":
            return BaselineTransformer(**kwargs)
        if method in ("adjoint", "adjoint-frozen"):
            return AdjointHCTransformer(**kwargs, freeze_aux_updates=method.endswith("frozen"))
        raise RuntimeError(f"{method} requires lm.phase_adjoint.create_residual_model") from error
    return create_residual_model(method, **kwargs)


def scale_residual_initialization(model, scale):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, CausalSelfAttention):
                module.o_proj.weight.mul_(scale)
            elif isinstance(module, MLP):
                module.proj.weight.mul_(scale)


def state_layout(model, config):
    if config["method"] == "block-attnres":
        blocks = config["layers"] // config["block_size"]
        return {"kind": "block_history", "embedding_values": 1, "completed_block_values": blocks,
                "partial_values_max": 1, "max_value_vectors_per_token": 1 + math.ceil(config["layers"] / config["block_size"]),
                "excludes": "Readout stacks, key/softmax tensors and autograd intermediates"}
    return {"kind": "residual_streams", "values_per_token": getattr(model, "n_streams", 1)}


def make_optimizer(model, config, device):
    decay, no_decay, names = [], [], {"decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = parameter.ndim < 2 or any(
            word in name.lower() for word in ("norm", "router", "gate", "gain")
        )
        group = "no_decay" if exempt else "decay"
        (no_decay if exempt else decay).append(parameter)
        names[group].append(name)
    groups = [
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups, lr=config["lr"], betas=(config["beta1"], config["beta2"]),
        fused=device.type == "cuda",
    )
    return optimizer, names


def learning_rate(update, config):
    warmup = config["warmup_updates"]
    if update < warmup:
        return config["lr"] * (update + 1) / warmup
    progress = (update - warmup) / max(1, config["updates"] - warmup - 1)
    return config["min_lr"] + (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2


def amp_settings(device, enabled):
    if device.type != "cuda" or not enabled:
        return None, None
    native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if native_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    return dtype, scaler


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def configure_numerical_policy(config, dtype):
    """Quality uses raw eager execution; compiled backward matches the call outside AMP."""
    policy = {"evaluation": "raw_eager", "evaluation_autocast_dtype": str(dtype) if dtype else None,
              "backward_call_autocast": "off", "compiled_training": config["compile"]}
    if not config["compile"]:
        return policy
    settings = (
        ("torch._inductor.config", "emulate_precision_casts", True, config["compile_backend"] == "inductor"),
        ("torch._functorch.config", "backward_pass_autocast", "off", True),
    )
    for module_name, name, requested, applicable in settings:
        module = importlib.import_module(module_name)
        supported = hasattr(module, name)
        # Older CPU-only AOT tests have no autocast and need no backward-AMP option.
        if applicable and not supported and (name != "backward_pass_autocast" or dtype is not None):
            raise RuntimeError(f"Compiled precision policy requires installed {module_name}.{name}")
        applied = applicable and supported
        if applied:
            setattr(module, name, requested)
            if getattr(module, name) != requested:
                raise RuntimeError(f"Compiled precision policy was not applied: {module_name}.{name}")
        policy[name] = {"requested": requested, "effective": getattr(module, name) if applied else None,
                        "applicable": applicable, "supported": supported, "applied": applied,
                        "config_source_sha256": digest_file(module.__file__)}
    return policy


def rng_state(device):
    state = np.random.get_state()
    return {
        "python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    array = state["numpy"]
    np.random.set_state((array[0], np.asarray(array[1], dtype=np.uint32), *array[2:]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(state, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, allow_nan=False) + "\n")
    temporary.replace(path)


def source_provenance(model):
    paths = {Path(__file__).resolve(), ROOT / "lm/models.py", ROOT / "lm/adjoint.py"}
    # Include imported backbone helpers, even for a minimal remote snapshot.
    paths.update((ROOT / "lm").glob("*.py"))
    paths.update((ROOT / "isohc").glob("*.py"))
    for cls in type(model).__mro__:
        try:
            source = inspect.getsourcefile(cls)
        except TypeError:
            source = None
        if source and Path(source).resolve().is_relative_to(ROOT):
            paths.add(Path(source).resolve())
    factory = ROOT / "lm/phase_adjoint.py"
    if factory.exists():
        paths.add(factory)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    snapshot_path = ROOT / "source_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else None
    return {"git_commit": commit, "snapshot_base_commit": snapshot.get("base_commit") if snapshot else None,
            "source_sha256": {
        str(path.relative_to(ROOT)): digest_file(path) for path in sorted(paths)
    }, "torch": str(torch.__version__), "python": sys.version}


@torch.no_grad()
def evaluate(model, data, block_count, batch_size, device, dtype=None):
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    targets = 0
    try:
        for start in range(0, block_count, batch_size):
            x, y = data.batch(torch.arange(start, min(start + batch_size, block_count)))
            valid = int((y != -100).sum())
            with autocast(device, dtype):
                logits, loss = model(x.to(device), y.to(device))
            del logits
            if loss is None:
                raise FloatingPointError("Evaluation did not return a loss")
            loss_sum += loss.detach().double() * valid
            targets += valid
        nll = float(loss_sum) / max(targets, 1)
        if not targets or not math.isfinite(nll):
            raise FloatingPointError("Evaluation has no valid finite target loss")
        perplexity = math.exp(nll) if nll < math.log(sys.float_info.max) else None
        return {"nll": nll, "perplexity": perplexity, "targets": targets, "blocks": block_count}
    finally:
        model.train(was_training)


def optimizer_update(model, optimizer, data, order, config, device, dtype, scaler):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    for _ in range(config["grad_accum"]):
        x, y = data.batch(order.take(config["micro_batch"]))
        with autocast(device, dtype):
            logits, loss = model(x.to(device), y.to(device))
        del logits
        if loss is None:
            raise FloatingPointError("Training did not return a loss")
        loss_sum += loss.detach() / config["grad_accum"]
        scaled_loss = loss / config["grad_accum"]
        (scaled_loss if scaler is None else scaler.scale(scaled_loss)).backward()
    value = float(loss_sum)
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite training loss; no update was committed")
    if scaler is not None:
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["grad_clip"], error_if_nonfinite=True,
    )
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return value, float(norm)


def profile(model, optimizer, data, config, device, dtype, scaler):
    if device.type != "cuda":
        raise ValueError("--profile-steps is a separate CUDA-only measurement")
    order = BlockOrder(data.blocks, config["data_seed"])
    model.train()
    torch.cuda.synchronize(device)
    warmup_started = time.perf_counter()
    for _ in range(config["profile_warmup"]):
        optimizer_update(model, optimizer, data, order, config, device, dtype, scaler)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(config["profile_steps"]):
        optimizer_update(model, optimizer, data, order, config, device, dtype, scaler)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    tokens = config["profile_steps"] * config["micro_batch"] * config["grad_accum"] * config["context_length"]
    return {
        "kind": "profile", "optimizer_updates": config["profile_steps"], "tokens": tokens,
        "seconds": elapsed, "steady_seconds": elapsed, "warmup_seconds": warmup_seconds,
        "warmup_scope": "Warmup optimizer steps including lazy compilation and allocator startup",
        "tokens_per_second": tokens / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "scope": "Local-cache fetch, transfer, forward, backward, clipping and AdamW; excludes eval and saves",
        "gpu": torch.cuda.get_device_name(device),
    }


def run(config):
    """Run one method. Config is plain JSON data; output_dir is this run's directory."""
    invocation_started = time.perf_counter()
    prior_seconds, timing_complete = 0.0, True
    breakdown = dict.fromkeys(("setup_seconds", "first_update_seconds", "evaluation_seconds", "save_seconds"), 0.0)
    config = dict(config)
    validate_config(config)
    if config["device"] == "cuda" and config["deterministic"] and not config["dry_run"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    train = PackedTokens(config["train_cache"], config["vocab_size"], config["context_length"], config["train_manifest"])
    val = PackedTokens(config["val_cache"], config["vocab_size"], config["context_length"], config["val_manifest"])
    if train.metadata["split"] != "train" or val.metadata["split"] not in ("val", "validation"):
        raise ValueError("Cache manifests must identify the corresponding train/validation splits")
    if train.metadata["sha256"] == val.metadata["sha256"]:
        raise ValueError("Training and validation caches must be distinct")
    random.seed(config["model_seed"])
    np.random.seed(config["model_seed"] % 2**32)
    torch.manual_seed(config["model_seed"])
    model = build_model(config)
    scale_residual_initialization(model, config["residual_init_scale"])
    provenance = source_provenance(model)
    effective_batch = config["micro_batch"] * config["grad_accum"]
    tokens_per_update = effective_batch * config["context_length"]
    val_blocks = min(config["val_blocks"], val.blocks)
    excluded = {"output_dir", "resume", "stop_after", "dry_run", "profile_steps", "profile_warmup"}
    identity = {
        "config": {key: value for key, value in config.items() if key not in excluded},
        "train": train.metadata, "val": val.metadata, "source": provenance["source_sha256"],
        "torch": provenance["torch"],
        "n_streams": None if config["method"] == "block-attnres" else getattr(model, "n_streams", 1),
        "state_layout": state_layout(model, config),
    }
    metadata = {
        "config": config, "identity": identity, "provenance": provenance,
        "parameters": sum(p.numel() for p in model.parameters()),
        "n_streams": identity["n_streams"], "state_layout": identity["state_layout"], "effective_batch": effective_batch,
        "tokens_per_update": tokens_per_update, "planned_tokens": config["updates"] * tokens_per_update,
        "planned_block_passes": config["updates"] * effective_batch / train.blocks,
        "validation_targets": val_blocks * config["context_length"],
        "budget_formula": "hours = planned_tokens / (3600 * measured_tokens_per_second); add measured setup/startup/eval/save costs",
        "natural_language_quality_established": False,
    }
    if config["dry_run"]:
        return {"kind": "dry_run", **metadata}
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA device requested, but CUDA is unavailable")
    torch.use_deterministic_algorithms(config["deterministic"] or device.type == "cpu")
    model.to(device)
    dtype, scaler = amp_settings(device, config["amp"])
    identity["amp_dtype"] = str(dtype) if dtype else None
    policy = configure_numerical_policy(config, dtype)
    identity["numerical_policy"] = provenance["numerical_policy"] = policy
    optimizer, groups = make_optimizer(model, config, device)
    metadata.update(amp_dtype=identity["amp_dtype"], numerical_policy=policy, optimizer_groups=groups,
                    parameter_dtype=str(next(model.parameters()).dtype), cuda_runtime=torch.version.cuda)
    if device.type == "cuda":
        metadata.update(gpu=torch.cuda.get_device_name(device),
                        capability=list(torch.cuda.get_device_capability(device)))
    update, cursor, best_nll = 0, 0, float("inf")
    if config["resume"]:
        saved = torch.load(config["resume"], map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Resume identity mismatch: config, source, cache, vocabulary or context changed")
        update, cursor, best_nll = saved["update"], saved["data_cursor"], saved["best_nll"]
        if cursor != update * effective_batch or saved["tokens"] != update * tokens_per_update:
            raise ValueError("Checkpoint update/token/data cursor is inconsistent")
        if update > config["updates"]:
            raise ValueError("Checkpoint exceeds the fixed update schedule")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scaler is not None:
            scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"], device)
        timing = saved.get("timing", {})
        try:
            sidecar = json.loads(Path(config["resume"] + ".timing.json").read_text())
            if not saved.get("checkpoint_tag") or sidecar["checkpoint_tag"] != saved["checkpoint_tag"]:
                raise ValueError("Timing/checkpoint tag mismatch")
            if not math.isfinite(sidecar["cumulative_elapsed_seconds"]) or sidecar["cumulative_elapsed_seconds"] < timing.get("cumulative_elapsed_seconds", 0):
                raise ValueError("Timing watermark precedes checkpoint")
            if set(sidecar["timing_breakdown_seconds"]) != set(breakdown):
                raise ValueError("Timing categories are incomplete")
            timing = sidecar
        except (OSError, ValueError, KeyError, TypeError):
            timing_complete = False
        prior_seconds = timing.get("cumulative_elapsed_seconds", 0.0)
        breakdown.update(timing.get("timing_breakdown_seconds", {}))
        timing_complete = timing_complete and timing.get("timing_complete", False)
    train_model = model
    if config["compile"]:
        train_model = torch.compile(
            model, backend=config["compile_backend"],
            mode=config["compile_mode"] if config["compile_backend"] == "inductor" else None,
        )
    output = Path(config["output_dir"])
    if not config["resume"] and any((output / name).exists() for name in ("final.pt", "train.jsonl", "summary.json")):
        raise FileExistsError("Output already contains a run; choose another output directory or resume")
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    breakdown["setup_seconds"] += time.perf_counter() - invocation_started
    metadata["timing_scope"] = (
        "Active run() wall time; excludes process import, gaps between invocations and terminal sidecar/log/summary "
        "writes after the saved watermark. Incomplete clocks are lower bounds, unsuitable for complete cost curves. "
        "Setup includes compile-wrapper construction; first update includes any lazy compilation."
    )
    def clock():
        elapsed = time.perf_counter() - invocation_started
        return {"elapsed_this_invocation": elapsed, "cumulative_elapsed_seconds": prior_seconds + elapsed,
                "timing_complete": timing_complete}
    if config["profile_steps"]:
        summary = {**metadata, **profile(train_model, optimizer, train, config, device, dtype, scaler),
                   **clock(), "timing_breakdown_seconds": breakdown}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
    order = BlockOrder(train.blocks, config["data_seed"], cursor)
    stop = config["updates"] if config["stop_after"] is None else config["stop_after"]
    if stop < update:
        raise ValueError("--stop-after precedes the checkpoint update")

    def save():
        save_started, tag = time.perf_counter(), str(time.time_ns())
        atomic_save({
            "format_version": 1, "identity": identity, "config": config, "provenance": provenance,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
            "tokens": update * tokens_per_update, "data_cursor": order.cursor,
            "best_nll": best_nll, "rng": rng_state(device),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "checkpoint_tag": tag, "timing": {**clock(), "timing_breakdown_seconds": dict(breakdown)},
        }, output / "final.pt")
        breakdown["save_seconds"] += time.perf_counter() - save_started
        timing = {"checkpoint_tag": tag, **clock(), "timing_breakdown_seconds": dict(breakdown)}
        atomic_json(timing, output / "final.pt.timing.json")
        return timing

    def timed_evaluate():
        evaluation_started = time.perf_counter()
        metrics = evaluate(model, val, val_blocks, config["micro_batch"], device, dtype)
        breakdown["evaluation_seconds"] += time.perf_counter() - evaluation_started
        return {**metrics, **clock()}

    started = time.perf_counter()
    initial_update = update
    train_model.train()
    with (output / "train.jsonl").open("a" if config["resume"] else "w") as log:
        log.write(json.dumps({"kind": "start", "update": update, **metadata, **clock()}, allow_nan=False) + "\n")
        while update < stop:
            lr = learning_rate(update, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            update_started = time.perf_counter()
            loss, norm = optimizer_update(train_model, optimizer, train, order, config, device, dtype, scaler)
            if update == initial_update:
                breakdown["first_update_seconds"] += time.perf_counter() - update_started
            update += 1
            record = {"kind": "update", "update": update, "tokens": update * tokens_per_update,
                      "data_cursor": order.cursor, "loss": loss, "grad_norm": norm, "lr": lr}
            if update % config["eval_every"] == 0:
                record["validation"] = timed_evaluate()
                best_nll = min(best_nll, record["validation"]["nll"])
            record.update(clock())
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if config["checkpoint_every"] and update % config["checkpoint_every"] == 0:
                save()
        train_seconds = time.perf_counter() - started
        final_eval = timed_evaluate()
        best_nll = min(best_nll, final_eval["nll"])
        final_timing = save()
        summary = {
            **metadata, "kind": "training", "status": "complete" if update == config["updates"] else "stopped",
            "update": update, "tokens": update * tokens_per_update, "data_cursor": order.cursor,
            "updates_this_invocation": update - initial_update, "train_seconds": train_seconds,
            "tokens_per_second_including_periodic_eval_and_saves":
                (update - initial_update) * tokens_per_update / max(train_seconds, 1e-12),
            "final_validation": final_eval, "best_nll": best_nll, **final_timing,
        }
        log.write(json.dumps({**summary, "kind": "end"}, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def validate_config(config):
    positive = ("vocab_size", "context_length", "layers", "width", "heads", "micro_batch",
                "grad_accum", "updates", "val_blocks", "eval_every")
    if any(config[key] < 1 for key in positive) or config["vocab_size"] < 2:
        raise ValueError("Vocabulary >=2 and positive dimensions/budgets are required")
    if config["width"] % config["heads"] or config["method"] not in METHODS:
        raise ValueError("Invalid head width or method")
    if not 0 <= config["warmup_updates"] < config["updates"]:
        raise ValueError("Warmup must be smaller than the fixed update schedule")
    if not 0 <= config["min_lr"] <= config["lr"] or config["lr"] <= 0:
        raise ValueError("Require 0 <= min_lr <= lr and lr > 0")
    if config["stop_after"] is not None and not 0 <= config["stop_after"] <= config["updates"]:
        raise ValueError("--stop-after is an absolute update within the fixed schedule")
    if config["profile_steps"] < 0 or config["profile_warmup"] < 1 or config["checkpoint_every"] < 0:
        raise ValueError("Invalid profiling/checkpoint frequency")
    if config["profile_steps"] and (config["device"] != "cuda" or config["resume"]):
        raise ValueError("Profiling is CUDA-only and cannot resume a scientific training run")
    if config["device"] not in ("cpu", "cuda") or config["grad_clip"] <= 0:
        raise ValueError("Explicit cpu/cuda device and positive gradient clipping are required")
    if not math.isfinite(config["residual_init_scale"]) or config["residual_init_scale"] <= 0:
        raise ValueError("Residual initialization scale must be finite and positive")
    floats = ("lr", "min_lr", "weight_decay", "grad_clip", "mlp_ratio", "dropout", "beta1", "beta2")
    if any(not math.isfinite(config[key]) for key in floats):
        raise ValueError("Training hyperparameters must be finite")
    if config["weight_decay"] < 0 or config["mlp_ratio"] <= 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("Invalid weight decay, MLP ratio or dropout")
    if any(not 0 <= config[key] < 1 for key in ("beta1", "beta2")) or config["block_size"] < 1:
        raise ValueError("Invalid Adam betas or attention-residual block size")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=[
        "baseline", "gain", "phase-adjoint", "terminal-adjoint", "boundary-skip", "phase-adjoint-post-frozen", "block-attnres",
    ])
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    for name, default in (("layers", 24), ("width", 256), ("heads", 4), ("micro-batch", 8),
                          ("grad-accum", 1), ("updates", 1000), ("val-blocks", 128),
                          ("eval-every", 100), ("checkpoint-every", 100), ("model-seed", 0),
                          ("data-seed", 1729), ("block-size", 4), ("profile-steps", 0), ("profile-warmup", 3)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("mlp-ratio", 4.), ("dropout", 0.), ("lr", 3e-4), ("min-lr", 3e-5),
                          ("weight-decay", .1), ("beta1", .9), ("beta2", .95), ("grad-clip", 1.)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--residual-init-scale", type=float)
    parser.add_argument("--boundary", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", default="outputs/residual_lm_diagnostic")
    parser.add_argument("--resume")
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend", choices=("inductor", "aot_eager"), default="inductor")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args(argv))
    if args["warmup_updates"] is None:
        args["warmup_updates"] = min(100, args["updates"] // 10)
    if args["boundary"] is None:
        args["boundary"] = args["layers"] // 2
    if args["residual_init_scale"] is None:
        args["residual_init_scale"] = 1 / math.sqrt(2 * args["layers"])
    if args["resume"] and len(args["methods"]) != 1:
        parser.error("--resume requires exactly one method")
    return args


def main(argv=None):
    args = parse_args(argv)
    methods = args.pop("methods")
    for method in methods:
        config = dict(args, method=method)
        config["output_dir"] = str(Path(args["output_dir"]) / f"{method}_seed{args['model_seed']}")
        print(json.dumps(run(config), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
