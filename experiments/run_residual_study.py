#!/usr/bin/env python3
"""Sequential seven-method GPU study; planning never imports the runner or tokens.

First use --dry-run to write/review output-root/plan.json. The identical command
with --execute-reviewed-plan starts jobs; the existing plan must match exactly.
--profile-rates is a JSON object mapping every method to measured tokens/second.
Stage estimates add --stage-overhead-seconds for setup, validation and saves.
With --compile, --compile-cache-root must reuse the reviewed per-method caches
from profiling; both LRs and continuation stages use cache-root/<method>.
Reserve is BEFORE the explicit UTC deadline: now + estimate + reserve <= deadline.
The controller never terminates a running stage at the deadline.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

METHODS = (
    "baseline", "gain", "phase-adjoint", "terminal-adjoint", "boundary-skip",
    "phase-adjoint-post-frozen", "block-attnres",
)
LEARNING_RATES = (3e-4, 6e-4)
PILOT_UPDATES = 2048
PROTOCOL = {
    "vocab_size": 50257, "context_length": 512, "layers": 24, "width": 256,
    "heads": 4, "micro_batch": 32, "grad_accum": 1, "updates": 16384,
    "warmup_updates": 256, "model_seed": 419, "data_seed": 20260929,
    "val_blocks": 3906, "eval_every": 2048, "checkpoint_every": 2048,
    "boundary": 12, "block_size": 4, "mlp_ratio": 4.0, "dropout": 0.0,
    "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
    "residual_init_scale": 1 / math.sqrt(48), "device": "cuda",
}


class SchedulingStopped(RuntimeError):
    """A launched child's exit is unconfirmed; scheduling must not continue."""


def utc_now():
    return datetime.now(timezone.utc)


def parse_utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() != timedelta(0):
        raise ValueError("Deadline must explicitly include UTC (+00:00 or Z)")
    return result


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def controller_lock(output_root):
    """OS lock survives stale PID text; children inherit it until their exit."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "controller.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another controller or its active child holds this output-root lock") from error
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc_now().isoformat()}, handle)
        handle.flush()
        # This fd closes with the context; an inheriting active child retains it.
        yield handle.fileno()


def select_learning_rate(results):
    """Failed/nonfinite trials are ineligible; exact ties choose 3e-4."""
    eligible = []
    for lr in LEARNING_RATES:
        result = results.get(lr, {})
        nll = result.get("nll")
        if (result.get("status") == "completed" and not isinstance(nll, bool)
                and isinstance(nll, (int, float)) and math.isfinite(nll) and nll >= 0):
            eligible.append((nll, lr))
    return min(eligible)[1] if eligible else None


def can_start_stage(now, deadline, estimate_seconds, reserve_seconds):
    values = (estimate_seconds, reserve_seconds)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Stage estimate and reserve must be finite and non-negative")
    return now + timedelta(seconds=sum(values)) <= deadline


def arm_root(plan, lr):
    return Path(plan["paths"]["output_root"]) / ("lr3e-4" if lr == LEARNING_RATES[0] else "lr6e-4")


def arm_directory(plan, method, lr):
    return arm_root(plan, lr) / f"{method}_seed{PROTOCOL['model_seed']}"


def stage_command(plan, method, lr, stop_after, resume=False):
    paths = plan["paths"]
    command = [paths["python"], str(Path(paths["repo_root"]) / "experiments/residual_lm_diagnostic.py"),
               "--methods", method, "--train-cache", paths["train_cache"], "--val-cache", paths["val_cache"],
               "--train-manifest", paths["data_manifest"], "--val-manifest", paths["data_manifest"]]
    for key, value in plan["protocol"].items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    command.extend(["--lr", str(lr), "--min-lr", str(lr / 10), "--stop-after", str(stop_after),
                    "--output-dir", str(arm_root(plan, lr)), "--compile-backend", "inductor",
                    "--compile-mode", "default"])
    if plan["compile"]:
        command.append("--compile")
    if resume:
        command.extend(["--resume", str(arm_directory(plan, method, lr) / "final.pt")])
    return command


def stage_environment(plan, method):
    overrides = {"PYTHONUNBUFFERED": "1"}
    if plan["compile"]:
        overrides["TORCHINDUCTOR_CACHE_DIR"] = str(Path(plan["compile_cache_root"]) / method)
    return overrides


def expected_numerical_policy(compiled):
    policy = {"evaluation": "raw_eager", "backward_call_autocast": "off", "compiled_training": compiled,
              "evaluation_autocast_dtype": "must match identity.amp_dtype (torch.bfloat16 or torch.float16)"}
    if compiled:
        for name, value in (("emulate_precision_casts", True), ("backward_pass_autocast", "off")):
            policy[name] = {"requested": value, "effective": value, "applicable": True,
                            "supported": True, "applied": True}
    return policy


def validate_numerical_policy(expected, identity):
    actual = identity["numerical_policy"]
    if (actual.get("evaluation_autocast_dtype") != identity.get("amp_dtype")
            or identity.get("amp_dtype") not in ("torch.bfloat16", "torch.float16")):
        raise ValueError("Evaluation autocast must match the declared CUDA training dtype")
    for name, value in expected.items():
        if name == "evaluation_autocast_dtype":
            continue
        applied = actual.get(name)
        if isinstance(value, dict):
            matches = isinstance(applied, dict) and all(applied.get(key) == v for key, v in value.items())
        else:
            matches = applied == value
        if not matches:
            raise ValueError(f"Runner numerical policy differs from the locked plan: {name}")


def build_plan(args):
    paths = {key: str(Path(getattr(args, key)).expanduser().resolve()) for key in
             ("repo_root", "output_root", "train_cache", "val_cache", "data_manifest")}
    paths["python"] = args.python
    if args.compile and args.compile_cache_root is None:
        raise ValueError("--compile requires --compile-cache-root for reviewed profile caches")
    rates = json.loads(args.profile_rates.read_text())
    if set(rates) != set(METHODS) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0 for value in rates.values()
    ):
        raise ValueError("Profile JSON must provide a finite positive tokens/second rate for all seven methods")
    manifest = json.loads(Path(paths["data_manifest"]).read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Reviewed data manifest must be complete")
    cache_hashes = {}
    for split in ("train", "val"):
        entry = manifest["outputs"][split]
        if Path(entry["path"]).name != Path(paths[f"{split}_cache"]).name:
            raise ValueError(f"{split} cache name does not match the reviewed manifest")
        cache_hashes[split] = entry["file_sha256"]
    repo = Path(paths["repo_root"])
    source = {"controller": digest_file(__file__)}
    source_paths = [repo / "experiments/residual_lm_diagnostic.py"]
    source_paths += sorted((repo / "lm").glob("*.py")) + sorted((repo / "isohc").glob("*.py"))
    source.update({str(path.relative_to(repo)): digest_file(path) for path in source_paths})
    return {
        "format_version": 1, "paths": paths, "protocol": PROTOCOL, "compile": args.compile,
        "compile_cache_root": str(args.compile_cache_root.expanduser().resolve()) if args.compile_cache_root else None,
        "amp": True, "deterministic": False, "compile_backend": "inductor", "compile_mode": "default",
        "min_lr_ratio": 0.1,
        "numerical_policy": expected_numerical_policy(args.compile),
        "pilot_updates": PILOT_UPDATES, "learning_rates": list(LEARNING_RATES),
        "pilot_order": [{"method": method, "lr": lr} for method in METHODS for lr in LEARNING_RATES],
        "primary_order": list(METHODS), "matched_phase_lr": args.matched_phase_lr,
        "matched_order": ["phase-adjoint-post-frozen", "boundary-skip"],
        "selection_policy": "Lowest finite successful pilot final NLL; exact ties use 3e-4; failed trials ineligible",
        "selection_freeze": "Any primary start freezes all selections; restarts never retry tuning afterward",
        "deadline_utc": parse_utc(args.deadline_utc).isoformat(), "reserve_seconds": args.reserve_seconds,
        "reserve_semantics": "Before deadline; never extend deadline or terminate an in-flight stage",
        "stage_overhead_seconds": args.stage_overhead_seconds, "profile_rates": rates,
        "estimate_formula": "remaining_updates * 32 * 512 / method_tokens_per_second + stage_overhead_seconds",
        "profiles_sha256": digest_file(args.profile_rates), "source_sha256": source,
        "data_manifest_sha256": digest_file(paths["data_manifest"]), "cache_sha256": cache_hashes,
        "wall_cost_semantics": "Child monotonic wall includes imports, setup, train, evaluation, saves and exit; "
                               "tuning includes both pilots; selected trajectory includes its pilot plus continuation; "
                               "these two views overlap. Profiles, preparation and inter-stage gaps are excluded.",
    }


def validate_summary(plan, method, lr, stop_after, directory):
    summary = json.loads((directory / "summary.json").read_text())
    expected_status = "complete" if stop_after == PROTOCOL["updates"] else "stopped"
    if (summary.get("kind") != "training" or summary.get("status") != expected_status
            or summary.get("update") != stop_after or not (directory / "final.pt").is_file()):
        raise ValueError("Runner did not finish the requested stage with its checkpoint")
    config = summary["config"]
    expected = {**plan["protocol"], "method": method, "lr": lr, "min_lr": lr / 10,
                "compile": plan["compile"], "train_cache": plan["paths"]["train_cache"],
                "val_cache": plan["paths"]["val_cache"], "amp": plan["amp"],
                "deterministic": plan["deterministic"], "compile_backend": plan["compile_backend"],
                "compile_mode": plan["compile_mode"]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Runner config does not match the locked study protocol")
    identity = summary["identity"]
    validate_numerical_policy(plan["numerical_policy"], identity)
    for split in ("train", "val"):
        if (identity[split]["sha256"] != plan["cache_sha256"][split]
                or identity[split]["manifest_sha256"] != plan["data_manifest_sha256"]):
            raise ValueError("Runner data provenance differs from the reviewed plan")
    for path, sha in plan["source_sha256"].items():
        if path != "controller" and summary["provenance"]["source_sha256"].get(path) != sha:
            raise ValueError(f"Runner source differs from the reviewed plan: {path}")
    final = summary["final_validation"]
    nll = final["nll"]
    if (isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0
            or final.get("blocks") != PROTOCOL["val_blocks"]
            or final.get("targets") != PROTOCOL["val_blocks"] * PROTOCOL["context_length"]):
        raise ValueError("Runner final validation is nonfinite or has the wrong evaluation scope")
    return nll


def execute_stage(plan, state, stage_id, method, lr, stop_after, initial_update, lock_fd):
    root = Path(plan["paths"]["output_root"])
    previous = state["stages"].get(stage_id, {})
    if previous.get("status") == "completed":
        expected = {"method": method, "lr": lr, "stop_after": stop_after}
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Completed stage identity mismatch: {stage_id}")
        return previous
    estimate = ((stop_after - initial_update) * PROTOCOL["micro_batch"] * PROTOCOL["context_length"]
                / plan["profile_rates"][method] + plan["stage_overhead_seconds"])
    record = {"method": method, "lr": lr, "stop_after": stop_after, "estimate_seconds": estimate,
              "attempt": previous.get("attempt", 0) + 1, "created_utc": utc_now().isoformat(),
              "prior_attempt_wall_seconds": previous.get("cumulative_child_wall_seconds", 0),
              "child_wall_seconds": 0.0, "child_wall_complete": True,
              "cumulative_child_wall_seconds": previous.get("cumulative_child_wall_seconds", 0),
              "cumulative_wall_complete": previous.get("cumulative_wall_complete", True)
                                          and previous.get("status") != "running"}
    if not can_start_stage(utc_now(), parse_utc(plan["deadline_utc"]), estimate, plan["reserve_seconds"]):
        record["status"] = "skipped_deadline"
    else:
        directory = arm_directory(plan, method, lr)
        record["command"] = stage_command(plan, method, lr, stop_after, (directory / "final.pt").is_file())
        log_path = root / "logs" / f"{stage_id}_attempt{record['attempt']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record.update(status="running", log_path=str(log_path), started_utc=utc_now().isoformat())
        record["environment_overrides"] = stage_environment(plan, method)
        state["stages"][stage_id] = record
        if stage_id.startswith("primary_"):
            state["selections_frozen"] = True
        atomic_json(root / "status.json", state)
        print(shlex.join(record["command"]), flush=True)
        child_started = None
        child = None
        child_finished = False
        try:
            with log_path.open("w") as log:
                child_started = time.monotonic()
                child = subprocess.Popen(record["command"], cwd=plan["paths"]["repo_root"],
                                         stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, **record["environment_overrides"]}, pass_fds=(lock_fd,))
                record["child_pid"] = child.pid
                atomic_json(root / "status.json", state)
                record["returncode"] = child.wait()
                child_finished = True
                record["child_wall_seconds"] = time.monotonic() - child_started
                record["cumulative_child_wall_seconds"] += record["child_wall_seconds"]
            if record["returncode"] != 0:
                raise RuntimeError(f"Runner exited with code {record['returncode']}")
            record["nll"] = validate_summary(plan, method, lr, stop_after, directory)
            record.update(status="completed", checkpoint=str(directory / "final.pt"),
                          summary_sha256=digest_file(directory / "summary.json"))
            atomic_json(root / "records" / f"{stage_id}.json", json.loads((directory / "summary.json").read_text()))
        except KeyboardInterrupt:
            if child_started is not None and not child_finished:
                record["child_wall_seconds"] = time.monotonic() - child_started
                record["cumulative_child_wall_seconds"] += record["child_wall_seconds"]
                record.update(child_wall_complete=False, cumulative_wall_complete=False)
            record.update(status="interrupted", finished_utc=utc_now().isoformat())
            state.update(status="partial", wall_costs=wall_costs(state))
            atomic_json(root / "status.json", state)
            raise
        except Exception as error:
            if child is not None and not child_finished:
                record["child_wall_seconds"] = time.monotonic() - child_started
                record["cumulative_child_wall_seconds"] += record["child_wall_seconds"]
                record.update(child_wall_complete=False, cumulative_wall_complete=False)
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            if child is not None and not child_finished:
                record.update(child_exit_unconfirmed=True, finished_utc=utc_now().isoformat())
                state.update(status="partial", wall_costs=wall_costs(state))
                try:
                    atomic_json(root / "status.json", state)
                except Exception as journal_error:
                    print(f"Could not persist failure status: {journal_error}", file=sys.stderr)
                raise SchedulingStopped(
                    f"Child {record.get('child_pid', 'unknown')} exit is unconfirmed; controller scheduling stopped"
                ) from error
        record["finished_utc"] = utc_now().isoformat()
    state["stages"][stage_id] = record
    atomic_json(root / "status.json", state)
    return record


def wall_costs(state):
    stages = state["stages"]
    spent = lambda record: record.get("cumulative_child_wall_seconds", 0.0)
    selected = {}
    for method, selection in state.get("selections", {}).items():
        lr = selection["selected_lr"]
        selected[method] = None if lr is None else (
            spent(stages.get(f"pilot_{method}_{lr_tag(lr)}", {})) + spent(stages.get(f"primary_{method}", {}))
        )
    return {"tuning_child_wall_seconds": sum(spent(r) for key, r in stages.items() if key.startswith("pilot_")),
            "primary_continuation_child_wall_seconds": sum(spent(r) for key, r in stages.items() if key.startswith("primary_")),
            "matched_continuation_child_wall_seconds": sum(spent(r) for key, r in stages.items() if key.startswith("matched_")),
            "total_child_wall_seconds": sum(spent(r) for r in stages.values()),
            "selected_trajectory_child_wall_seconds": selected,
            "timing_complete": all(r.get("cumulative_wall_complete", True) for r in stages.values())}


def lr_tag(lr):
    return "3e-4" if lr == LEARNING_RATES[0] else "6e-4"


def run_study(plan, lock_fd):
    root = Path(plan["paths"]["output_root"])
    status_path = root / "status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {
        "plan_sha256": json_digest(plan), "status": "running", "stages": {}, "selections": {},
    }
    if state["plan_sha256"] != json_digest(plan):
        raise ValueError("Existing controller state belongs to a different plan")
    state["status"] = "running"
    frozen = state.get("selections_frozen", False) or any(
        key.startswith("primary_") and (record.get("started_utc") is not None
            or record.get("status") in ("running", "completed", "failed", "interrupted"))
        for key, record in state["stages"].items()
    )
    if frozen:
        state["selections_frozen"] = True
        stored = json.loads((root / "selection.json").read_text())
        if (set(state["selections"]) != set(METHODS) or stored != {
            "plan_sha256": state["plan_sha256"], "methods": state["selections"]
        }):
            raise ValueError("Frozen selections are missing or inconsistent; refusing to reselect")
    else:
        for arm in plan["pilot_order"]:
            method, lr = arm["method"], arm["lr"]
            execute_stage(plan, state, f"pilot_{method}_{lr_tag(lr)}", method, lr, PILOT_UPDATES, 0, lock_fd)
        for method in METHODS:
            candidates = {lr: state["stages"][f"pilot_{method}_{lr_tag(lr)}"] for lr in LEARNING_RATES}
            selected = select_learning_rate(candidates)
            state["selections"][method] = {
                "selected_lr": selected, "comparison_complete": all(r["status"] == "completed" for r in candidates.values()),
                "candidates": {lr_tag(lr): result for lr, result in candidates.items()},
            }
        atomic_json(root / "selection.json", {"plan_sha256": state["plan_sha256"], "methods": state["selections"]})
    atomic_json(status_path, state)
    for method in plan["primary_order"]:
        lr = state["selections"][method]["selected_lr"]
        if lr is not None:
            execute_stage(plan, state, f"primary_{method}", method, lr, PROTOCOL["updates"], PILOT_UPDATES, lock_fd)
        else:
            state["stages"][f"primary_{method}"] = {"status": "skipped_ineligible_pilot", "method": method}
    primaries_complete = all(state["stages"].get(f"primary_{method}", {}).get("status") == "completed" for method in METHODS)
    if plan["matched_phase_lr"] and primaries_complete:
        phase_lr = state["selections"]["phase-adjoint"]["selected_lr"]
        for method in plan["matched_order"]:
            if phase_lr != state["selections"][method]["selected_lr"]:
                pilot = state["stages"][f"pilot_{method}_{lr_tag(phase_lr)}"]
                if pilot["status"] == "completed":
                    execute_stage(plan, state, f"matched_{method}", method, phase_lr,
                                  PROTOCOL["updates"], PILOT_UPDATES, lock_fd)
                else:
                    state["stages"][f"matched_{method}"] = {"status": "skipped_ineligible_pilot"}
    state["status"] = "complete" if primaries_complete and all(
        record["status"] == "completed" for record in state["stages"].values()
    ) else "partial"
    state["finished_utc"] = utc_now().isoformat()
    state["wall_costs"] = wall_costs(state)
    atomic_json(status_path, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("output-root", "train-cache", "val-cache", "data-manifest", "profile-rates"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--reserve-seconds", type=float, default=1200)
    parser.add_argument("--stage-overhead-seconds", type=float, default=600)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-cache-root", type=Path)
    parser.add_argument("--matched-phase-lr", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args(argv)
    can_start_stage(utc_now(), parse_utc(args.deadline_utc), args.stage_overhead_seconds, args.reserve_seconds)
    plan = build_plan(args)
    with controller_lock(plan["paths"]["output_root"]) as lock_fd:
        path = Path(plan["paths"]["output_root"]) / "plan.json"
        if path.exists():
            if json.loads(path.read_text()) != plan:
                raise ValueError("Locked plan differs from these arguments/source/data/profile hashes")
        elif args.execute_reviewed_plan:
            raise FileNotFoundError("Generate and review plan.json with --dry-run before executing")
        else:
            atomic_json(path, plan)
        if not args.execute_reviewed_plan:
            print(json.dumps(plan, indent=2, allow_nan=False))
            for arm in plan["pilot_order"]:
                print(shlex.join(stage_command(plan, arm["method"], arm["lr"], PILOT_UPDATES)))
            return 0
        try:
            state = run_study(plan, lock_fd)
        except SchedulingStopped as error:
            print(str(error), file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("Controller interrupted; existing checkpoints and status are retained.", file=sys.stderr)
            return 2
        print(json.dumps({"status": state["status"], "status_path": str(args.output_root / "status.json")}, indent=2))
        return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
