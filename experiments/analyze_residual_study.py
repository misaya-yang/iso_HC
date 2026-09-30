#!/usr/bin/env python3
"""Read-only, stdlib-only analysis of a locked residual-study snapshot.

Quality endpoints are final checkpoint NLL at fixed D, never best_nll. Missing
or unfinished stages remain visible but cannot enter equal-budget comparisons.
No runner import, tensor load, training, statistical test, or plotting occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

METHODS = ("baseline", "gain", "phase-adjoint", "terminal-adjoint", "boundary-skip",
           "phase-adjoint-post-frozen", "block-attnres")
RATES = (3e-4, 6e-4)
TOKENS = 268435456
TARGETS = 1999872
PRIORITY_NATS = .02
EXCLUDED_CONFIG = {"output_dir", "resume", "stop_after", "dry_run", "profile_steps", "profile_warmup"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def matches(actual, expected):
    return actual == expected and isinstance(actual, bool) == isinstance(expected, bool)


def as_dict(value):
    return value if isinstance(value, dict) else {}


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON literal: {value}")


def lr_tag(lr):
    return "3e-4" if lr == RATES[0] else "6e-4"


def read_json(path, inputs, issues):
    try:
        content = path.read_bytes()
        inputs[str(path)] = digest(content)
        value = json.loads(content, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issues.append(f"{path.name}: {type(error).__name__}: {error}")
        return {}


def validate_plan(plan):
    p = plan.get("protocol", {})
    errors = []
    if not isinstance(p, dict):
        return ["Missing protocol object"]
    checks = {
        "updates": 16384, "micro_batch": 32, "grad_accum": 1, "context_length": 512,
        "val_blocks": 3906, "model_seed": 419, "device": "cuda",
    }
    for key, value in checks.items():
        if not matches(p.get(key), value):
            errors.append(f"Locked protocol requires {key}={value}")
    if plan.get("primary_order") != list(METHODS) or plan.get("learning_rates") != list(RATES):
        errors.append("Locked seven-method/two-LR grid is missing or different")
    if plan.get("pilot_updates") != 2048:
        errors.append("Locked pilot budget must be 2048 updates")
    if as_dict(plan.get("numerical_policy")).get("evaluation") != "raw_eager":
        errors.append("Locked evaluator must be raw_eager")
    if not isinstance(plan.get("compile"), bool) or plan.get("amp") is not True:
        errors.append("Locked CUDA compile/AMP policy is missing")
    if plan.get("compile"):
        policy = as_dict(plan.get("numerical_policy"))
        for key, value in (("emulate_precision_casts", True), ("backward_pass_autocast", "off")):
            if not matches(as_dict(policy.get(key)).get("effective"), value):
                errors.append(f"Locked compiled precision setting missing: {key}")
    for key in ("paths", "source_sha256", "cache_sha256"):
        if not isinstance(plan.get(key), dict) or not plan[key]:
            errors.append(f"Missing locked {key} object")
    if not as_dict(plan.get("source_sha256")).get("experiments/residual_lm_diagnostic.py"):
        errors.append("Locked runner source SHA is missing")
    if not plan.get("data_manifest_sha256"):
        errors.append("Locked data manifest SHA is missing")
    return errors


def validate_identity(summary, plan, method, lr):
    """Return a shared quality signature; method and grid-selected LR may differ."""
    errors = []
    config, identity, provenance = (summary.get(key, {}) for key in ("config", "identity", "provenance"))
    if not all(isinstance(value, dict) for value in (config, identity, provenance)):
        return ["Missing config/identity/provenance object"], None
    paths = plan["paths"]
    expected = {**plan["protocol"], "method": method, "lr": lr,
                "min_lr": lr * plan.get("min_lr_ratio", .1) if lr in RATES else None,
                "train_cache": paths.get("train_cache"), "val_cache": paths.get("val_cache"),
                "train_manifest": paths.get("data_manifest"), "val_manifest": paths.get("data_manifest")}
    expected.update({key: plan.get(key) for key in
                     ("compile", "compile_backend", "compile_mode", "amp", "deterministic")})
    for key, value in expected.items():
        if not matches(config.get(key), value):
            errors.append(f"Config mismatch: {key}")
    if config.get("dry_run") is not False or config.get("profile_steps") != 0:
        errors.append("A dry-run/profile summary is not a training endpoint")
    stable_config = {key: value for key, value in config.items() if key not in EXCLUDED_CONFIG}
    if identity.get("config") != stable_config:
        errors.append("Checkpoint identity/config disagreement")
    source = {key: value for key, value in plan["source_sha256"].items() if key != "controller"}
    for location in (identity.get("source", {}), provenance.get("source_sha256", {})):
        if location != source:
            errors.append("Source SHA differs from locked plan")
    for split in ("train", "val"):
        data = identity.get(split, {})
        if (not isinstance(data, dict) or data.get("sha256") != plan["cache_sha256"].get(split)
                or data.get("manifest_sha256") != plan["data_manifest_sha256"]
                or data.get("split") not in (("train",) if split == "train" else ("val", "validation"))):
            errors.append(f"Cache identity mismatch: {split}")
    policy = identity.get("numerical_policy", {})
    dtype = identity.get("amp_dtype")
    if (not isinstance(policy, dict) or dtype not in ("torch.bfloat16", "torch.float16")
            or policy.get("evaluation_autocast_dtype") != dtype):
        errors.append("Evaluator AMP dtype mismatch")
    for key, value in plan["numerical_policy"].items():
        if key == "evaluation_autocast_dtype":
            continue
        actual = policy.get(key) if isinstance(policy, dict) else None
        good = (isinstance(actual, dict) and all(matches(actual.get(k), v) for k, v in value.items())
                if isinstance(value, dict) else matches(actual, value))
        if not good:
            errors.append(f"Numerical policy mismatch: {key}")
    if summary.get("numerical_policy") != policy or provenance.get("numerical_policy") != policy:
        errors.append("Numerical policy metadata disagreement")
    if not identity.get("torch") or identity.get("torch") != provenance.get("torch"):
        errors.append("Torch identity/provenance disagreement")
    if errors:
        return errors, None
    shared_config = {key: value for key, value in stable_config.items() if key not in ("method", "lr", "min_lr")}
    signature = json_digest({"config": shared_config, "source": source, "torch": identity["torch"],
                             "amp_dtype": dtype, "numerical_policy": policy,
                             "train": identity["train"], "val": identity["val"]})
    return [], signature


def validate_validation(value, blocks):
    if not isinstance(value, dict):
        return ["Missing validation object"]
    if not finite(value.get("nll")):
        return ["Final/checkpoint NLL must be finite and nonnegative"]
    if value.get("targets") != TARGETS or value.get("blocks") != blocks:
        return ["Validation scope is not 1999872 targets / 3906 blocks"]
    return []


def audit_record(root, plan, stage_id, method, lr, stop, stages, inputs):
    issues = []
    summary = read_json(root / "records" / f"{stage_id}.json", inputs, issues)
    stage = stages.get(stage_id, {})
    tokens_per_update = 32 * 512
    errors, signature = validate_identity(summary, plan, method, lr) if summary else ([], None)
    issues.extend(errors)
    expected_status = "complete" if stop == 16384 else "stopped"
    expected = {"kind": "training", "status": expected_status, "update": stop,
                "tokens": stop * tokens_per_update, "data_cursor": stop * 32,
                "planned_tokens": TOKENS, "validation_targets": TARGETS}
    if summary:
        for key, value in expected.items():
            if not matches(summary.get(key), value):
                issues.append(f"Stage endpoint mismatch: {key}")
        issues.extend(validate_validation(summary.get("final_validation"), 3906))
    if stage.get("status") != "completed":
        issues.append(f"Driver stage is {stage.get('status', 'missing')}, not completed")
    if summary:
        for key, value in {"method": method, "lr": lr, "stop_after": stop}.items():
            if not matches(stage.get(key), value):
                issues.append(f"Driver stage identity mismatch: {key}")
        nll = as_dict(summary.get("final_validation")).get("nll")
        if not matches(stage.get("nll"), nll):
            issues.append("Driver/final checkpoint NLL disagreement")
    else:
        nll = None
    eligible = bool(summary) and not issues
    if eligible:
        classification = "completed"
    elif stage.get("status") in ("failed", "interrupted") or (summary and stage.get("status") == "completed"):
        classification = "ineligible"
    elif summary:
        classification = "partial"
    else:
        classification = "unfinished"
    return {"stage_id": stage_id, "method": method, "lr": lr if lr in RATES else None, "classification": classification,
            "eligible": eligible, "driver_status": stage.get("status", "missing"), "issues": issues,
            "update": summary.get("update") if finite(summary.get("update")) else None,
            "tokens": summary.get("tokens") if finite(summary.get("tokens")) else None,
            "observed_final_checkpoint_nll": nll if finite(nll) else None,
            "final_checkpoint_nll": nll if eligible else None, "quality_signature": signature}, summary


def wall_ledger(stages, selections, summaries):
    costs, issues = {}, []
    for name, stage in stages.items():
        value = stage.get("cumulative_child_wall_seconds")
        skipped = str(stage.get("status", "")).startswith("skipped_")
        if value is None and skipped and not stage.get("started_utc"):
            value = 0.0
        known = finite(value)
        if not known:
            issues.append(f"Missing/nonfinite cumulative child wall: {name}")
        unstarted_zero = skipped and not stage.get("started_utc") and value == 0 and "cumulative_wall_complete" not in stage
        complete = known and (stage.get("cumulative_wall_complete") is True or unstarted_zero)
        complete = complete and stage.get("status") != "running" and not stage.get("child_exit_unconfirmed", False)
        costs[name] = {"seconds": value if known else 0.0, "measurement_complete": complete}
    groups = {prefix: sum(value["seconds"] for name, value in costs.items() if name.startswith(prefix + "_"))
              for prefix in ("pilot", "primary", "matched")}
    selected, runner_active = {}, {}
    for method, selection in selections.items():
        lr = selection.get("selected_lr")
        if lr not in RATES or isinstance(lr, bool):
            continue
        ids = (f"pilot_{method}_{lr_tag(lr)}", f"primary_{method}")
        selected[method] = {
            "seconds": sum(costs.get(name, {}).get("seconds", 0.0) for name in ids),
            "measurement_complete": all(costs.get(name, {}).get("measurement_complete", False) for name in ids),
            "trajectory_complete": stages.get(ids[1], {}).get("status") == "completed",
            "stage_ids": list(ids),
        }
        summary = summaries.get(ids[1]) or summaries.get(ids[0]) or {}
        active = summary.get("cumulative_elapsed_seconds")
        breakdown = as_dict(summary.get("timing_breakdown_seconds"))
        clock_valid = finite(active) and bool(breakdown) and all(finite(value) for value in breakdown.values())
        runner_active[method] = {
            "cumulative_elapsed_seconds": active if finite(active) else None,
            "timing_complete": clock_valid and summary.get("timing_complete") is True,
            "timing_breakdown_seconds": {key: value if finite(value) else None for key, value in breakdown.items()},
            "scope": "Latest runner cumulative clock includes its pilot; never sum pilot and resumed clocks",
        }
    return {
        "tuning_child_wall_seconds": groups["pilot"],
        "primary_continuation_child_wall_seconds": groups["primary"],
        "matched_continuation_child_wall_seconds": groups["matched"],
        "total_child_wall_seconds": sum(value["seconds"] for value in costs.values()),
        "selected_trajectory_child_wall": selected, "runner_selected_trajectory_active_clock": runner_active,
        "selected_pilot_overlap_seconds": sum(value["seconds"] for name, value in costs.items()
            if any(name == f"pilot_{method}_{lr_tag(s['selected_lr'])}" for method, s in selections.items()
                   if s.get("selected_lr") in RATES)),
        "measurement_complete": bool(costs) and all(value["measurement_complete"] for value in costs.values()),
        "stage_costs": costs, "issues": issues,
        "scope": "Recorded child monotonic wall includes imports/setup/train/eval/save/exit and retries. "
                 "Profiles, data preparation and inter-stage gaps are excluded. Running/incomplete clocks are lower bounds. "
                 "Tuning and selected-trajectory views overlap; total counts each stage once.",
    }


def curve(root, plan, method, lr, inputs):
    path = root / ("lr" + lr_tag(lr)) / f"{method}_seed419" / "train.jsonl"
    result = {"method": method, "lr": lr, "path": str(path), "available": path.is_file(), "points": [], "issues": []}
    if not result["available"]:
        result["issues"].append("Per-LR train.jsonl is missing")
        return result
    try:
        content = path.read_bytes()
    except OSError as error:
        result["issues"].append(f"Cannot read JSONL snapshot: {error}")
        result["available"] = False
        return result
    inputs[str(path)] = digest(content)
    invocation, identity_ok, last_clock = 0, False, -1.0
    lines = content.splitlines()
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError("expected an event object")
        except ValueError as error:
            label = "Partial trailing JSONL line" if number == len(lines) and not content.endswith(b"\n") else "Invalid JSONL line"
            result["issues"].append(f"{label} {number}: {error}")
            continue
        if row.get("kind") == "start":
            invocation += 1
            errors, _ = validate_identity(row, plan, method, lr)
            identity_ok = not errors
            result["issues"].extend(f"Invocation {invocation}: {issue}" for issue in errors)
        validation = row.get("validation") if row.get("kind") == "update" else row.get("final_validation")
        if validation is not None:
            errors = validate_validation(validation, 3906)
            update, tokens = row.get("update"), row.get("tokens")
            if not isinstance(update, int) or isinstance(update, bool) or not 0 <= update <= 16384 or tokens != update * 16384:
                errors.append("Curve update/token count mismatch")
            if not identity_ok:
                errors.append("No valid invocation identity for this observation")
            timestamp = validation.get("cumulative_elapsed_seconds") if isinstance(validation, dict) else None
            elapsed = validation.get("elapsed_this_invocation") if isinstance(validation, dict) else None
            clock_errors = []
            event_clock = row.get("cumulative_elapsed_seconds")
            if (not finite(timestamp) or timestamp < last_clock or not finite(elapsed)
                    or not finite(event_clock) or timestamp > event_clock):
                clock_errors.append("Missing/nonmonotonic evaluation clock")
            point = {"event": "periodic" if row.get("kind") == "update" else "final",
                     "invocation": invocation, "line": number,
                     "update": update if finite(update) else None, "tokens": tokens if finite(tokens) else None,
                     "nll": validation.get("nll") if isinstance(validation, dict) and finite(validation.get("nll")) else None,
                     "targets": validation.get("targets") if isinstance(validation, dict) else None,
                     "cumulative_elapsed_seconds": timestamp if finite(timestamp) else None,
                     "elapsed_this_invocation": elapsed if finite(elapsed) else None,
                     "quality_eligible": not errors, "clock_complete": not clock_errors and as_dict(validation).get("timing_complete") is True,
                     "issues": errors + clock_errors}
            result["points"].append(point)
        timestamp = row.get("cumulative_elapsed_seconds")
        if not finite(timestamp) or timestamp < last_clock:
            result["issues"].append(f"Missing/nonmonotonic event clock at line {number}")
        else:
            last_clock = timestamp
    result["scope"] = "Within-run checkpoints, not independent seeds; repeated update evaluations remain separate events"
    return result


def comparison(reference, candidate, reference_cost=None, candidate_cost=None):
    improvement = reference["final_checkpoint_nll"] - candidate["final_checkpoint_nll"]
    result = {"reference": reference["method"], "candidate": candidate["method"],
              "reference_lr": reference["lr"], "candidate_lr": candidate["lr"],
              "nll_improvement_nats": improvement, "tokens_each": TOKENS,
              "development_priority_signal": improvement >= PRIORITY_NATS,
              "interpretation": "One development seed; threshold prioritizes follow-up, not equivalence or significance"}
    if (reference_cost and candidate_cost and reference_cost["measurement_complete"] and candidate_cost["measurement_complete"]):
        result["selected_trajectory_wall_seconds_difference"] = candidate_cost["seconds"] - reference_cost["seconds"]
    return result


def analyze(study_root):
    root = Path(study_root).expanduser().resolve()
    inputs, issues = {}, []
    plan = read_json(root / "plan.json", inputs, issues)
    plan_errors = validate_plan(plan)
    report = {"format_version": 1, "study_root": str(root), "analysis_source_sha256": digest(Path(__file__).read_bytes()),
              "scope": "Single development seed 419; grid-selected final checkpoint NLL at fixed D. "
                       "No SOTA, equivalence, significance or mature-quality claim. Curves are not independent replicates.",
              "fixed_training_tokens": TOKENS, "fixed_validation_targets": TARGETS,
              "development_priority_threshold_nats": PRIORITY_NATS, "issues": issues, "plan_issues": plan_errors,
              "inputs_sha256": inputs, "input_snapshot_scope": "Each file read once; a live directory is not a transaction"}
    if plan_errors:
        return {**report, "status": "invalid_plan", "main_comparisons": [], "main_arms": []}
    plan_sha = json_digest(plan)
    report["plan_sha256"] = plan_sha
    state = read_json(root / "status.json", inputs, issues)
    selected = read_json(root / "selection.json", inputs, issues)
    state_ok = state.get("plan_sha256") == plan_sha
    selection_ok = selected.get("plan_sha256") == plan_sha and selected.get("methods") == state.get("selections")
    if not state_ok:
        issues.append("Status plan SHA is missing or differs from locked plan")
    if not selection_ok:
        issues.append("Selection is missing or inconsistent with status/locked plan")
    stages = state.get("stages", {})
    selections = selected.get("methods", {})
    if not isinstance(stages, dict) or not all(isinstance(r, dict) for r in stages.values()):
        issues.append("Status stages must be an object of stage objects")
        stages = {}
    if not isinstance(selections, dict) or not all(isinstance(r, dict) for r in selections.values()):
        issues.append("Selection methods must be an object of method objects")
        selections = {}
    pilots, primaries, summaries, tuning = {}, {}, {}, {}
    for method in METHODS:
        candidates = []
        for lr in RATES:
            stage_id = f"pilot_{method}_{lr_tag(lr)}"
            audit, summary = audit_record(root, plan, stage_id, method, lr, 2048, stages, inputs)
            pilots[stage_id], summaries[stage_id] = audit, summary
            candidates.append(audit)
        selection = selections.get(method, {})
        lr = selection.get("selected_lr")
        complete = all(candidate["eligible"] for candidate in candidates)
        optimum = min(candidates, key=lambda r: (r["final_checkpoint_nll"], r["lr"]))["lr"] if complete else None
        declared = selection.get("candidates")
        expected_candidates = {lr_tag(rate): stages.get(f"pilot_{method}_{lr_tag(rate)}") for rate in RATES}
        candidates_match = declared == expected_candidates
        good = (state_ok and selection_ok and complete and candidates_match
                and selection.get("comparison_complete") is True and lr == optimum)
        tuning[method] = {"eligible": good, "selected_lr": lr if lr in RATES else None,
                          "both_lr_pilots_successful": complete, "recomputed_optimum_lr": optimum,
                          "candidate_records_match": candidates_match}
        stage_id = f"primary_{method}"
        audit, summary = audit_record(root, plan, stage_id, method, lr, 16384, stages, inputs)
        summaries[stage_id] = summary
        if not good:
            audit["eligible"] = False
            audit["issues"].append("Successful verified two-LR selection is required for grid comparison")
            if audit["classification"] == "completed":
                audit["classification"] = "ineligible"
        primaries[method] = audit
    ledger = wall_ledger(stages, selections, summaries)
    eligible = [row for row in primaries.values() if row["eligible"]]
    baseline = primaries["baseline"]
    comparisons = []
    for row in eligible:
        if row["method"] != "baseline" and baseline["eligible"]:
            if row["quality_signature"] != baseline["quality_signature"]:
                row["eligible"] = False
                row["classification"] = "ineligible"
                row["issues"].append("Quality identity differs from baseline")
            else:
                comparisons.append(comparison(baseline, row, ledger["selected_trajectory_child_wall"].get("baseline"),
                                              ledger["selected_trajectory_child_wall"].get(row["method"])))
    controls = []
    phase = primaries["phase-adjoint"]
    for method in ("boundary-skip", "phase-adjoint-post-frozen"):
        control = primaries[method]
        same_lr = control["lr"] == phase["lr"] and phase["lr"] in RATES
        if not same_lr and phase["lr"] in RATES:
            stage_id = f"matched_{method}"
            control, summary = audit_record(root, plan, stage_id, method, phase["lr"], 16384, stages, inputs)
            summaries[stage_id] = summary
        good = (phase["eligible"] and control["eligible"] and control["lr"] == phase["lr"]
                and control["quality_signature"] == phase["quality_signature"]
                and pilots.get(f"pilot_{method}_{lr_tag(phase['lr'])}", {}).get("eligible", False))
        item = {"control": method, "eligible": good, "record": control,
                "scope": "Phase LR and D must both match; grid-selected LR mismatches cannot identify mechanism"}
        if good:
            item["comparison"] = comparison(control, phase)
            ids = (f"pilot_{method}_{lr_tag(phase['lr'])}", control["stage_id"])
            item["trajectory_child_wall_seconds"] = sum(ledger["stage_costs"].get(name, {}).get("seconds", 0.) for name in ids)
            item["wall_measurement_complete"] = all(ledger["stage_costs"].get(name, {}).get("measurement_complete", False) for name in ids)
        controls.append(item)
    main_complete = all(row["eligible"] for row in primaries.values())
    complete = main_complete and (not plan.get("matched_phase_lr") or all(row["eligible"] for row in controls))
    report.update(status="complete" if complete else "partial", main_grid_complete=main_complete,
                  driver_status=state.get("status", "missing"), tuning=tuning, pilots=list(pilots.values()),
                  main_arms=list(primaries.values()), main_comparisons=comparisons, phase_lr_matched_controls=controls,
                  wall_ledger=ledger, curves=[curve(root, plan, method, lr, inputs) for method in METHODS for lr in RATES])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Analysis JSON; default writes to stdout")
    parser.add_argument("--csv", type=Path, help="Optional primary-arm endpoint table")
    args = parser.parse_args(argv)
    report = analyze(args.study_root)
    for output in (args.output, args.csv):
        if output:
            path = output.expanduser().resolve()
            root = args.study_root.expanduser().resolve()
            reserved = path.name in ("plan.json", "status.json", "selection.json", "train.jsonl", "summary.json", "final.pt")
            if str(path) in report["inputs_sha256"] or reserved or path.parent == root / "records":
                parser.error("Analysis output cannot overwrite an input or reserved study artifact")
    if args.output and args.csv and args.output.resolve() == args.csv.resolve():
        parser.error("JSON and CSV outputs must be distinct files")
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(content)
    else:
        sys.stdout.write(content)
    if args.csv:
        columns = ("method", "lr", "classification", "eligible", "update", "tokens", "final_checkpoint_nll")
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["main_arms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
