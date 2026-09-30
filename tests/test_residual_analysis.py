"""Synthetic JSON contracts only; no runner import, tensor loads or training."""

import copy
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments import analyze_residual_study as analysis


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        repo = Path(__file__).resolve().parents[1]
        self.frozen = {path: path.read_bytes() for path in (
            repo / "experiments/residual_lm_diagnostic.py", repo / "experiments/run_residual_study.py",
            repo / "lm/phase_adjoint.py")}
        self.plan = json.loads((repo / "results/phase_adjoint_20260929/study_plan.json").read_text())
        self.plan["paths"]["output_root"] = str(self.root)
        self.plan_sha = analysis.json_digest(self.plan)
        self.stages, self.selections = {}, {}
        self.write("plan.json", self.plan)

    def tearDown(self):
        for path, original in self.frozen.items():
            self.assertEqual(path.read_bytes(), original)
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        return path

    def summary(self, method, lr, update, nll):
        config = {**self.plan["protocol"], "method": method, "lr": lr, "min_lr": lr / 10,
                  "train_cache": self.plan["paths"]["train_cache"], "val_cache": self.plan["paths"]["val_cache"],
                  "train_manifest": self.plan["paths"]["data_manifest"], "val_manifest": self.plan["paths"]["data_manifest"],
                  "stop_after": update, "output_dir": "/fixture/arm", "resume": None,
                  "dry_run": False, "profile_steps": 0, "profile_warmup": 3}
        config.update({key: self.plan[key] for key in ("compile", "compile_backend", "compile_mode", "amp", "deterministic")})
        source = {key: value for key, value in self.plan["source_sha256"].items() if key != "controller"}
        policy = copy.deepcopy(self.plan["numerical_policy"])
        policy["evaluation_autocast_dtype"] = "torch.bfloat16"
        policy["emulate_precision_casts"]["config_source_sha256"] = "e" * 64
        policy["backward_pass_autocast"]["config_source_sha256"] = "f" * 64
        identity = {"config": {key: value for key, value in config.items() if key not in analysis.EXCLUDED_CONFIG},
                    "source": source, "torch": "2.12.1+cu130", "amp_dtype": "torch.bfloat16",
                    "numerical_policy": policy}
        for split in ("train", "val"):
            identity[split] = {"sha256": self.plan["cache_sha256"][split], "split": split,
                               "manifest_sha256": self.plan["data_manifest_sha256"]}
        clock = 9. if update == 2048 else 100.
        return {"kind": "training", "status": "stopped" if update < 16384 else "complete",
                "config": config, "identity": identity, "numerical_policy": policy,
                "provenance": {"source_sha256": source, "torch": identity["torch"], "numerical_policy": policy},
                "update": update, "tokens": update * 16384, "data_cursor": update * 32,
                "planned_tokens": analysis.TOKENS, "validation_targets": analysis.TARGETS,
                "final_validation": {"nll": nll, "targets": analysis.TARGETS, "blocks": 3906,
                                     "cumulative_elapsed_seconds": clock - 1, "elapsed_this_invocation": clock - 1,
                                     "timing_complete": True}, "best_nll": .1,
                "cumulative_elapsed_seconds": clock, "elapsed_this_invocation": clock,
                "timing_complete": True, "timing_breakdown_seconds": {"evaluation_seconds": 2., "save_seconds": 1.}}

    def add_stage(self, stage_id, method, lr, update, nll, seconds):
        self.write(f"records/{stage_id}.json", self.summary(method, lr, update, nll))
        self.stages[stage_id] = {"method": method, "lr": lr, "stop_after": update,
                                 "status": "completed", "nll": nll, "cumulative_child_wall_seconds": seconds,
                                 "cumulative_wall_complete": True, "started_utc": "2026-09-29T23:00:00+00:00"}

    def finish_state(self):
        self.write("status.json", {"plan_sha256": self.plan_sha, "status": "complete",
                                   "stages": self.stages, "selections": self.selections})
        self.write("selection.json", {"plan_sha256": self.plan_sha, "methods": self.selections})

    def fixture(self, matched=True):
        for index, method in enumerate(analysis.METHODS):
            selected_lr = 6e-4 if method == "phase-adjoint" else 3e-4
            for lr, seconds in zip(analysis.RATES, (10., 12.)):
                nll = 4.0 if lr == selected_lr else 4.1
                self.add_stage(f"pilot_{method}_{analysis.lr_tag(lr)}", method, lr, 2048, nll, seconds)
            self.selections[method] = {"selected_lr": selected_lr, "comparison_complete": True,
                "candidates": {analysis.lr_tag(lr): copy.deepcopy(self.stages[f"pilot_{method}_{analysis.lr_tag(lr)}"])
                               for lr in analysis.RATES}}
            self.add_stage(f"primary_{method}", method, selected_lr, 16384, 3.5 - index * .01, 80.)
        if matched:
            for method in ("boundary-skip", "phase-adjoint-post-frozen"):
                self.add_stage(f"matched_{method}", method, 6e-4, 16384, 3.49, 82.)
        self.finish_state()

    def test_final_nll_fixed_budget_and_cost_views_avoid_pilot_double_count(self):
        self.fixture()
        report = analysis.analyze(self.root)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["main_comparisons"]), 6)
        baseline = report["main_arms"][0]
        self.assertEqual(baseline["final_checkpoint_nll"], 3.5)
        self.assertNotEqual(baseline["final_checkpoint_nll"], .1)
        ledger = report["wall_ledger"]
        self.assertEqual(ledger["tuning_child_wall_seconds"], 154.)
        self.assertEqual(ledger["selected_trajectory_child_wall"]["baseline"]["seconds"], 90.)
        self.assertEqual(ledger["total_child_wall_seconds"], 878.)
        self.assertEqual(ledger["selected_pilot_overlap_seconds"], 72.)
        self.assertEqual(ledger["runner_selected_trajectory_active_clock"]["baseline"]["cumulative_elapsed_seconds"], 100.)
        self.assertTrue(all(row["eligible"] for row in report["phase_lr_matched_controls"]))

    def test_partial_token_budget_is_visible_and_excluded_from_comparisons(self):
        self.fixture()
        self.write("records/primary_gain.json", self.summary("gain", 3e-4, 8192, 2.0))
        self.stages["primary_gain"]["status"] = "running"
        self.finish_state()
        report = analysis.analyze(self.root)
        arm = next(row for row in report["main_arms"] if row["method"] == "gain")
        self.assertEqual(arm["classification"], "partial")
        self.assertFalse(arm["eligible"])
        self.assertIsNone(arm["final_checkpoint_nll"])
        self.assertEqual(arm["observed_final_checkpoint_nll"], 2.0)
        self.assertFalse(any(row["candidate"] == "gain" for row in report["main_comparisons"]))
        self.assertFalse(report["wall_ledger"]["measurement_complete"])

    def test_two_successful_lr_trials_and_correct_frozen_selection_are_required(self):
        self.fixture()
        self.stages["pilot_baseline_6e-4"]["status"] = "failed"
        self.selections["baseline"]["comparison_complete"] = False
        self.finish_state()
        report = analysis.analyze(self.root)
        self.assertFalse(report["tuning"]["baseline"]["eligible"])
        self.assertFalse(report["main_arms"][0]["eligible"])
        self.fixture()
        self.selections["baseline"]["selected_lr"] = 6e-4
        self.finish_state()
        report = analysis.analyze(self.root)
        self.assertEqual(report["tuning"]["baseline"]["recomputed_optimum_lr"], 3e-4)
        self.assertFalse(report["tuning"]["baseline"]["eligible"])

    def test_source_evaluator_and_validation_scope_mismatches_fail_closed(self):
        self.fixture()
        original = self.summary("gain", 3e-4, 16384, 3.49)
        for fault in ("source", "policy", "targets", "status_plan"):
            with self.subTest(fault=fault):
                row = copy.deepcopy(original)
                if fault == "source":
                    row["identity"]["source"]["experiments/residual_lm_diagnostic.py"] = "0" * 64
                elif fault == "policy":
                    row["identity"]["numerical_policy"]["evaluation"] = "compiled"
                elif fault == "targets":
                    row["final_validation"]["targets"] -= 1
                self.write("records/primary_gain.json", row)
                self.finish_state()
                if fault == "status_plan":
                    state = json.loads((self.root / "status.json").read_text())
                    state["plan_sha256"] = "wrong"
                    self.write("status.json", state)
                report = analysis.analyze(self.root)
                self.assertFalse(next(arm for arm in report["main_arms"] if arm["method"] == "gain")["eligible"])

    def test_lr_mismatched_or_partial_controls_do_not_support_mechanism_comparison(self):
        self.fixture(matched=False)
        report = analysis.analyze(self.root)
        self.assertTrue(all(arm["eligible"] for arm in report["main_arms"]))
        self.assertTrue(all(not row["eligible"] for row in report["phase_lr_matched_controls"]))
        self.add_stage("matched_boundary-skip", "boundary-skip", 6e-4, 8192, 2., 20.)
        self.finish_state()
        report = analysis.analyze(self.root)
        self.assertFalse(report["phase_lr_matched_controls"][0]["eligible"])

    def test_exact_lr_tie_uses_lower_lr_and_nonfinite_final_record_is_ineligible(self):
        self.fixture()
        self.add_stage("pilot_baseline_6e-4", "baseline", 6e-4, 2048, 4., 12.)
        self.selections["baseline"]["candidates"]["6e-4"] = copy.deepcopy(self.stages["pilot_baseline_6e-4"])
        self.finish_state()
        report = analysis.analyze(self.root)
        self.assertTrue(report["tuning"]["baseline"]["eligible"])
        self.assertEqual(report["tuning"]["baseline"]["recomputed_optimum_lr"], 3e-4)
        self.write("records/primary_gain.json", self.summary("gain", 3e-4, 16384, float("nan")))
        report = analysis.analyze(self.root)
        arm = next(row for row in report["main_arms"] if row["method"] == "gain")
        self.assertFalse(arm["eligible"])
        self.assertTrue(any("Nonfinite JSON" in issue for issue in arm["issues"]))
        json.dumps(report, allow_nan=False)

    def test_incomplete_wall_keeps_quality_but_omits_a_complete_cost_comparison(self):
        self.fixture()
        self.stages["primary_gain"].update(cumulative_child_wall_seconds=95., cumulative_wall_complete=False)
        self.finish_state()
        report = analysis.analyze(self.root)
        ledger = report["wall_ledger"]
        self.assertEqual(ledger["total_child_wall_seconds"], 893.)
        self.assertFalse(ledger["measurement_complete"])
        gain = next(row for row in report["main_comparisons"] if row["candidate"] == "gain")
        self.assertNotIn("selected_trajectory_wall_seconds_difference", gain)
        self.stages["primary_gain"]["status"] = "skipped_deadline"
        self.finish_state()
        self.assertFalse(analysis.analyze(self.root)["wall_ledger"]["measurement_complete"])

    def test_curve_preserves_resume_clocks_actual_tokens_and_repeated_evaluations(self):
        self.fixture()
        pilot, primary = self.summary("baseline", 3e-4, 2048, 4.), self.summary("baseline", 3e-4, 16384, 3.5)
        def start(summary, update, clock):
            return {"kind": "start", "update": update, "cumulative_elapsed_seconds": clock,
                    **{key: summary[key] for key in ("config", "identity", "provenance", "numerical_policy")}}
        def periodic(update, clock, nll):
            return {"kind": "update", "update": update, "tokens": update * 16384,
                    "cumulative_elapsed_seconds": clock + .1,
                    "validation": {"nll": nll, "targets": analysis.TARGETS, "blocks": 3906,
                                   "cumulative_elapsed_seconds": clock, "elapsed_this_invocation": clock,
                                   "timing_complete": True}}
        rows = [start(pilot, 0, 0.), periodic(2048, 7., 4.), {**pilot, "kind": "end"},
                start(primary, 2048, 10.), periodic(4096, 20., 3.9), {**primary, "kind": "end"}]
        path = self.root / "lr3e-4/baseline_seed419/train.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"kind":"update"')
        report = analysis.analyze(self.root)
        curve = report["curves"][0]
        self.assertEqual([point["tokens"] for point in curve["points"]], [33554432, 33554432, 67108864, analysis.TOKENS])
        self.assertEqual([point["cumulative_elapsed_seconds"] for point in curve["points"]], [7., 8., 20., 99.])
        self.assertEqual([point["invocation"] for point in curve["points"]], [1, 1, 2, 2])
        self.assertTrue(all(point["quality_eligible"] and point["clock_complete"] for point in curve["points"]))
        self.assertTrue(any("Partial trailing" in issue for issue in curve["issues"]))

    def test_missing_partial_state_and_invalid_plan_report_explicit_issues(self):
        report = analysis.analyze(self.root)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(len(report["main_arms"]), 7)
        self.assertTrue(all(row["classification"] == "unfinished" for row in report["main_arms"]))
        self.assertTrue(report["issues"])
        self.assertFalse(report["wall_ledger"]["measurement_complete"])
        self.plan["protocol"]["updates"] -= 1
        self.write("plan.json", self.plan)
        report = analysis.analyze(self.root)
        self.assertEqual(report["status"], "invalid_plan")
        self.assertEqual(report["main_comparisons"], [])

    def test_cli_writes_inspectable_json_and_csv_and_protects_inputs(self):
        self.fixture()
        output, table = self.root / "analysis.json", self.root / "analysis.csv"
        self.assertEqual(analysis.main(["--study-root", str(self.root), "--output", str(output), "--csv", str(table)]), 0)
        self.assertEqual(json.loads(output.read_text())["fixed_training_tokens"], analysis.TOKENS)
        with table.open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 7)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            analysis.main(["--study-root", str(self.root), "--output", str(self.root / "plan.json")])


if __name__ == "__main__":
    unittest.main()
