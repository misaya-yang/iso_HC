"""Controller contracts only: every subprocess is mocked; no training is launched."""

import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments import run_residual_study as study

NOW = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)


def minimal_plan(root):
    return {
        "paths": {"repo_root": str(root), "output_root": str(root / "output"), "python": "chosen-python",
                  "train_cache": "/data/train.pt", "val_cache": "/data/val.pt",
                  "data_manifest": "/data/manifest.json"},
        "protocol": dict(study.PROTOCOL), "compile": False,
        "compile_cache_root": None,
        "profile_rates": dict.fromkeys(study.METHODS, 100000), "stage_overhead_seconds": 600,
        "reserve_seconds": 1200, "deadline_utc": (NOW + timedelta(days=1)).isoformat(),
        "pilot_order": [{"method": m, "lr": lr} for m in study.METHODS for lr in study.LEARNING_RATES],
        "primary_order": list(study.METHODS), "matched_phase_lr": True,
        "matched_order": ["phase-adjoint-post-frozen", "boundary-skip"],
    }


class StudyTests(unittest.TestCase):
    def test_finite_successful_lr_selection_and_exact_tie(self):
        records = {3e-4: {"status": "completed", "nll": 4.0},
                   6e-4: {"status": "completed", "nll": 3.9}}
        self.assertEqual(study.select_learning_rate(records), 6e-4)
        records[6e-4]["nll"] = 4.0
        self.assertEqual(study.select_learning_rate(records), 3e-4)

    def test_failed_missing_and_nonfinite_trials_are_ineligible(self):
        for nll in (float("nan"), float("inf"), float("-inf"), -1, None, True):
            with self.subTest(nll=nll):
                self.assertEqual(study.select_learning_rate({
                    3e-4: {"status": "completed", "nll": nll},
                    6e-4: {"status": "completed", "nll": 4.1},
                }), 6e-4)
        self.assertEqual(study.select_learning_rate({
            3e-4: {"status": "failed", "nll": 0.01},
            6e-4: {"status": "completed", "nll": 4.1},
        }), 6e-4)
        self.assertIsNone(study.select_learning_rate({3e-4: {"status": "failed", "nll": 0.01}}))

    def test_numerical_policy_requires_raw_eager_eval_cast_emulation_and_backward_amp_off(self):
        expected = study.expected_numerical_policy(True)
        actual = {**expected, "evaluation_autocast_dtype": "torch.bfloat16"}
        identity = {"amp_dtype": "torch.bfloat16", "numerical_policy": actual}
        study.validate_numerical_policy(expected, identity)
        for change in ({"evaluation": "compiled"}, {"backward_call_autocast": "same_as_forward"},
                       {"emulate_precision_casts": {"effective": False}}, {"evaluation_autocast_dtype": "torch.float16"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                study.validate_numerical_policy(expected, {**identity, "numerical_policy": {**actual, **change}})

    def test_deadline_reserve_is_before_deadline_and_fits_exact_boundary(self):
        deadline = NOW + timedelta(seconds=120)
        self.assertTrue(study.can_start_stage(NOW, deadline, 100, 20))
        self.assertFalse(study.can_start_stage(NOW, deadline, 101, 20))
        self.assertFalse(study.can_start_stage(deadline, deadline, 1, 0))
        for estimate, reserve in ((-1, 0), (0, -1), (float("nan"), 0)):
            with self.assertRaises(ValueError):
                study.can_start_stage(NOW, deadline, estimate, reserve)
        self.assertEqual(study.parse_utc("2026-09-30T07:15:00Z").utcoffset(), timedelta(0))
        for value in ("2026-09-30T07:15:00", "2026-09-30T07:15:00+08:00"):
            with self.assertRaises(ValueError):
                study.parse_utc(value)

    def test_pilot_and_resume_commands_keep_full_schedule_and_expected_arm_path(self):
        plan = minimal_plan(Path("/study"))
        pilot = study.stage_command(plan, "terminal-adjoint", 6e-4, 2048)
        resumed = study.stage_command(plan, "terminal-adjoint", 6e-4, 16384, resume=True)
        self.assertEqual(pilot[0], "chosen-python")
        for command in (pilot, resumed):
            for flag, value in (("--updates", "16384"), ("--warmup-updates", "256"),
                                ("--micro-batch", "32"), ("--grad-accum", "1"),
                                ("--val-blocks", "3906"), ("--data-seed", "20260929"),
                                ("--model-seed", "419"), ("--min-lr", str(6e-4 / 10))):
                self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--resume", pilot)
        self.assertEqual(pilot[pilot.index("--stop-after") + 1], "2048")
        self.assertEqual(resumed[resumed.index("--stop-after") + 1], "16384")
        self.assertEqual(resumed[resumed.index("--output-dir") + 1], "/study/output/lr6e-4")
        self.assertEqual(resumed[resumed.index("--resume") + 1],
                         "/study/output/lr6e-4/terminal-adjoint_seed419/final.pt")

    def test_lock_rejects_duplicate_active_controller_but_allows_stale_pid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with study.controller_lock(root):
                with self.assertRaisesRegex(RuntimeError, "active child"):
                    with study.controller_lock(root):
                        self.fail("Duplicate controller acquired the lock")
            self.assertTrue((root / "controller.lock").exists())
            with study.controller_lock(root) as fd:
                self.assertGreaterEqual(fd, 0)

    def test_compiled_lr_arms_and_continuations_reuse_profile_cache_per_method(self):
        plan = minimal_plan(Path("/study"))
        plan.update(compile=True, compile_cache_root="/r5/cache/canonical-m32")
        first = study.stage_environment(plan, "phase-adjoint")
        self.assertEqual(first["TORCHINDUCTOR_CACHE_DIR"], "/r5/cache/canonical-m32/phase-adjoint")
        for lr in study.LEARNING_RATES:
            for resume in (False, True):
                command = study.stage_command(plan, "phase-adjoint", lr, 16384, resume=resume)
                self.assertIn("--compile", command)
                self.assertEqual(study.stage_environment(plan, "phase-adjoint"), first)
        self.assertNotEqual(study.stage_environment(plan, "baseline"), first)

    def test_deadline_skip_never_launches_a_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = minimal_plan(Path(directory))
            plan["deadline_utc"] = (NOW + timedelta(seconds=100)).isoformat()
            state = {"stages": {}}
            with patch.object(study, "utc_now", return_value=NOW), patch.object(study.subprocess, "Popen") as popen:
                result = study.execute_stage(plan, state, "pilot_baseline", "baseline", 3e-4, 2048, 0, 9)
            self.assertEqual(result["status"], "skipped_deadline")
            popen.assert_not_called()
            self.assertEqual(json.loads((Path(plan["paths"]["output_root"]) / "status.json").read_text()), state)

    def test_nonzero_exit_cannot_use_stale_finite_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = minimal_plan(Path(directory))
            state = {"stages": {}}
            child = Mock(pid=1234)
            child.wait.return_value = 7
            with patch.object(study, "utc_now", return_value=NOW), patch.object(
                study.subprocess, "Popen", return_value=child
            ) as popen, patch.object(study, "validate_summary") as validate, patch.object(
                study.time, "monotonic", side_effect=[100.0, 112.5]
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = study.execute_stage(plan, state, "pilot_baseline", "baseline", 3e-4, 2048, 0, 9)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["returncode"], 7)
            self.assertEqual(result["child_wall_seconds"], 12.5)
            self.assertEqual(result["cumulative_child_wall_seconds"], 12.5)
            self.assertTrue(result["child_wall_complete"])
            self.assertIsNone(study.select_learning_rate({3e-4: result}))
            validate.assert_not_called()
            self.assertEqual(popen.call_args.kwargs["pass_fds"], (9,))

    def test_completed_stage_reuse_checks_method_lr_and_stop_after(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = minimal_plan(Path(directory))
            record = {"status": "completed", "method": "baseline", "lr": 3e-4, "stop_after": 16384}
            state = {"stages": {"primary_baseline": record}}
            with patch.object(study.subprocess, "Popen") as popen:
                self.assertIs(study.execute_stage(
                    plan, state, "primary_baseline", "baseline", 3e-4, 16384, 2048, 9
                ), record)
                for change in ({"method": "gain"}, {"lr": 6e-4}, {"stop_after": 2048}):
                    state["stages"]["primary_baseline"] = {**record, **change}
                    with self.subTest(change=change), self.assertRaisesRegex(ValueError, "identity mismatch"):
                        study.execute_stage(plan, state, "primary_baseline", "baseline", 3e-4, 16384, 2048, 9)
                popen.assert_not_called()

    def test_status_failure_after_spawn_stops_scheduling_with_child_exit_unconfirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = minimal_plan(Path(directory))
            child = Mock(pid=1234)
            writes = 0
            real_atomic_json = study.atomic_json

            def atomic_json(path, value):
                nonlocal writes
                writes += 1
                if writes == 2:
                    raise OSError("post-spawn status write failed")
                real_atomic_json(path, value)

            with patch.object(study, "utc_now", return_value=NOW), patch.object(
                study.subprocess, "Popen", return_value=child
            ) as popen, patch.object(study, "atomic_json", side_effect=atomic_json), patch.object(
                study.time, "monotonic", side_effect=[100.0, 105.0]
            ), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(study.SchedulingStopped, "exit is unconfirmed"):
                    study.run_study(plan, 9)
            self.assertEqual(popen.call_count, 1)
            child.wait.assert_not_called()
            child.kill.assert_not_called()
            child.terminate.assert_not_called()
            state = json.loads((Path(plan["paths"]["output_root"]) / "status.json").read_text())
            record = state["stages"]["pilot_baseline_3e-4"]
            self.assertEqual(state["status"], "partial")
            self.assertEqual(record["status"], "failed")
            self.assertTrue(record["child_exit_unconfirmed"])
            self.assertFalse(record["child_wall_complete"])
            self.assertEqual(record["child_wall_seconds"], 5.0)
            self.assertFalse(state["wall_costs"]["timing_complete"])

    def test_restart_after_any_primary_start_freezes_selection_and_does_not_retry_pilots(self):
        for primary_status in ("running", "completed", "failed", "interrupted"):
            with self.subTest(primary_status=primary_status), tempfile.TemporaryDirectory() as directory:
                plan = minimal_plan(Path(directory))
                stages = {}
                for method in study.METHODS:
                    for lr in study.LEARNING_RATES:
                        status = "completed"
                        if lr == 6e-4 and method == "baseline":
                            status = "failed"
                        if lr == 6e-4 and method == "gain":
                            status = "skipped_deadline"
                        stages[f"pilot_{method}_{study.lr_tag(lr)}"] = {
                            "method": method, "lr": lr, "stop_after": 2048, "status": status,
                            "nll": 4.0 if lr == 3e-4 else 3.0,
                        }
                stages["primary_baseline"] = {"method": "baseline", "lr": 3e-4, "stop_after": 16384,
                                              "status": primary_status, "started_utc": NOW.isoformat()}
                selections = {method: {"selected_lr": 3e-4, "comparison_complete": False} for method in study.METHODS}
                state = {"plan_sha256": study.json_digest(plan), "status": "partial",
                         "stages": stages, "selections": selections}
                root = Path(plan["paths"]["output_root"])
                study.atomic_json(root / "status.json", state)
                selection_record = {"plan_sha256": state["plan_sha256"], "methods": selections}
                study.atomic_json(root / "selection.json", selection_record)
                original_pilots = {key: dict(value) for key, value in stages.items() if key.startswith("pilot_")}
                calls = []

                def stage(plan, state, stage_id, method, lr, stop, initial, fd):
                    self.assertTrue(stage_id.startswith("primary_"))
                    self.assertEqual(lr, 3e-4)
                    calls.append(stage_id)
                    record = {"status": "completed", "method": method, "lr": lr, "stop_after": stop, "nll": 3.5}
                    state["stages"][stage_id] = record
                    return record

                with patch.object(study, "execute_stage", side_effect=stage):
                    resumed = study.run_study(plan, 9)
                self.assertEqual(calls, [f"primary_{method}" for method in study.METHODS])
                self.assertTrue(resumed["selections_frozen"])
                self.assertEqual(resumed["selections"], selections)
                self.assertEqual({key: value for key, value in resumed["stages"].items() if key.startswith("pilot_")}, original_pilots)
                self.assertEqual(json.loads((root / "selection.json").read_text()), selection_record)

    def test_all_pilots_precede_primaries_then_matched_controls_in_requested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = minimal_plan(Path(directory))
            calls = []

            def stage(plan, state, stage_id, method, lr, stop, initial, fd):
                calls.append(stage_id)
                optimum = 6e-4 if method == "phase-adjoint" else 3e-4
                record = {"status": "completed", "nll": 4.0 if lr == optimum else 4.1}
                state["stages"][stage_id] = record
                return record

            with patch.object(study, "execute_stage", side_effect=stage):
                state = study.run_study(plan, 9)
            self.assertEqual(len(calls), 23)
            self.assertTrue(all(stage.startswith("pilot_") for stage in calls[:14]))
            self.assertEqual(calls[14:21], [f"primary_{method}" for method in study.METHODS])
            self.assertEqual(calls[21:], ["matched_phase-adjoint-post-frozen", "matched_boundary-skip"])
            self.assertEqual(state["status"], "complete")

    def test_wall_ledger_distinguishes_tuning_from_selected_trajectory_without_double_counting_total(self):
        state = {"selections": {"baseline": {"selected_lr": 3e-4}}, "stages": {
            "pilot_baseline_3e-4": {"cumulative_child_wall_seconds": 10},
            "pilot_baseline_6e-4": {"cumulative_child_wall_seconds": 12},
            "primary_baseline": {"cumulative_child_wall_seconds": 80},
        }}
        costs = study.wall_costs(state)
        self.assertEqual(costs["tuning_child_wall_seconds"], 22)
        self.assertEqual(costs["selected_trajectory_child_wall_seconds"], {"baseline": 90})
        self.assertEqual(costs["total_child_wall_seconds"], 102)

    def test_dry_run_reads_no_token_tensors_and_execution_requires_existing_reviewed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "experiments/residual_lm_diagnostic.py").write_text("# source fixture\n")
            rates = root / "rates.json"
            rates.write_text(json.dumps(dict.fromkeys(study.METHODS, 100000)))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"status": "complete", "outputs": {
                "train": {"path": str(root / "train.pt"), "file_sha256": "a" * 64},
                "val": {"path": str(root / "val.pt"), "file_sha256": "b" * 64},
            }}))
            common = ["--repo-root", str(root), "--train-cache", str(root / "train.pt"),
                      "--val-cache", str(root / "val.pt"), "--data-manifest", str(manifest),
                      "--profile-rates", str(rates), "--deadline-utc", "2026-09-30T07:15:00+00:00"]
            with patch.object(study.subprocess, "Popen") as popen:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(study.main(common + ["--output-root", str(root / "plan"), "--dry-run"]), 0)
                self.assertTrue((root / "plan/plan.json").is_file())
                self.assertFalse((root / "train.pt").exists())
                self.assertFalse((root / "val.pt").exists())
                with self.assertRaisesRegex(FileNotFoundError, "review plan.json"):
                    study.main(common + ["--output-root", str(root / "unreviewed"), "--execute-reviewed-plan"])
                popen.assert_not_called()
