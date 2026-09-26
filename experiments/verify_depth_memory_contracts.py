"""CPU contracts for bounded, leased depth-memory slots; no LM training.

Run: python3 experiments/verify_depth_memory_contracts.py
This reference accepts only the current proposal, current control decisions, and
past state. Leases in the capacity trace are externally supplied, not learned.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform

import torch

ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
TOL = 1e-12


@dataclass(frozen=True)
class Memory:
    values: torch.Tensor
    keys: torch.Tensor
    valid: torch.Tensor
    lease_until: torch.Tensor
    version: torch.Tensor


def empty_memory(slots, width, key_width=2):
    return Memory(torch.zeros(slots, width, dtype=DTYPE),
                  torch.zeros(slots, key_width, dtype=DTYPE),
                  torch.zeros(slots, dtype=torch.bool),
                  torch.zeros(slots, dtype=torch.int64),
                  torch.zeros(slots, dtype=torch.int64))


def bounded(value, radius):
    """Euclidean projection onto the closed radius-R ball."""
    return value * (radius / value.norm().clamp_min(radius))


def protected_rows(memory, depth):
    # A lease [commit_depth, lease_until) expires before lease_until's write.
    return memory.valid & (memory.lease_until > depth)


def functional_memory_step(memory, proposal, key, *, depth, radius, eta,
                           lease_until, eviction_risk=None):
    """One atomic slot commit, or rejection; workspace always receives v.

    Memory values must initially lie in the radius-R ball. Controls, including
    eta, selection, expiry, and proposal, are inputs here, not trained networks.
    The caller supplies the key metadata for the committed content; semantic
    correctness of that key is not a contract verified by this script.
    """
    if radius <= 0 or not 0 <= eta <= 1 or lease_until < depth + 1:
        raise ValueError("Require R>0, eta in [0,1], and expiry >= depth+1.")
    v = bounded(proposal, radius)
    available = ~protected_rows(memory, depth)
    candidates = torch.where(available & ~memory.valid)[0]
    if not len(candidates):
        candidates = torch.where(available)[0]
    if not len(candidates) or eta == 0:
        return memory, v, None
    if eviction_risk is None:
        slot = int(candidates[0])
    else:
        slot = int(candidates[torch.argmin(eviction_risk[candidates])])
    values, keys = memory.values.clone(), memory.keys.clone()
    valid, expiry = memory.valid.clone(), memory.lease_until.clone()
    version = memory.version.clone()
    old_value = values[slot] if bool(valid[slot]) else torch.zeros_like(v)
    values[slot] = (1 - eta) * old_value + eta * v
    keys[slot], valid[slot], expiry[slot] = key, True, lease_until
    version[slot] += 1
    return Memory(values, keys, valid, expiry, version), v, slot


def equal(a, b):
    return torch.equal(a, b)


def check(name, passed, **details):
    return {"name": name, "passed": bool(passed), **details}


def protection_and_metadata():
    memory = empty_memory(5, 6)
    values = torch.stack([bounded(torch.randn(6, dtype=DTYPE), 1.) for _ in range(5)])
    memory = Memory(values, torch.randn(5, 2, dtype=DTYPE),
                    torch.tensor([True, True, False, True, True]),
                    torch.tensor([8, 3, 0, 10, 4]), torch.tensor([2, 4, 0, 7, 9]))
    original_values = memory.values.clone()
    mask = protected_rows(memory, depth=3)
    new, workspace, slot = functional_memory_step(
        memory, torch.randn(6, dtype=DTYPE), torch.tensor([9., 7.], dtype=DTYPE),
        depth=3, radius=1., eta=.6, lease_until=12)
    unchanged = torch.arange(5) != slot
    return [check("protected_rows_are_bitwise_unchanged",
                  equal(new.values[mask], memory.values[mask]),
                  protected_slots=torch.where(mask)[0].tolist(), committed_slot=slot),
            check("commit_updates_one_slot_and_metadata_atomically",
                  slot == 2 and equal(new.values[unchanged], memory.values[unchanged])
                  and equal(new.keys[unchanged], memory.keys[unchanged])
                  and equal(new.valid[unchanged], memory.valid[unchanged])
                  and equal(new.lease_until[unchanged], memory.lease_until[unchanged])
                  and equal(new.version[unchanged], memory.version[unchanged])
                  and int(new.version[slot]) == int(memory.version[slot]) + 1
                  and equal(new.values[slot], .6 * workspace)
                  and equal(new.keys[slot], torch.tensor([9., 7.], dtype=DTYPE))
                  and bool(new.valid[slot]) and int(new.lease_until[slot]) == 12
                  and equal(memory.values, original_values),
                  scope="Invalid slot values are treated as zero despite nonzero backing storage. Commits increment slot version; expired slots retain validity until overwritten.")]


def long_run_bound():
    memory, radius = empty_memory(4, 8), 1.7
    max_norm, protected_error, commits, rejections = 0., 0., 0, 0
    for depth in range(4096):
        previous = memory
        mask = protected_rows(previous, depth)
        memory, workspace, slot = functional_memory_step(
            previous, 100 * torch.randn(8, dtype=DTYPE),
            torch.randn(2, dtype=DTYPE), depth=depth, radius=radius,
            eta=float(torch.rand(())), lease_until=depth + (depth % 9) + 1,
            eviction_risk=torch.rand(4, dtype=DTYPE))
        max_norm = max(max_norm, float(memory.values.norm(dim=-1).max()),
                       float(workspace.norm()))
        if bool(mask.any()):
            protected_error = max(protected_error, float(
                (memory.values[mask] - previous.values[mask]).abs().max()))
        commits += slot is not None
        rejections += slot is None
    return check("convex_slot_bound_over_4096_steps",
                 max_norm <= radius + TOL and protected_error == 0,
                 radius=radius, maximum_row_or_workspace_norm=max_norm,
                 maximum_protected_error=protected_error, commits=commits,
                 rejections=rejections, tolerance=TOL,
                 scope="Bounded state under arbitrary sampled proposals; no claim about training gradients.")


def rejection_and_expiry():
    memory = Memory(torch.eye(2, dtype=DTYPE), torch.eye(2, dtype=DTYPE),
                    torch.ones(2, dtype=torch.bool), torch.tensor([5, 9]),
                    torch.tensor([3, 4]))
    proposal, key = torch.tensor([-2., 1.], dtype=DTYPE), torch.tensor([4., 5.], dtype=DTYPE)
    rejected, workspace, slot = functional_memory_step(
        memory, proposal, key, depth=4, radius=1., eta=1., lease_until=10)
    reused, _, new_slot = functional_memory_step(
        rejected, proposal, key, depth=5, radius=1., eta=1., lease_until=10)
    zero_weight, _, zero_slot = functional_memory_step(
        memory, proposal, key, depth=5, radius=1., eta=0., lease_until=10)
    return [check("all_protected_rejects_commit_but_workspace_progresses",
                  slot is None and rejected is memory and
                  zero_slot is None and zero_weight is memory and
                  equal(workspace, bounded(proposal, 1.)) and
                  not equal(workspace, memory.values[0]),
                  scope="Current branch proposal remains usable; this does not prove a full network avoids stalls."),
            check("expiry_reuses_capacity_without_corrupting_live_slot",
                  new_slot == 0 and equal(reused.values[1], memory.values[1])
                  and equal(reused.keys[1], memory.keys[1])
                  and int(reused.version[0]) == int(memory.version[0]) + 1
                  and int(reused.version[1]) == int(memory.version[1])
                  and equal(reused.values[0], bounded(proposal, 1.)),
                  expiry_boundary=5, reused_slot=new_slot)]


def jacobian_scope():
    slots, width, eta, selected = 3, 2, .7, 1
    x = (.1 * torch.randn(slots, width, dtype=DTYPE)).requires_grad_()
    proposal = torch.randn(width, dtype=DTYPE)

    def frozen_update(values):
        memory = Memory(values, torch.zeros(slots, 2, dtype=DTYPE),
                        torch.ones(slots, dtype=torch.bool),
                        torch.zeros(slots, dtype=torch.int64),
                        torch.zeros(slots, dtype=torch.int64))
        new, _, committed = functional_memory_step(
            memory, proposal, torch.zeros(2, dtype=DTYPE), depth=0,
            radius=1., eta=eta, lease_until=1,
            eviction_risk=torch.tensor([1., 0., 1.], dtype=DTYPE))
        assert committed == selected
        return new.values

    jac = torch.autograd.functional.jacobian(frozen_update, x).reshape(6, 6)
    diagonal = torch.ones(slots, dtype=DTYPE)
    diagonal[selected] = 1 - eta
    expected = torch.kron(torch.diag(diagonal), torch.eye(width, dtype=DTYPE))
    error = float((jac - expected).abs().max())
    # Bounded proposal can still have a large derivative w.r.t. prior memory.
    z = torch.zeros((), dtype=DTYPE, requires_grad=True)
    full_jac = float(torch.autograd.functional.jacobian(
        lambda a: .5 * a + .5 * torch.tanh(8 * a), z))
    return [check("frozen_controls_and_proposal_have_diagonal_nonexpansive_transport",
                  error <= TOL and float(torch.linalg.matrix_norm(jac, ord=2)) <= 1 + TOL,
                  maximum_error=error, diagonal=diagonal.tolist(),
                  operator_norm=float(torch.linalg.matrix_norm(jac, ord=2)),
                  scope="Partial derivative through memory carry only; controls and proposal fixed."),
            check("bounded_state_does_not_imply_stable_full_jacobian",
                  abs(full_jac - 4.5) <= TOL and full_jac > 1,
                  full_scalar_derivative_at_zero=full_jac,
                  update="x_next = 0.5*x + 0.5*tanh(8*x), for x in [-1,1]",
                  scope="Counterexample: controller/branch derivatives must be studied separately.")]


def nonorthogonal_view_counterexample():
    p = torch.tensor([1., 0.], dtype=DTYPE)
    write = torch.tensor([0., 3.], dtype=DTYPE)
    key = p.clone()
    transport = torch.eye(2, dtype=DTYPE) + torch.outer(write, key)
    repeated_gain = float((torch.linalg.matrix_power(transport, 16) @ p).norm())
    operator_norm = float(torch.linalg.matrix_norm(transport, ord=2))
    return check("protected_view_nullspace_write_can_be_nonnormal_and_amplifying",
                 equal(p @ transport, p) and operator_norm > 3 and repeated_gain > 48,
                 protected_view=p.tolist(), write_direction=write.tolist(),
                 read_key=key.tolist(), transport=transport.tolist(),
                 spectral_radius=float(torch.linalg.eigvals(transport).abs().max()),
                 operator_norm=operator_norm, gain_after_16_steps=repeated_gain,
                 scope="Read invariance alone gives no Euclidean norm bound; motivates coordinate slots plus bounded convex writes.")


def run_causal_prefix(proposals):
    memory = empty_memory(3, 4)
    history = []
    for depth, proposal in enumerate(proposals):
        # Decisions use present depth/proposal and past state only.
        memory, workspace, slot = functional_memory_step(
            memory, proposal, proposal[:2], depth=depth, radius=1., eta=.8,
            lease_until=depth + 2, eviction_risk=memory.values.norm(dim=1))
        history.append((memory, workspace, slot))
    return history


def causal_prefix_check():
    first = torch.randn(12, 4, dtype=DTYPE)
    second = first.clone()
    second[7:] = torch.randn(5, 4, dtype=DTYPE) * 100
    a, b = run_causal_prefix(first), run_causal_prefix(second)
    same = all(equal(x.values, y.values) and equal(x.keys, y.keys)
               and equal(x.valid, y.valid) and equal(x.lease_until, y.lease_until)
               and equal(x.version, y.version)
               and equal(wx, wy) and sx == sy
               for (x, wx, sx), (y, wy, sy) in zip(a[:7], b[:7]))
    return check("reference_step_is_causal_under_identical_input_prefix",
                 same, identical_prefix_steps=7, changed_future_steps=5,
                 scope="Reference step and supplied current-only controls; not a learned-controller causality audit.")


def external_lease_capacity():
    # Known lifetimes are a trace contract, never future information for an LM.
    writes = [(t, t + 1 + t % 4) for t in range(64)]
    peak = max(sum(start <= t < end for start, end in writes) for t in range(68))

    def replay(capacity):
        memory, retained, rejected = empty_memory(capacity, 2, 1), True, 0
        contents = {}
        for depth, expiry in writes:
            v = torch.tensor([1., depth / 64], dtype=DTYPE)
            v = bounded(v, 1.)
            memory, _, slot = functional_memory_step(
                memory, v, torch.tensor([depth], dtype=DTYPE), depth=depth,
                radius=1., eta=1., lease_until=expiry)
            rejected += slot is None
            if slot is not None:
                contents[depth] = v
            for start, end in writes[:depth + 1]:
                if start <= depth < end:
                    matches = memory.valid & (memory.keys[:, 0] == start)
                    retained &= bool(matches.sum() == 1) and equal(
                        memory.values[matches].reshape(-1), contents.get(start, torch.empty(0)))
        return retained, rejected

    retained, rejected = replay(peak)
    _, rejected_below = replay(peak - 1)
    return check("external_lease_trace_capacity_tracks_concurrent_live_versions",
                 retained and rejected == 0 and rejected_below > 0,
                 total_writes=len(writes), maximum_concurrent_live_versions=peak,
                 slots=peak, rejected_at_peak_capacity=rejected,
                 rejected_with_one_fewer_slot=rejected_below,
                 lifetime_rule="write at t, expiry t+1+(t mod 4), half-open lifetime",
                 scope="Exact retention for externally supplied leases and distinct versions in this trace. No learned lease, semantic compression, task gain, or general performance claim.")


def predicted_eviction_regret():
    true_risk = torch.rand(1024, 6, dtype=DTYPE)
    epsilon = .075
    errors = epsilon * (2 * torch.rand_like(true_risk) - 1)
    predicted = true_risk + errors
    selected = predicted.argmin(dim=1)
    selected_true = true_risk.gather(1, selected[:, None]).squeeze(1)
    regret = selected_true - true_risk.min(dim=1).values
    return check("minimum_predicted_risk_regret_is_at_most_twice_error_bound",
                 float(regret.max()) <= 2 * epsilon + TOL,
                 samples=1024, assumed_uniform_prediction_error_bound=epsilon,
                 theoretical_regret_bound=2 * epsilon, maximum_sampled_regret=float(regret.max()),
                 scope="Elementary selection bound conditional on known uniform error; no estimator calibration or learned eviction evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/depth_memory_contracts_20260925/contracts.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checks = (protection_and_metadata() + [long_run_bound()] + rejection_and_expiry()
              + jacobian_scope() + [nonorthogonal_view_counterexample(), causal_prefix_check(),
                                    external_lease_capacity(), predicted_eviction_regret()])
    report = {"seed": args.seed, "precision": str(DTYPE), "device": "cpu",
              "python": platform.python_version(), "torch": torch.__version__,
              "training_performed": False,
              "contract_scope": "Functional slot-memory prototype and numerical mathematical checks; no model integration, optimization, benchmark, learned controller, or SOTA evidence.",
              "checks": checks, "passed": all(item["passed"] for item in checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}")
    print(f"Receipt: {args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
