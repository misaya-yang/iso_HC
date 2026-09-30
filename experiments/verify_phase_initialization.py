"""Independent fixed-source audit of the two-phase reciprocal residual rule.

This is a CPU algebra/derivative receipt, not training or a language benchmark.
It does not import the candidate implementation.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


def global_kernel(angles, boundary):
    phase = torch.arange(len(angles)) >= boundary
    base = angles.new_tensor(math.pi / 4) * phase
    address = torch.stack(((angles + base).cos(), (angles + base).sin()), dim=-1)
    scale = base.cos()
    return (address @ address.T) * scale[None, :] / scale[:, None]


def audit():
    torch.set_num_threads(1)
    dtype = torch.float64
    count, width, boundary = 6, 4, 3
    angles = torch.zeros(count, dtype=dtype, requires_grad=True)
    kernel = global_kernel(angles, boundary)
    mask = torch.tril(torch.ones_like(kernel, dtype=torch.bool), diagonal=-1)
    kernel_error = (kernel[mask] - 1).abs().max().item()
    entry = kernel[boundary, 0]
    early_gradient = torch.autograd.grad(entry, angles)[0][0].item()
    tied_kernel = torch.cos(angles[:, None] - angles[None, :])
    tied_gradient = torch.autograd.grad(tied_kernel[boundary, 0], angles)[0][0].item()

    generator = torch.Generator().manual_seed(931)
    weights = torch.randn(count, width, width, generator=generator, dtype=dtype) * .2
    biases = torch.randn(count, width, generator=generator, dtype=dtype) * .1
    initial = torch.randn(width, generator=generator, dtype=dtype)
    target = torch.randn(width, generator=generator, dtype=dtype)

    def branch(index, state):
        return .3 * torch.tanh(weights[index] @ state + biases[index])

    def forward(route, coordinates):
        primary, auxiliary = initial.clone(), torch.zeros_like(initial)
        for index in range(count):
            t = route[index].tanh()
            c0 = (1 + t.square()).rsqrt()
            c1 = t * c0
            if coordinates:
                if index == boundary:
                    primary, auxiliary = primary + auxiliary, auxiliary - primary
                delta = branch(index, c0 * primary + c1 * auxiliary)
                primary, auxiliary = primary + c0 * delta, auxiliary + c1 * delta
            else:
                if index < boundary:
                    q0, q1, divisor = c0, c1, 1
                else:
                    q0, q1, divisor = c0 - c1, c0 + c1, 2
                delta = branch(index, q0 * primary + q1 * auxiliary)
                primary, auxiliary = primary + q0 * delta / divisor, auxiliary + q1 * delta / divisor
        return primary if coordinates else primary + auxiliary

    baseline = initial.clone()
    sources = []
    for index in range(count):
        delta = branch(index, baseline)
        sources.append(delta)
        baseline = baseline + delta
    zero = torch.zeros(count, dtype=dtype, requires_grad=True)
    initial_output = forward(zero, True)
    init_error = (initial_output - baseline).abs().max().item()
    initial_gradient = torch.autograd.grad(((initial_output - target) ** 2).sum() / 2, zero)[0]

    boundary_state = initial.clone()
    for index in range(boundary):
        boundary_state = boundary_state + branch(index, boundary_state)
    boundary_state = boundary_state.detach().requires_grad_(True)
    downstream = boundary_state
    for index in range(boundary, count):
        downstream = downstream + branch(index, downstream)
    boundary_adjoint = torch.autograd.grad(((downstream - target) ** 2).sum() / 2, boundary_state)[0]
    predicted = torch.stack([boundary_adjoint @ sources[index] for index in range(boundary)])
    gradient_error = (predicted - initial_gradient[:boundary]).abs().max().item()

    nonzero = torch.randn(count, generator=generator, dtype=dtype) * .2
    route_global = nonzero.clone().requires_grad_(True)
    route_frame = nonzero.clone().requires_grad_(True)
    output_global, output_frame = forward(route_global, False), forward(route_frame, True)
    gradient_global = torch.autograd.grad(output_global.square().sum(), route_global)[0]
    gradient_frame = torch.autograd.grad(output_frame.square().sum(), route_frame)[0]
    frame_error = (output_global - output_frame).abs().max().item()
    frame_gradient_error = (gradient_global - gradient_frame).abs().max().item()

    transform = torch.tensor([[1., 1.], [-1., 1.]], dtype=dtype)
    transform_sv = torch.linalg.svdvals(transform).tolist()
    history_gain = .7
    scaled_rotation = torch.tensor([[1., history_gain], [-history_gain, 1.]], dtype=dtype)
    boundary_norm = torch.linalg.matrix_norm(scaled_rotation, ord=2).item()
    boundary_condition = torch.linalg.cond(scaled_rotation).item()
    read = torch.tensor([1., 1.], dtype=dtype)
    write = read / 2
    jacobian_branch = torch.randn(width, width, generator=generator, dtype=dtype) * .1
    jacobian = torch.eye(2 * width, dtype=dtype) + torch.kron(write[:, None] @ read[None, :], jacobian_branch)
    expected_sv = torch.sort(torch.cat((torch.linalg.svdvals(torch.eye(width, dtype=dtype) + jacobian_branch), torch.ones(width, dtype=dtype))))[0]
    spectrum_error = (torch.sort(torch.linalg.svdvals(jacobian))[0] - expected_sv).abs().max().item()

    # Three noncollinear phases do NOT preserve arbitrary-source baseline kernels.
    base_three = torch.tensor([0., math.pi / 4, -math.pi / 4], dtype=dtype)
    address_three = torch.stack((base_three.cos(), base_three.sin()), dim=-1)
    scales_three = base_three.cos()
    failing_three = (address_three @ address_three.T) * scales_three[None, :] / scales_three[:, None]
    checks = {
        "causal_initial_kernel_is_one": kernel_error < 1e-12,
        "baseline_function_exact_in_frame": init_error == 0,
        "preboundary_history_kernel_first_derivative_is_one": abs(early_gradient - 1) < 1e-12,
        "aligned_tied_history_kernel_first_derivative_is_zero": tied_gradient == 0,
        "real_innovation_gradient_matches_boundary_adjoint": gradient_error < 1e-12,
        "coordinate_and_global_functions_agree": frame_error < 1e-12,
        "coordinate_and_global_route_gradients_agree": frame_gradient_error < 1e-12,
        "frozen_step_spectrum_matches_residual_plus_identity": spectrum_error < 1e-12,
        "boundary_is_scaled_rotation_not_isometry": all(abs(s - math.sqrt(2)) < 1e-12 for s in transform_sv),
        "boundary_reaches_prescribed_history_gain_norm_lower_bound":
            abs(boundary_norm - math.sqrt(1 + history_gain**2)) < 1e-12,
        "boundary_introduces_no_directional_conditioning":
            abs(boundary_condition - 1) < 1e-12,
        "three_noncollinear_phases_break_baseline": abs(failing_three[2, 1].item() - 1) > .9,
    }
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU float64 fixed-source and nonlinear algebra; no model training, data download, GPU, or LM conclusion",
        "torch_version": torch.__version__,
        "seed": 931,
        "shape": {"branches": count, "width": width, "boundary": boundary},
        "checks": checks,
        "measurements": {
            "initial_causal_kernel_max_error": kernel_error,
            "frame_baseline_output_max_error": init_error,
            "prewriter_kernel_derivative": early_gradient,
            "aligned_tied_kernel_derivative": tied_gradient,
            "prewriter_loss_gradients": initial_gradient[:boundary].tolist(),
            "boundary_adjoint_gradient_max_error": gradient_error,
            "coordinate_output_max_error": frame_error,
            "coordinate_route_gradient_max_error": frame_gradient_error,
            "frozen_spectrum_max_error": spectrum_error,
            "boundary_singular_values": transform_sv,
            "boundary_fixed_history_gain": history_gain,
            "boundary_norm_at_fixed_history_gain": boundary_norm,
            "boundary_condition_at_fixed_history_gain": boundary_condition,
            "third_phase_second_source_coefficient": failing_three[2, 1].item(),
        },
        "conclusions_not_supported": ["LM quality advantage", "novelty", "whole-network stability", "GPU efficiency", "arbitrary-depth selective retrieval"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/phase_adjoint_20260929/algebra.json"))
    args = parser.parse_args()
    result = audit()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checks": result["checks"], "measurements": result["measurements"]}, ensure_ascii=False))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
