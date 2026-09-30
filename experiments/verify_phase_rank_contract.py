"""Independent CPU audit of the reciprocal-scale reference-address contract.

The audit uses NumPy float64 and imports no candidate model. It checks the
all-past, positive-scale, unit-address, no-transport initialization interface:
X0 = e1*x0; read c_t^T X/a_t; write a_t*c_t*delta; a_t = c_t dot e1 > 0.
The embedding is the first reference address, and one serialized final reader
may be included. This is algebra verification, not an n-stream LM, training,
GPU evaluation, novelty claim, or a universal HC memory/state lower bound.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TOLERANCE = 1e-12
SEED = 20260929


def max_abs(value):
    value = np.asarray(value)
    return float(np.max(np.abs(value))) if value.size else 0.0


def array_sha256(value):
    value = np.ascontiguousarray(value)
    metadata = json.dumps({"shape": value.shape, "dtype": str(value.dtype)}, sort_keys=True)
    return hashlib.sha256(metadata.encode() + value.tobytes()).hexdigest()


def chain_gram(scales):
    indices = np.arange(len(scales))
    earlier = np.minimum.outer(indices, indices)
    later = np.maximum.outer(indices, indices)
    return scales[later] / scales[earlier]


def kernel(addresses, scales):
    return (addresses @ addresses.T) * scales[None, :] / scales[:, None]


def construct_reference_addresses(scales, streams):
    """Fresh orthogonal innovations; the first address is the embedding e1."""
    scales = np.asarray(scales, dtype=np.float64)
    if (
        scales.ndim != 1 or not len(scales) or scales[0] != 1.0
        or np.any(scales <= 0) or np.any(np.diff(scales) > 0)
    ):
        raise ValueError("scales must start at 1 and be positive and nonincreasing")
    if len(np.unique(scales)) > streams:
        raise ValueError("distinct scale levels exceed the reference-address dimension")
    basis = np.eye(streams, dtype=np.float64)
    addresses = [basis[0].copy()]
    next_direction = 1
    for previous_scale, scale in zip(scales[:-1], scales[1:]):
        if scale == previous_scale:
            addresses.append(addresses[-1].copy())
        else:
            rho = float(scale / previous_scale)
            address = rho * addresses[-1] + math.sqrt(1 - rho * rho) * basis[next_direction]
            addresses.append(address)
            next_direction += 1
    return np.stack(addresses)


def nonlinear_baseline_audit(addresses, scales):
    """Sample nonlinear residual program; indices 0/last are input/output."""
    branches = len(scales) - 2
    width = 7
    generator = np.random.default_rng(SEED)
    initial = generator.normal(size=width)
    weights = generator.normal(size=(branches, width, width)) * 0.2
    biases = generator.normal(size=(branches, width)) * 0.1

    def branch(index, state):
        normed = state / math.sqrt(float(np.mean(state * state)) + 1e-6)
        return 0.2 * np.tanh(weights[index] @ normed + biases[index]) + 0.04 * np.sin(normed)

    state = np.outer(addresses[0], initial)
    baseline = initial.copy()
    input_errors, delta_errors, after_update_errors = [], [], []
    for index in range(branches):
        address_index = index + 1
        c, a = addresses[address_index], scales[address_index]
        read = c @ state / a
        input_errors.append(max_abs(read - baseline))
        delta, baseline_delta = branch(index, read), branch(index, baseline)
        delta_errors.append(max_abs(delta - baseline_delta))
        state = state + a * c[:, None] * delta[None, :]
        baseline = baseline + baseline_delta
        after_update_errors.append(max_abs(c @ state / a - baseline))
    output = addresses[-1] @ state / scales[-1]
    return {
        "seed": SEED,
        "branches": branches,
        "width": width,
        "branch": "shared RMS normalization then 0.2*tanh(W@z+b)+0.04*sin(z)",
        "input_sha256": array_sha256(initial),
        "weights_sha256": array_sha256(weights),
        "biases_sha256": array_sha256(biases),
        "branch_input_max_errors": input_errors,
        "branch_delta_max_errors": delta_errors,
        "active_read_after_update_max_errors": after_update_errors,
        "final_reader_max_error": max_abs(output - baseline),
    }


def audit():
    # Embedding at 0; branch readers/writers at 1..5; final reader at 6.
    scales = np.array([1.0, 1.0, 0.8, 0.8, 0.5, 0.5, 0.5], dtype=np.float64)
    addresses = construct_reference_addresses(scales, streams=3)
    gram = addresses @ addresses.T
    causal = np.tril(np.ones_like(gram, dtype=bool), k=-1)
    unit_error = max_abs(np.sum(addresses * addresses, axis=1) - 1)
    anchor_error = max_abs(addresses[:, 0] - scales)
    gram_error = max_abs(gram - chain_gram(scales))
    causal_error = max_abs(kernel(addresses, scales)[causal] - 1)
    distinct_levels = len(np.unique(scales))
    gram_rank = int(np.linalg.matrix_rank(gram, tol=TOLERANCE))
    repeated_error = max(
        max_abs(addresses[i] - addresses[j])
        for i in range(len(scales)) for j in range(len(scales)) if scales[i] == scales[j]
    )

    innovation_records = []
    for index in range(1, len(scales)):
        rho = float(scales[index] / scales[index - 1])
        innovation = addresses[index] - rho * addresses[index - 1]
        innovation_records.append({
            "index": index,
            "rho": rho,
            "strict_drop": bool(scales[index] < scales[index - 1]),
            "history_inner_product_max_error": max_abs(addresses[:index] @ innovation),
            "innovation_norm_squared": float(innovation @ innovation),
            "expected_norm_squared": 1 - rho * rho,
            "norm_squared_error": abs(float(innovation @ innovation) - (1 - rho * rho)),
            "address_prefix_rank": int(np.linalg.matrix_rank(addresses[:index + 1], tol=TOLERANCE)),
            "expected_prefix_rank": len(np.unique(scales[:index + 1])),
        })
    innovation_error = max(row["history_inner_product_max_error"] for row in innovation_records)
    innovation_norm_error = max(row["norm_squared_error"] for row in innovation_records)

    queries = addresses / scales[:, None]
    brownian_times = 1 / (scales * scales)
    earlier_indices = np.minimum.outer(np.arange(len(scales)), np.arange(len(scales)))
    query_gram_error = max_abs(queries @ queries.T - brownian_times[earlier_indices])
    query_innovation_errors, query_norm_errors = [], []
    for index in range(1, len(scales)):
        increment = queries[index] - queries[index - 1]
        query_innovation_errors.append(max_abs(queries[:index] @ increment))
        query_norm_errors.append(float(abs(
            float(increment @ increment) - (brownian_times[index] - brownian_times[index - 1])
        )))

    # Three positive serialized levels give a rank-three target Gram, not a
    # realizable two-dimensional unit-address Gram in the specified interface.
    impossible_scales = np.array([1.0, 0.8, 0.5], dtype=np.float64)
    impossible_gram = chain_gram(impossible_scales)
    impossible_rank = int(np.linalg.matrix_rank(impossible_gram, tol=TOLERANCE))
    impossible_eigenvalues = np.linalg.eigvalsh(impossible_gram)
    impossible_det = float(np.linalg.det(impossible_gram))
    construction_rejected = False
    try:
        construct_reference_addresses(impossible_scales, streams=2)
    except ValueError:
        construction_rejected = True

    # Positivity is necessary for counting signed a values as distinct levels.
    negative_addresses = np.array([[1.0], [-1.0], [1.0], [-1.0]], dtype=np.float64)
    negative_scales = negative_addresses[:, 0]
    negative_gram = negative_addresses @ negative_addresses.T
    negative_causal = np.tril(np.ones_like(negative_gram, dtype=bool), k=-1)
    negative_error = max_abs(kernel(negative_addresses, negative_scales)[negative_causal] - 1)
    negative_rank = int(np.linalg.matrix_rank(negative_gram, tol=TOLERANCE))
    negative_distinct = len(np.unique(negative_scales))

    # Terminal readers are parallel, never sources for one another. The
    # serialized all-pairs Gram constraint cannot be imposed between them.
    writers = np.array([[1.0, 0.0]] * 3, dtype=np.float64)
    readers = np.array([[0.8, 0.6], [0.8, -0.6], [0.6, 0.8]], dtype=np.float64)
    writer_scales, reader_scales = writers[:, 0], readers[:, 0]
    parallel_kernel = (readers @ writers.T) * writer_scales[None, :] / reader_scales[:, None]
    parallel_addresses = np.concatenate((writers, readers), axis=0)
    parallel_scales = parallel_addresses[:, 0]
    parallel_rank = int(np.linalg.matrix_rank(parallel_addresses @ parallel_addresses.T, tol=TOLERANCE))
    parallel_distinct = len(np.unique(parallel_scales))
    parallel_error = max_abs(parallel_kernel - 1)
    parallel_unit_error = max_abs(np.sum(parallel_addresses**2, axis=1) - 1)
    parallel_equal_level_direction_gap = float(np.linalg.norm(readers[0] - readers[1]))
    unrequired_reader_kernel = kernel(readers, reader_scales)
    reader_lower = np.tril(np.ones((len(readers), len(readers)), dtype=bool), k=-1)
    unrequired_reader_error = max_abs(unrequired_reader_kernel[reader_lower] - 1)

    # Approximate equality does not imply the exact rank/level identity.
    epsilon = 1e-3
    approximate_tolerance = 1e-6
    angles = np.linspace(0.0, epsilon, 17, dtype=np.float64)
    approximate_addresses = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    approximate_scales = approximate_addresses[:, 0]
    approximate_gram = approximate_addresses @ approximate_addresses.T
    approximate_causal = np.tril(np.ones_like(approximate_gram, dtype=bool), k=-1)
    approximate_error = max_abs(kernel(approximate_addresses, approximate_scales)[approximate_causal] - 1)
    approximate_rank = int(np.linalg.matrix_rank(approximate_gram, tol=TOLERANCE))
    approximate_distinct = len(np.unique(approximate_scales))
    approximate_bound = math.sin(epsilon) * math.tan(epsilon)
    approximate_unit_error = max_abs(np.sum(approximate_addresses**2, axis=1) - 1)

    nonlinear = nonlinear_baseline_audit(addresses, scales)
    checks = {
        "n3_unit_addresses": unit_error < TOLERANCE,
        "n3_positive_nonincreasing_scales_equal_anchor_projection":
            bool(np.all(scales > 0) and np.all(np.diff(scales) <= 0)) and anchor_error < TOLERANCE,
        "n3_all_causal_coefficients_including_input_and_final_reader_equal_one": causal_error < TOLERANCE,
        "n3_gram_has_markov_ratio_form": gram_error < TOLERANCE,
        "n3_gram_rank_equals_distinct_positive_levels": gram_rank == distinct_levels == 3,
        "n3_repeated_level_directions_identical": repeated_error == 0,
        "n3_new_innovations_orthogonal_to_entire_history": innovation_error < TOLERANCE,
        "n3_innovation_norms_match_one_minus_rho_squared": innovation_norm_error < TOLERANCE,
        "n3_every_prefix_rank_matches_distinct_levels":
            all(row["address_prefix_rank"] == row["expected_prefix_rank"] for row in innovation_records),
        "query_gram_has_brownian_min_time_form": query_gram_error < TOLERANCE,
        "query_increments_orthogonal_to_history": max(query_innovation_errors) < TOLERANCE,
        "query_increment_norms_match_time_increments": max(query_norm_errors) < TOLERANCE,
        "n2_three_distinct_levels_target_gram_has_rank_three":
            impossible_rank == 3 and float(impossible_eigenvalues.min()) > TOLERANCE and impossible_det > 0,
        "n2_three_distinct_levels_constructor_rejected": construction_rejected,
        "negative_scale_counterexample_keeps_exact_causal_kernel": negative_error < TOLERANCE,
        "negative_scale_counterexample_refutes_signed_level_count_and_monotonicity":
            negative_rank == 1 and negative_distinct == 2 and bool(np.any(np.diff(negative_scales) > 0)),
        "parallel_readers_all_true_source_coefficients_equal_one": parallel_error < TOLERANCE,
        "parallel_counterexample_keeps_positive_unit_addresses":
            bool(np.all(parallel_scales > 0)) and parallel_unit_error < TOLERANCE,
        "parallel_readers_have_same_scale_different_directions":
            reader_scales[0] == reader_scales[1] and parallel_equal_level_direction_gap > 1,
        "parallel_readers_have_more_levels_than_address_dimension": parallel_distinct > parallel_rank == 2,
        "parallel_counterexample_violates_only_unrequired_reader_to_reader_edges": unrequired_reader_error > 0.5,
        "approximate_two_dimensional_kernel_has_many_distinct_levels":
            approximate_distinct == 17 and approximate_rank == 2,
        "approximate_counterexample_keeps_positive_unit_nonincreasing_addresses":
            bool(np.all(approximate_scales > 0) and np.all(np.diff(approximate_scales) <= 0))
            and approximate_unit_error < TOLERANCE,
        "approximate_counterexample_is_close_but_not_exact":
            TOLERANCE < approximate_error < approximate_tolerance,
        "approximate_counterexample_respects_analytic_error_bound": approximate_error <= approximate_bound + TOLERANCE,
        "sample_nonlinear_branches_have_baseline_inputs_and_updates":
            max(nonlinear["branch_input_max_errors"] + nonlinear["branch_delta_max_errors"]
                + nonlinear["active_read_after_update_max_errors"]) < TOLERANCE,
        "sample_nonlinear_final_reader_matches_baseline": nonlinear["final_reader_max_error"] < TOLERANCE,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent CPU float64 reference-address algebra and sample nonlinear residual program; no candidate model import, LM implementation/training, data download, or GPU use",
        "comparison_class": {
            "addresses": "real Euclidean unit reference vectors in n stream dimensions",
            "initial_state": "X0=e1*x0, counted as the first reference address",
            "scales": "a_t=c_t dot e1 > 0",
            "read_write": "read=c_t^T X/a_t; write=a_t*c_t*delta; no transport",
            "source_contract": "exact coefficients for every past common branch source, not just adjacent sources",
            "output": "one serialized terminal reader may be included; parallel readers have no reader-to-reader source edges",
            "rank_tolerance": TOLERANCE,
            "algebra_tolerance": TOLERANCE,
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dtype": "float64",
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "measurements": {
            "n3": {
                "streams": 3, "scales": scales.tolist(), "addresses": addresses.tolist(),
                "embedding_index": 0, "branch_indices": list(range(1, len(scales) - 1)),
                "final_reader_index": len(scales) - 1,
                "unit_max_error": unit_error, "anchor_projection_max_error": anchor_error,
                "causal_kernel_max_error": causal_error, "gram_formula_max_error": gram_error,
                "gram_rank": gram_rank, "distinct_levels": distinct_levels,
                "gram_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
                "repeated_direction_max_error": repeated_error,
                "innovations": innovation_records,
                "brownian_times": brownian_times.tolist(), "query_gram_max_error": query_gram_error,
                "query_increment_history_max_errors": query_innovation_errors,
                "query_increment_norm_squared_errors": query_norm_errors,
            },
            "n2_unrealizable_target": {
                "scales": impossible_scales.tolist(), "target_gram": impossible_gram.tolist(),
                "target_rank": impossible_rank, "target_eigenvalues": impossible_eigenvalues.tolist(),
                "target_determinant": impossible_det, "constructor_rejected": construction_rejected,
                "scope": "impossibility only for this exact positive-scale reference-address interface",
            },
            "negative_scale_counterexample": {
                "relaxed_assumption": "positive a_t", "addresses": negative_addresses.tolist(),
                "scales": negative_scales.tolist(), "causal_kernel_max_error": negative_error,
                "gram_rank": negative_rank, "distinct_signed_levels": negative_distinct,
            },
            "parallel_reader_counterexample": {
                "relaxed_assumption": "serialized all-past source contract between every pair of addresses",
                "writer_addresses_including_embedding": writers.tolist(), "reader_addresses": readers.tolist(),
                "source_to_reader_kernel": parallel_kernel.tolist(), "source_kernel_max_error": parallel_error,
                "combined_gram_rank": parallel_rank, "distinct_positive_levels": parallel_distinct,
                "unit_max_error": parallel_unit_error,
                "same_scale_direction_gap": parallel_equal_level_direction_gap,
                "unrequired_reader_to_reader_kernel": unrequired_reader_kernel.tolist(),
                "unrequired_serial_kernel_max_error": unrequired_reader_error,
            },
            "approximate_kernel_counterexample": {
                "relaxed_assumption": "exact all-past coefficient equality",
                "streams": 2, "angles": angles.tolist(), "scales": approximate_scales.tolist(),
                "distinct_levels": approximate_distinct, "gram_rank": approximate_rank,
                "unit_max_error": approximate_unit_error,
                "causal_kernel_max_error": approximate_error,
                "declared_approximation_tolerance": approximate_tolerance,
                "analytic_error_upper_bound": approximate_bound,
            },
            "sample_nonlinear_residual_program": nonlinear,
        },
        "conclusions_not_supported": [
            "language-model quality or training advantage", "GPU efficiency",
            "whole-network Jacobian stability", "universal HC state or task-memory lower bound",
            "rank equals level count under approximate coefficients or parallel terminal readers",
            "dynamic-router derivative claims from frozen reference-address rank alone",
            "novelty of Markov/Brownian Gram identities or linear algebra",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/phase_adjoint_20260929/phase_rank.json"),
    )
    args = parser.parse_args()
    result = audit()
    source = Path(__file__).resolve()
    result["source_path"] = str(source.relative_to(source.parents[1]))
    result["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "all_checks_passed": result["all_checks_passed"],
        "checks_passed": sum(result["checks"].values()),
        "checks_total": len(result["checks"]),
        "n3_causal_kernel_max_error": result["measurements"]["n3"]["causal_kernel_max_error"],
        "n3_query_gram_max_error": result["measurements"]["n3"]["query_gram_max_error"],
        "nonlinear_final_reader_max_error": result["measurements"]["sample_nonlinear_residual_program"]["final_reader_max_error"],
        "source_sha256": result["source_sha256"],
    }, ensure_ascii=False, allow_nan=False))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
