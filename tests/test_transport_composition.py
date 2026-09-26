import json
import unittest

import torch

from lm.transport_analysis import collect_transport_report, mean_zero_basis


class TransportCompositionTests(unittest.TestCase):
    def test_mean_complement_round_trip_survives_full_composition(self):
        # A quarter turn exchanges the mean and complement directions.  The
        # second turn returns the signal with sign -1, although each compressed
        # step U.T @ H @ U is zero.
        exchange = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        report = collect_transport_report([exchange, exchange])

        self.assertLess(report["prefix"][0]["composite_sv_max"], 1e-6)
        for summary in ("min", "mean", "max"):
            self.assertAlmostEqual(
                report["final"][f"composite_sv_{summary}"], 1.0, places=6,
            )
            self.assertLess(
                report["final"][f"projected_step_product_sv_{summary}"], 1e-10,
            )
        self.assertGreater(report["steps"][0]["mean_preservation_error"], 1.0)
        self.assertEqual(report["schema_version"], 2)
        json.dumps(report)

    def test_mean_preserving_mixers_match_historical_projected_composition(self):
        n = 4
        U = mean_zero_basis(n)
        mean_projector = torch.ones(n, n) / n
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        operators = [
            torch.eye(n),
            0.5 * torch.eye(n) + 0.5 * mean_projector,
            mean_projector + U @ rotation @ U.T,
        ]
        report = collect_transport_report(operators)
        historical_product = torch.eye(n - 1)

        for H, step, prefix in zip(operators, report["steps"], report["prefix"]):
            historical_product = (U.T @ H @ U) @ historical_product
            s = torch.linalg.svdvals(historical_product)
            for summary, expected in (("min", s.min()), ("mean", s.mean()), ("max", s.max())):
                self.assertAlmostEqual(
                    prefix[f"composite_sv_{summary}"], expected.item(), places=5,
                )
                self.assertAlmostEqual(
                    prefix[f"projected_step_product_sv_{summary}"],
                    expected.item(), places=6,
                )
            self.assertLess(step["mean_preservation_error"], 1e-6)

    def test_noncommuting_prefixes_use_forward_transport_order(self):
        first = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        second = torch.tensor([[1.0, 0.0], [3.0, 2.0]])
        U = mean_zero_basis(2)
        report = collect_transport_report([first, second])
        expected = torch.linalg.svdvals(U.T @ (second @ first) @ U).item()
        reversed_order = torch.linalg.svdvals(U.T @ (first @ second) @ U).item()

        self.assertGreater(abs(expected - reversed_order), 0.1)
        self.assertAlmostEqual(report["final"]["composite_sv_mean"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
