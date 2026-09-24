"""Does AdamW weight decay alone explain the trained static-Birkhoff spectrum?

Replays the exact LR schedule of each recorded run, applies only decoupled
weight decay (p <- p * (1 - lr_t * wd)) to the MHCMixing logits from their
initialization, and compares the resulting 1_perp singular values with the
values recorded in the run's run_summary.json.
"""
import json
import math
import os
import sys

import torch

REPO = os.environ.get("ISOHC_REPO", str(__import__("pathlib").Path(__file__).resolve().parents[3]))
sys.path.insert(0, REPO)
from lm.mixing import MHCMixing  # noqa: E402
from isohc.projection import construct_orthogonal_complement  # noqa: E402

RUNS = {
    "48L FineWeb-Edu (0525)": "docs/0605_alldoc/0525_results_raw/0525_fe_fair_deep48_p33013_20m/fe-deep-48l-512_mhc_seed0/run_summary.json",
    "24L TinyStories (0524)": "docs/0605_alldoc/0524_results_raw/0524_deep_stress_512/deep-stress-512_mhc_seed0/run_summary.json",
}


def decay_factor(cfg):
    tokens_per_step = cfg["batch_size"] * cfg["context_length"] * cfg.get("grad_accum_steps", 1)
    total = math.ceil(cfg["total_tokens"] / tokens_per_step)
    warm = cfg["warmup_tokens"] // tokens_per_step

    def lr(s):
        if s < warm:
            return cfg["max_lr"] * (s + 1) / warm
        p = (s - warm) / (total - warm)
        return cfg["min_lr"] + (cfg["max_lr"] - cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * p))

    f = 1.0
    for s in range(total):
        f *= 1 - lr(s) * cfg["weight_decay"]
    return f, total


def main():
    n = 4
    U = construct_orthogonal_complement(n, device="cpu", dtype=torch.float64)
    for name, rel in RUNS.items():
        d = json.load(open(os.path.join(REPO, rel)))
        cfg, meas = d["config"], d["posthoc"]["h_diagnostics"]
        f, steps = decay_factor(cfg)
        torch.manual_seed(0)
        svs, comp = [], torch.eye(n - 1, dtype=torch.float64)
        num_transports = 2 * cfg["num_layers"]
        with torch.no_grad():
            for _ in range(num_transports):
                m = MHCMixing(n)  # repo default: diag_bias=4, noise 0.01, 10 Sinkhorn iters
                m.logits.mul_(f)
                B = U.T @ m().double() @ U
                svs.append(torch.linalg.svdvals(B))
                comp = B @ comp
        svs = torch.stack(svs)
        e0 = math.exp(4.0)
        print(f"== {name}: {steps} steps, weight-decay factor on logits = {f:.4f}")
        print(f"   init 1_perp sv (analytic, diag_bias=4)  : {(e0 - 1) / (e0 + n - 1):.5f}")
        print(f"   weight-decay-only prediction  mean/min/max: {svs.mean():.5f} / {svs.min():.5f} / {svs.max():.5f}")
        print(f"   measured after training       mean/min/max: {meas['sv_mean_1perp_mean']:.5f} / "
              f"{meas['sv_min_1perp_mean']:.5f} / {meas['sv_max_1perp_mean']:.5f}")
        print(f"   predicted composite gain over {num_transports} transports: "
              f"{torch.linalg.svdvals(comp).mean().item():.3e}")


if __name__ == "__main__":
    main()
