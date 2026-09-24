"""CPU mini version of the never-run P1 'accessibility' question.

Question: if the stream complement is coupled at O(1) (large read/write
gates), does transport geometry (identity / IsoHC / Birkhoff / unconstrained)
change LM loss, and is the complement causally used?

Uses the repo's TwoBranchHCTransformer unchanged. Char-level TinyShakespeare.
HC-specific parameters (mixings, gates, read/write vectors, stream embed) are
excluded from weight decay so the geometry is set by the loss, not by AdamW.
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from lm.models import TwoBranchHCTransformer, BaselineTransformer
from isohc.projection import construct_orthogonal_complement

# --- extra mixing arms (probe-only; repo untouched) ---
import math as _m
import torch.nn as _nn
import lm.models as _lmm
from lm.mixing import StreamMixing as _SM, create_mixing as _orig_create
class LeakyMeanMixing(_SM):
    """Lossy recency: mean gain g=2*sigmoid(s) (init 1), complement identity. Not isometric."""
    def __init__(self, n):
        super().__init__(n); self.s = _nn.Parameter(torch.zeros(()))
        self.register_buffer('P', torch.ones(n, n) / n)
    def forward(self):
        g = 2 * torch.sigmoid(self.s)
        return torch.eye(self.n_streams, device=self.P.device, dtype=self.P.dtype) - (1 - g) * self.P
class ExchangeMixing(_SM):
    """Isometric mean<->complement exchange: rotation by theta in span(e0, e1), e0 = 1/sqrt(n)."""
    def __init__(self, n):
        super().__init__(n); self.theta = _nn.Parameter(torch.zeros(()))
        e0 = torch.ones(n) / _m.sqrt(n); e1 = torch.zeros(n); e1[0] = 1.0; e1 = e1 - (e1 @ e0) * e0; e1 = e1 / e1.norm()
        self.register_buffer('e0', e0); self.register_buffer('e1', e1)
    def forward(self):
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        e0, e1 = self.e0, self.e1
        I = torch.eye(self.n_streams, device=e0.device, dtype=e0.dtype)
        return I + (c - 1) * (torch.outer(e0, e0) + torch.outer(e1, e1)) + s * (torch.outer(e1, e0) - torch.outer(e0, e1))
def _create(n, t, **kw):
    if t == 'leaky': return LeakyMeanMixing(n)
    if t == 'exchange': return ExchangeMixing(n)
    return _orig_create(n, t, **kw)
_lmm.create_mixing = _create

p = argparse.ArgumentParser()
p.add_argument('--method', required=True)  # baseline|identity|isohc|mhc|unconstrained
p.add_argument('--lam', type=float, default=0.01)
p.add_argument('--diag_bias', type=float, default=4.0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--layers', type=int, default=16)
p.add_argument('--d', type=int, default=64)
p.add_argument('--heads', type=int, default=4)
p.add_argument('--ctx', type=int, default=64)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--steps', type=int, default=2000)
p.add_argument('--lr', type=float, default=3e-3)
p.add_argument('--hc_wd', type=float, default=0.0)
p.add_argument('--out', required=True)
p.add_argument('--threads', type=int, default=1)
p.add_argument('--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tinyshakespeare.txt'),
               help='char-level corpus; download from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
p.add_argument('--scaled_init', action='store_true', help='GPT-2 style: output projections std 0.02/sqrt(2L)')
args = p.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)

text = open(args.data).read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data)); train, val = data[:split], data[split:]
V = len(chars)

def batch(src, gen, B):
    ix = torch.randint(len(src) - args.ctx - 1, (B,), generator=gen)
    x = torch.stack([src[i:i + args.ctx] for i in ix]); y = torch.stack([src[i + 1:i + args.ctx + 1] for i in ix])
    return x, y

if args.method == 'baseline':
    model = BaselineTransformer(V, args.d, args.layers, args.heads, args.ctx, use_flash=False)
else:
    mk = {}
    if args.method == 'mhc':
        mk = dict(diag_bias=args.diag_bias, noise_std=0.01)
    model = TwoBranchHCTransformer(V, args.d, args.layers, args.heads, 4, args.ctx,
                                   mixing_type=args.method, lambda_a=args.lam, lambda_b=args.lam,
                                   ns_steps=5, svd_fallback=False, use_flash=False, mixing_kwargs=mk)

if args.scaled_init:
    with torch.no_grad():
        f = 1.0 / math.sqrt(2 * args.layers)
        for name_, mod in model.named_modules():
            if name_.endswith('o_proj') or name_.endswith('mlp.proj') or (name_.startswith('mlps.') and name_.endswith('.proj')):
                mod.weight.mul_(f)
hc_names = ('mixings', 'lambda', 'readout_weights', 'injection_weights', 'readout_final', 'stream_embed')
decay, no_decay, hc = [], [], []
for n_, prm in model.named_parameters():
    if any(h in n_ for h in hc_names):
        hc.append(prm)
    elif prm.dim() >= 2:
        decay.append(prm)
    else:
        no_decay.append(prm)
groups = [dict(params=decay, weight_decay=0.1), dict(params=no_decay, weight_decay=0.0)]
if hc:
    groups.append(dict(params=hc, weight_decay=args.hc_wd))
opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
warm = 100
def lr_at(s):
    if s < warm: return args.lr * (s + 1) / warm
    q = (s - warm) / max(1, args.steps - warm)
    return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * q))

gen = torch.Generator().manual_seed(1000 + args.seed)
t0 = time.time()
for s in range(args.steps):
    for g in opt.param_groups: g['lr'] = lr_at(s)
    x, y = batch(train, gen, args.batch)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
train_time = time.time() - t0

@torch.no_grad()
def evaluate(**kw):
    g = torch.Generator().manual_seed(7)
    tot = 0.0; nb = 16
    for _ in range(nb):
        x, y = batch(val, g, 32)
        _, l = model(x, y, **kw) if kw else model(x, y)
        tot += l.item()
    return tot / nb

model.eval()
res = dict(vars(args)); res['train_time'] = train_time
res['val_loss'] = evaluate()
if args.method != 'baseline':
    S = 2 * args.layers  # states 1..S after each transport; 0 = embedding
    res['clamp_all_meanonly'] = evaluate(stream_intervention={'state_index': list(range(0, S + 1)), 'mode': 'mean_only'}) - res['val_loss']
    res['remove_perp_mid'] = evaluate(stream_intervention={'state_index': S // 2, 'mode': 'scale_perp', 'scale': 0.0}) - res['val_loss']
    res['replace_H_identity'] = evaluate(mixing_overrides='identity') - res['val_loss']
    res['lambda_read'] = model.readout_lambda.item(); res['lambda_write'] = model.injection_lambda.item()
    res['lambda_final'] = model.readout_final_lambda.item()
    U = construct_orthogonal_complement(4, device='cpu', dtype=torch.float64)
    ones = torch.ones(4, dtype=torch.float64)
    Hs = [m().detach().double() for m in list(model.attn_mixings) + list(model.mlp_mixings)]
    comp = torch.eye(3, dtype=torch.float64); meang = 1.0
    for H in Hs:  # order within composite not needed for sv of single steps; composite approx
        pass
    # composite in forward order (attn l, mlp l)
    order = []
    for l in range(args.layers):
        order += [model.attn_mixings[l]().detach().double(), model.mlp_mixings[l]().detach().double()]
    Ccomp = torch.eye(4, dtype=torch.float64)
    for H in order: Ccomp = H @ Ccomp
    res['composite_perp_sv'] = torch.linalg.svdvals(U.T @ Ccomp @ U).tolist()
    res['composite_mean_gain'] = (ones @ Ccomp @ ones / 4).item()
    res['step_perp_sv_mean'] = float(torch.stack([torch.linalg.svdvals(U.T @ H @ U) for H in order]).mean())
    res['step_dist_identity'] = float(sum(torch.linalg.matrix_norm(H - torch.eye(4, dtype=torch.float64)) for H in order) / len(order))
    res['step_mean_leak'] = float(sum((H @ ones - ones).norm() + (ones @ H - ones).norm() for H in order) / len(order))
    e0 = ones / 2.0  # 1/sqrt(4)
    res['per_transport'] = [dict(
        mean_gain=float(e0 @ H @ e0),
        mean_to_perp=float((U.T @ H @ e0).norm()),
        perp_to_mean=float((e0 @ H @ U).norm()),
        sv_min=float(torch.linalg.svdvals(H).min()), sv_max=float(torch.linalg.svdvals(H).max()),
        perp_sv=[float(v) for v in torch.linalg.svdvals(U.T @ H @ U)]) for H in order]
    # complement share of the final stream state
    x, _ = batch(val, torch.Generator().manual_seed(3), 16)
    _, _, states = model(x, return_stream_states=True)
    Xf = states[-1]; mean = Xf.mean(0, keepdim=True)
    res['final_perp_norm_ratio'] = ((Xf - mean).norm() / Xf.norm()).item()
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items() if k not in ('out',)}))
