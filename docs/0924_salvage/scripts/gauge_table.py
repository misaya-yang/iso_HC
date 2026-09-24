"""Gauge table for static HC on the repo's TwoBranchHCTransformer.
X_k = G_k Z_k, G_{k+1} = H_k G_k:  read a~ = G_k^T a,  write b~ = G_{k+1}^{-1} b.
Checks (i) function equality with identity transport when reads/writes are FREE,
(ii) whether transformed vectors stay inside the repo class a = 1 + lam*P_perp w,
(iii) how large the transformed write vectors get (conditioning cost)."""
import sys, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
import lm.models as M
torch.set_default_dtype(torch.float64); torch.manual_seed(1)
n, L = 4, 6
cfg = dict(vocab_size=97, d_model=32, num_layers=L, num_heads=4, n_streams=n, context_length=16,
           use_flash=False, lambda_a=0.7, lambda_b=0.9)

class FreeVecHC(M.TwoBranchHCTransformer):
    """identity transport with arbitrary (free) read/write vectors stored in buffers"""
    def set_free(self, reads, writes, final):
        self._reads, self._writes, self._final = reads, writes, final
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        X = (self.token_embedding(input_ids) + self.pos_embedding(torch.arange(T))).unsqueeze(0) + self.stream_embed.unsqueeze(1)
        i = 0
        for l in range(self.num_layers):
            for br in ('attn', 'mlp'):
                a, b = self._reads[i], self._writes[i]
                z = torch.einsum('s,sbtd->btd', a, X) / n
                blk = self.attns[l] if br == 'attn' else self.mlps[l]
                nrm = self.attn_norms[l] if br == 'attn' else self.mlp_norms[l]
                X = X + b.view(n, 1, 1, 1) * blk(nrm(z)).unsqueeze(0)
                i += 1
        z = torch.einsum('s,sbtd->btd', self._final, X) / n
        return self.lm_head(self.norm_final(z)), None

def vecs(model):
    lam_a, lam_b = model.readout_lambda, model.injection_lambda
    mk = lambda w, lam: torch.ones(n) + lam * (w - w.mean())
    reads, writes, Hs = [], [], []
    for l in range(L):
        for br in ('attn', 'mlp'):
            reads.append(mk(getattr(model, f'{br}_readout_weights')[l], lam_a))
            writes.append(mk(getattr(model, f'{br}_injection_weights')[l], lam_b))
            Hs.append(getattr(model, f'{br}_mixings')[l]().double())
    return reads, writes, Hs, mk(model.readout_final, model.readout_final_lambda)

x = torch.randint(0, 97, (2, 16))
for kind in ('isohc', 'mhc', 'orthogonal', 'unconstrained'):
    torch.manual_seed(2)
    mk = {'mhc': dict(diag_bias=2.0, noise_std=0.3)} .get(kind, {})
    src = M.TwoBranchHCTransformer(mixing_type=kind, ns_steps=30, svd_fallback=True, mixing_kwargs=mk, **cfg).double()
    with torch.no_grad():
        for m in list(src.attn_mixings) + list(src.mlp_mixings):
            if hasattr(m, 'H_raw'): m.H_raw.copy_(torch.eye(n) + 0.6 * torch.randn(n, n))
        for nm in ('attn_readout_weights', 'attn_injection_weights', 'mlp_readout_weights', 'mlp_injection_weights'):
            for p in getattr(src, nm): p.copy_(torch.randn(n))
    ref, _ = src(x)
    reads, writes, Hs, fin = vecs(src)
    G = torch.eye(n); ra, wb = [], []
    with torch.no_grad():
        for a, b, H in zip(reads, writes, Hs):
            ra.append(G.T @ a); G = H @ G; wb.append(torch.linalg.solve(G, b))
        fa = G.T @ fin
    dst = FreeVecHC(mixing_type='identity', **cfg).double()
    dst.load_state_dict({k: v for k, v in src.state_dict().items() if 'mixings' not in k}, strict=False)
    dst.set_free(ra, wb, fa)
    out, _ = dst(x)
    in_class = max(abs(v.sum().item() - n) for v in ra + wb + [fa])
    one = torch.ones(n); meanfix = max((H @ one - one).norm().item() + (one @ H - one).norm().item() for H in Hs)
    print(f'{kind:13s} max|H1-1|+|1H-1|={meanfix:.2e}  max|logit diff|={(out-ref).abs().max().item():.2e}  '
          f'max|1^T v~ - n| (0 => stays in repo class)={in_class:.2e}  max|b~|/max|b|={max(v.norm() for v in wb).item()/max(v.norm() for v in writes).item():.2f}')
