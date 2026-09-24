"""Verify: static fixed-vector-orthogonal HC (IsoHC) == identity-HC after a
change of read/write coordinates, on the repo's TwoBranchHCTransformer.

X_k = G_k Z_k with G_{k+1} = H_k G_k  =>  identity transport in Z with
  read  a~_k = G_k^T a_k,   write b~_k = G_{k+1}^{-1} b_k = G_{k+1}^T b_k.
Because G fixes 1 on both sides, a~ = 1 + lam * G^T P_perp w, which is
again of the repo's form 1 + lam * P_perp w~ with w~ = G^T P_perp w.
"""
import sys, torch, copy
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from lm.models import TwoBranchHCTransformer
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
cfg = dict(vocab_size=97, d_model=32, num_layers=6, num_heads=4, n_streams=4,
           context_length=16, use_flash=False, lambda_a=0.7, lambda_b=0.9)
iso = TwoBranchHCTransformer(mixing_type="isohc", ns_steps=30, svd_fallback=True, **cfg).double()
# push each IsoHC raw matrix far from identity so the rotations are large
with torch.no_grad():
    for m in list(iso.attn_mixings) + list(iso.mlp_mixings):
        m.H_raw.copy_(torch.randn(4, 4))
    for p in iso.parameters():
        if p.dim() >= 1 and p.numel() == 4:  # read/write/final vectors
            p.add_(torch.randn_like(p))
ident = TwoBranchHCTransformer(mixing_type="identity", **cfg).double()
ident.load_state_dict({k: v for k, v in iso.state_dict().items()
                       if not ('mixings' in k)}, strict=False)
n = 4
def zm(v):
    return v - v.mean()
G = torch.eye(n)
with torch.no_grad():
    for l in range(cfg['num_layers']):
        for branch in ('attn', 'mlp'):
            H = getattr(iso, f'{branch}_mixings')[l]().double()
            r = getattr(iso, f'{branch}_readout_weights')[l]
            w = getattr(iso, f'{branch}_injection_weights')[l]
            getattr(ident, f'{branch}_readout_weights')[l].copy_(G.T @ zm(r))
            G = H @ G
            getattr(ident, f'{branch}_injection_weights')[l].copy_(G.T @ zm(w))
    ident.readout_final.copy_(G.T @ zm(iso.readout_final))
x = torch.randint(0, 97, (3, 16))
li, _ = iso(x); lz, _ = ident(x)
orth = max((m().double().T @ m().double() - torch.eye(n)).abs().max().item()
           for m in list(iso.attn_mixings) + list(iso.mlp_mixings))
angle = max(torch.linalg.matrix_norm(m().double() - torch.eye(n)).item()
            for m in list(iso.attn_mixings) + list(iso.mlp_mixings))
print(f'max |Q^TQ-I| = {orth:.2e}; max ||Q-I||_F = {angle:.3f} (large rotations)')
print(f'max |logits_IsoHC - logits_identityHC(transformed)| = {(li-lz).abs().max().item():.3e}')
print(f'max |logits| = {li.abs().max().item():.3f}')
# Control: identity-HC WITHOUT the transform differs (so the check is not vacuous)
ident2 = TwoBranchHCTransformer(mixing_type="identity", **cfg).double()
ident2.load_state_dict({k: v for k, v in iso.state_dict().items() if 'mixings' not in k}, strict=False)
l2, _ = ident2(x)
print(f'control (no transform): max diff = {(li-l2).abs().max().item():.3e}')
