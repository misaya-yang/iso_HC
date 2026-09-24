import json, glob, sys
rows = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + '/*_s*.json' if len(sys.argv) > 1 else '../cpu_probe_runs/*_s*.json'))]
cols = [('method', 13), ('lam', 5), ('diag_bias', 4), ('scaled_init', 5), ('seed', 4), ('val_loss', 7), ('clamp_all_meanonly', 8), ('remove_perp_mid', 8),
        ('replace_H_identity', 8), ('lambda_read', 6), ('lambda_write', 6), ('composite_mean_gain', 7), ('step_perp_sv_mean', 7),
        ('step_dist_identity', 7), ('step_mean_leak', 7), ('final_perp_norm_ratio', 6)]
short = {'clamp_all_meanonly': 'clampAll', 'remove_perp_mid': 'rmMid', 'replace_H_identity': 'H->I', 'lambda_read': 'lamR',
         'lambda_write': 'lamW', 'composite_mean_gain': 'mGainK', 'step_perp_sv_mean': 'svPerp', 'step_dist_identity': 'dist(I)',
         'step_mean_leak': 'leak', 'final_perp_norm_ratio': 'perp%', 'diag_bias': 'db', 'val_loss': 'val', 'scaled_init': 'scInit'}
print(' '.join(f"{short.get(c, c):>{w}}" for c, w in cols))
rows.sort(key=lambda d: (d['lam'], bool(d.get('scaled_init')), d['method'], d.get('diag_bias', 0), d['seed']))
for d in rows:
    out = []
    for c, w in cols:
        v = d.get(c)
        out.append(f"{v:>{w}.4f}" if isinstance(v, float) else f"{str(v):>{w}}")
    print(' '.join(out))
