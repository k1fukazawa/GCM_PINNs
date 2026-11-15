#!/usr/bin/env python3
"""Statistical baselines that retain the diurnal cycle (no training; a few minutes).

Evaluated at the same levels and with the same metrics as evaluate.py:
  persist : the initial state held constant
  clim    : time-mean field of the training period
  dclim   : mean field of the training period for each UT hour (diurnal climatology)
  lag24   : latest truth at the same UT at or before the initial time (24-h-lagged persistence)
Output: <out>/diurnal.json
"""
import argparse, json, math, os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='./npy')
ap.add_argument('--t_train', type=int, default=600)
ap.add_argument('--horizon', type=int, default=143)
ap.add_argument('--heights', type=float, nargs='+', default=[0.0, 100.0, 120.0, 150.0, 200.0, 300.0])
ap.add_argument('--out', default='./results/baselines')
args = ap.parse_args()
gu = np.load(f'{args.dir}/gu.npy', mmap_mode='r')
gv = np.load(f'{args.dir}/gv.npy', mmap_mode='r')
gt = np.load(f'{args.dir}/gt.npy', mmap_mode='r')
gp = np.load(f'{args.dir}/gp.npy', mmap_mode='r')
gz = np.load(f'{args.dir}/multilayer_gz.npy', mmap_mode='r')
H, LEV = (gu.shape[0], gu.shape[1])
rs = args.t_train
horizon = min(args.horizon, H - rs - 1)
assert rs >= 24 and horizon > 0
lev_idx = np.unique(np.round(np.linspace(0, LEV - 1, 48))).astype(int)
Z = np.asarray(gz[:, lev_idx, :, :].mean(axis=(0, 2, 3)), dtype=np.float64)
ks = [int(np.argmin(np.abs(Z - h * 1000.0))) for h in args.heights]

def acc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / d if d > 0 else 0.0

def metrics(P, Tr, B0):
    out = {k: [] for k in ('rmse', 'bias', 'acc_clim')}
    for i in range(Tr.shape[0]):
        d = P[i] - Tr[i]
        out['rmse'].append(float(np.sqrt((d * d).mean())))
        out['bias'].append(float(d.mean()))
        out['acc_clim'].append(acc(P[i] - B0, Tr[i] - B0))
    return out

def series(get):
    A = np.stack([get(h) for h in range(0, rs + horizon + 1)])
    clim = A[:rs].mean(axis=0)
    dcl = np.stack([A[np.arange(u, rs, 24)].mean(axis=0) for u in range(24)])
    Tr = A[rs + 1:rs + horizon + 1]
    leads = np.arange(1, horizon + 1)
    P = {'persist': np.broadcast_to(A[rs], Tr.shape), 'clim': np.broadcast_to(clim, Tr.shape), 'dclim': np.stack([dcl[(rs + l) % 24] for l in leads]), 'lag24': np.stack([A[rs + l - 24 * int(math.ceil(l / 24.0))] for l in leads])}
    return {k: metrics(v, Tr, clim) for k, v in P.items()}
res = {'t_train': rs, 'horizon': horizon, 'leads_h': list(range(1, horizon + 1)), 'native': {}, 'note': 'dclim = mean field of the training period for each UT hour; lag24 = latest truth at the same UT at or before the initial time'}
for h, k in zip(args.heights, ks):
    ent = {'lev': int(k), 'height_km': float(Z[k] / 1000.0)}
    for var, g in (('u', gu), ('v', gv), ('T', gt)):
        kk = int(lev_idx[k])
        ent[var] = series(lambda t, g=g, kk=kk: np.asarray(g[t][kk], dtype=np.float64))
    if h == args.heights[0]:

        def getp(t):
            p = np.asarray(gp[t], dtype=np.float64)
            return p / 100.0 if p.mean() > 2000 else p
        ent['Ps'] = series(getp)
    res['native'][f'{h:.0f}km'] = ent
    print(f'[i] {h:.0f} km (k={k}, {Z[k] / 1000:.1f} km): RMSE of u at t+12 h  ' + '  '.join((f"{b}={ent['u'][b]['rmse'][min(11, horizon - 1)]:.1f}" for b in ('persist', 'clim', 'dclim', 'lag24'))))
os.makedirs(args.out, exist_ok=True)
json.dump(res, open(f'{args.out}/diurnal.json', 'w'), ensure_ascii=False, indent=1)
print(f'[i] saved {args.out}/diurnal.json')
