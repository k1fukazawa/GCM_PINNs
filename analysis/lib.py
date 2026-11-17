#!/usr/bin/env python3
"""Loader of the evaluation results (results/) and helper functions for the tables and figures.
Set the environment variable RESULTS to use another results directory.
"""
import json, glob, re, statistics as st, os
R = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')) + '/'
NAMES = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M2+S', 'M3+S', 'M4+S', 'M5+S', 'M2-P0', 'M2-P1', 'M2-P10', 'M5-P0', 'M2-2E', 'M5-2E', 'M5-4E', 'M5-8E', 'M5-P0-8E', 'M2-SZA', 'M5-SZA']
MAP = {n: n for n in NAMES}
E = {}
Rj = {}
for g in NAMES:
    for s in range(3):
        p = R + f'{g}_seed{s}/results_extra3.json'
        if os.path.exists(p):
            E[g, s] = json.load(open(p))
            Rj[g, s] = json.load(open(R + f'{g}_seed{s}/results.json'))

def seeds(g):
    return [s for s in range(3) if (g, s) in E]

def ser(g, s, h, v, key='rmse', src='pred'):
    return E[g, s]['native'][h][v][src]['series'][key]

def base(h, v, key='rmse', src='persist'):
    return E['M2', 0]['native'][h][v][src]['series'][key]

def vals(g, h, v, L, key='rmse'):
    return [ser(g, s, h, v, key)[L - 1] for s in seeds(g)]

def ms(xs, f='%.2f', sd=True):
    xs = list(xs)
    return f % st.mean(xs) + (' ± ' + f % st.stdev(xs) if sd and len(xs) > 1 else '')

def lastbelow(g, h, v, src):
    m = [st.mean([ser(g, s, h, v)[i] for s in seeds(g)]) for i in range(143)]
    b = base(h, v, 'rmse', src)
    for i in range(143):
        if m[i] >= b[i]:
            return i
    return 143

def psbias(g, s):
    d = E[g, s]['drift_series']
    t = d['truth'][-1]
    return ((d['pred'][-1] - t) / 128, (d['pred_nofix'][-1] - t) / 128 if d.get('pred_nofix') else None)
D = json.load(open(R + 'baselines/diurnal.json'))

def dser(h, v, b, key='rmse'):
    return D['native'][h][v][b][key]

def lastbelow2(g, h, v, b):
    m = [st.mean([ser(g, s, h, v)[i] for s in seeds(g)]) for i in range(143)]
    bb = dser(h, v, b)
    for i in range(143):
        if m[i] >= bb[i]:
            return i
    return 143
import numpy as np
RUN = {k: v for k, v in MAP.items()}

def fields(g, s=0):
    return np.load(R + f'{g}_seed{s}/fields.npz')

def damped(h, v, lev_i):
    f = fields('M5', 0)
    d0 = f['init_' + v][lev_i].astype(float) - f['clim_' + v][lev_i].astype(float)
    n0 = (d0 ** 2).mean()
    C = np.array(dser(h, v, 'clim'))
    P = np.array(dser(h, v, 'persist'))
    ip = (n0 + C ** 2 - P ** 2) / 2
    ip = np.maximum(ip, 0)
    return (np.sqrt(np.maximum(C ** 2 - ip ** 2 / n0, 0)), ip / n0)

def meanser(g, h, v, key='rmse'):
    return np.mean([ser(g, s, h, v, key) for s in seeds(g)], axis=0)

def minser(g, h, v):
    return np.min([ser(g, s, h, v) for s in seeds(g)], axis=0)

def maxser(g, h, v):
    return np.max([ser(g, s, h, v) for s in seeds(g)], axis=0)
