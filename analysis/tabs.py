#!/usr/bin/env python3
"""Tables 3-7 of the paper. Run: python tabs.py
"""
from lib import *
from logs import get, LOG
V = ['u', 'v', 'T', 'Ps']

def mm(g, h, v, L, f='%.2f', key='rmse'):
    return ms(vals(g, h, v, L, key), f)

def m1(g, h, v, L, f='%.2f', key='rmse'):
    return f % st.mean(vals(g, h, v, L, key))

def sgn(x, f='%.1f'):
    return ('+' if x >= 0 else '−') + f % abs(x)

def rng(xs, f='%.1f'):
    return sgn(min(xs), f) + ' to ' + sgn(max(xs), f)

def bias_cell(g):
    b = [psbias(g, s) for s in seeds(g)]
    if b[0][1] is None:
        return ('–', rng([x[0] for x in b]))
    return (rng([x[0] for x in b]), rng([x[1] for x in b]))

def table3():
    rows = [['Persistence'] + ['%.2f' % base('0km', v)[11] for v in V] + ['%.2f' % base('0km', v)[142] for v in ['T', 'Ps']] + ['–', '–'], ['Climatology'] + ['%.2f' % base('0km', v, 'rmse', 'clim')[11] for v in V] + ['%.2f' % base('0km', v, 'rmse', 'clim')[142] for v in ['T', 'Ps']] + ['–', '–']]
    for g in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
        f = '%.1f' if g == 'M0' else '%.2f'
        a, b = bias_cell(g)
        rows.append([g] + [mm(g, '0km', v, 12, f) for v in V] + [mm(g, '0km', v, 143, '%.1f') for v in ['T', 'Ps']] + [a, b])
    return rows

def table4():
    rows = []
    for L in [6, 12, 24, 48, 72]:
        dm = {v: damped('0km', v, 0)[0][L - 1] for v in ['u', 'v', 'T']}
        ent = [('M5', ['%s (%.2f)' % (mm('M5', '0km', v, L), st.mean(vals('M5', '0km', v, L, 'acc_clim'))) for v in V]), ('Persistence', ['%.2f (%.2f)' % (dser('0km', v, 'persist')[L - 1], dser('0km', v, 'persist', 'acc_clim')[L - 1]) for v in V]), ('Damped persistence', ['%.2f' % dm[v] for v in ['u', 'v', 'T']] + ['–']), ('24-h-lagged persistence', ['%.2f (%.2f)' % (dser('0km', v, 'lag24')[L - 1], dser('0km', v, 'lag24', 'acc_clim')[L - 1]) for v in V]), ('Climatology', ['%.2f' % dser('0km', v, 'clim')[L - 1] for v in V]), ('Diurnal climatology', ['%.2f' % dser('0km', v, 'dclim')[L - 1] for v in V])]
        for i, (n, c) in enumerate(ent):
            rows.append(['t+%d h' % L if i == 0 else '', n] + c)
    return rows

def ratio1(g):
    r = [get(g, s, 1, -1, 0) / get(g, s, 1, 0, 0) for s in seeds(g)]
    return '%.2f–%.2f' % (min(r), max(r))

def table5():
    rows = [['Persistence', '–'] + ['%.2f' % base('0km', v)[11] for v in V]]
    for g in ['M2', 'M2+S', 'M3', 'M3+S', 'M4', 'M4+S', 'M5', 'M5+S']:
        rows.append([g, ratio1(g)] + [mm(g, '0km', v, 12) for v in V])
    return rows

def table6():
    rows = []
    for g, lab, lam, ep in [('M2', 'M2', '0.03', '6'), ('M2-P0', 'M2-P0', '0', '6'), ('M2-P1', 'M2-P1', '1', '6'), ('M2-P10', 'M2-P10', '10', '6'), ('M2-2E', 'M2-2E', '0.03', '12'), ('M5', 'M5', '0.03', '6'), ('M5-P0', 'M5-P0', '0', '6'), ('M5-2E', 'M5-2E', '0.03', '12')]:
        rows.append([lab, lam, ep, mm(g, '0km', 'u', 12), mm(g, '0km', 'T', 12), mm(g, '0km', 'u', 72), mm(g, '0km', 'T', 72), mm(g, '0km', 'T', 143, '%.1f'), mm(g, '0km', 'Ps', 143, '%.1f')])
    return rows

def table7():
    rows = []
    for h, hn in (('100km', '100 km'), ('200km', '200 km'), ('300km', '300 km')):
        for L in (6, 24, 72):
            r = [hn if L == 6 else '', 't+%d h' % L]
            for v in ('u', 'T'):
                pass
            cells = []
            for b in ('persist', 'clim', 'dclim'):
                cells.append('%.0f / %.0f' % (dser(h, 'u', b)[L - 1], dser(h, 'T', b)[L - 1]))
            for g in ('M2', 'M5', 'M2-SZA', 'M5-SZA'):
                cells.append('%.0f / %.0f' % (st.mean(vals(g, h, 'u', L)), st.mean(vals(g, h, 'T', L))))
            rows.append(r + cells)
    return rows
if __name__ == '__main__':
    for t in (table3, table4, table5, table6, table7):
        print(t.__name__)
        for r in t():
            print('  ', ' | '.join(r))
