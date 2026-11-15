#!/usr/bin/env python3
"""Figures 3-8 of the paper (written to ../figs). Run: python figs.py
"""
from lib import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 0.8, 'lines.linewidth': 1.4, 'savefig.dpi': 300, 'figure.dpi': 100})
O = '../figs/'
os.makedirs(O, exist_ok=True)
L = np.arange(1, 144)
UN = {'u': 'm/s', 'v': 'm/s', 'T': 'K', 'Ps': 'hPa'}
NAME = {'u': 'u', 'v': 'v', 'T': 'T', 'Ps': '$P_s$'}

def save(fig, n):
    fig.savefig(O + n + '.png', bbox_inches='tight')
    fig.savefig(O + n + '.pdf', bbox_inches='tight')
    plt.close(fig)
cols = {'M0': '#999999', 'M1': '#56B4E9', 'M2': '#0072B2', 'M3': '#009E73', 'M4': '#E69F00', 'M5': '#D55E00'}
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
for a, v, lab in zip(ax.flat, ['u', 'v', 'T', 'Ps'], 'abcd'):
    a.plot(L, base('0km', v), 'k--', lw=1.2, label='Persistence')
    a.plot(L, base('0km', v, 'rmse', 'clim'), 'k:', lw=1.2, label='Climatology')
    for g in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
        a.plot(L, meanser(g, '0km', v), color=cols[g], label=g, lw=1.8 if g == 'M5' else 1.2)
    top = {'u': 9, 'v': 9, 'T': 12, 'Ps': 22}[v]
    a.set_ylim(0, top)
    a.set_xlim(0, 143)
    a.set_xticks(range(0, 144, 24))
    a.set_ylabel(f'RMSE of surface {NAME[v]} [{UN[v]}]')
    a.set_title(f'({lab})', loc='left', fontsize=9)
    a.grid(alpha=0.3, lw=0.5)
for a in ax[1]:
    a.set_xlabel('Lead time [h]')
ax[0, 0].legend(ncol=2, fontsize=7.5, frameon=False, loc='lower right')
fig.tight_layout()
save(fig, 'figure3')
fig, a = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for j, (g, ttl) in enumerate((('M4', '(a) M4'), ('M5', '(b) M5'))):
    d = E[g, 0]['drift_series']
    a[j].plot(L, np.array(d['truth']) / 128, 'k-', lw=1.6, label='Truth')
    for s in seeds(g):
        d = E[g, s]['drift_series']
        a[j].plot(L, np.array(d['pred']) / 128, color='#0072B2', lw=1.1, label='With mass fixer' if s == 0 else None)
        a[j].plot(L, np.array(d['pred_nofix']) / 128, color='#D55E00', lw=1.1, ls='--', label='Without mass fixer' if s == 0 else None)
    a[j].set_title(ttl, loc='left', fontsize=9)
    a[j].set_xlabel('Lead time [h]')
    a[j].set_xlim(0, 143)
    a[j].set_xticks(range(0, 144, 24))
    a[j].grid(alpha=0.3, lw=0.5)
a[0].set_ylabel('Global-mean $P_s$ [hPa]')
a[0].legend(fontsize=7.5, frameon=False)
fig.tight_layout()
save(fig, 'figure4')
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
for a, v, lab in zip(ax.flat, ['u', 'v', 'T', 'Ps'], 'abcd'):
    a.fill_between(L, minser('M5', '0km', v), maxser('M5', '0km', v), color='#D55E00', alpha=0.25, lw=0)
    a.plot(L, meanser('M5', '0km', v), color='#D55E00', lw=1.8, label='M5 (mean and range of three seeds)')
    a.plot(L, dser('0km', v, 'persist'), 'k--', lw=1.2, label='Persistence')
    a.plot(L, dser('0km', v, 'clim'), 'k:', lw=1.2, label='Climatology')
    a.plot(L, dser('0km', v, 'dclim'), color='#009E73', lw=1.1, label='Diurnal climatology')
    a.plot(L, dser('0km', v, 'lag24'), color='#0072B2', lw=1.1, label='24-h-lagged persistence')
    if v != 'Ps':
        a.plot(L, damped('0km', v, 0)[0], color='#CC79A7', lw=1.1, ls='-.', label='Damped persistence (optimal)')
    a.set_xlim(0, 143)
    a.set_xticks(range(0, 144, 24))
    a.set_ylim(0, {'u': 8, 'v': 8, 'T': 9, 'Ps': 26}[v])
    a.set_ylabel(f'RMSE of surface {NAME[v]} [{UN[v]}]')
    a.set_title(f'({lab})', loc='left', fontsize=9)
    a.grid(alpha=0.3, lw=0.5)
for a in ax[1]:
    a.set_xlabel('Lead time [h]')
ax[0, 0].legend(fontsize=6.8, frameon=False, loc='lower right')
fig.tight_layout()
save(fig, 'figure6')
gc = {'M2': '#0072B2', 'M5': '#D55E00', 'M5-SZA': '#009E73'}
fig, ax = plt.subplots(3, 2, figsize=(7.2, 7.0), sharex=True)
for i, (h, hn) in enumerate((('300km', '307 km'), ('200km', '195 km'), ('100km', '100 km'))):
    for j, v in enumerate(('u', 'T')):
        a = ax[i, j]
        a.plot(L, dser(h, v, 'persist'), 'k--', lw=1.0, label='Persistence')
        a.plot(L, dser(h, v, 'clim'), 'k:', lw=1.2, label='Climatology')
        a.plot(L, dser(h, v, 'dclim'), color='#CC79A7', lw=1.3, label='Diurnal climatology')
        for g in ['M2', 'M5', 'M5-SZA']:
            a.fill_between(L, minser(g, h, v), maxser(g, h, v), color=gc[g], alpha=0.2, lw=0)
            a.plot(L, meanser(g, h, v), color=gc[g], label=g)
        a.set_ylim(0, {'300km': {'u': 200, 'T': 130}, '200km': {'u': 150, 'T': 120}, '100km': {'u': 110, 'T': 45}}[h][v])
        a.set_xlim(0, 143)
        a.set_xticks(range(0, 144, 24))
        a.grid(alpha=0.3, lw=0.5)
        a.set_ylabel(f'RMSE of {v} [{UN[v]}]')
        a.set_title(f"({'abcdef'[i * 2 + j]}) {hn}", loc='left', fontsize=9)
for a in ax[2]:
    a.set_xlabel('Lead time [h]')
hh, ll = ax[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.legend(hh, ll, ncol=6, fontsize=7.5, frameon=False, loc='lower center')
save(fig, 'figure7')
f5 = fields('M5', 0)
fs = fields('M5-SZA', 0)
lat = f5['lat_deg']
lon = f5['lon_deg']
leads = list(f5['leads'])
print(leads, f5['heights_km'])

def panel(a, F, vmin, vmax, cmap, ttl):
    im = a.pcolormesh(lon, lat, F, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto', rasterized=True)
    a.set_title(ttl, loc='left', fontsize=8)
    a.set_xticks([0, 90, 180, 270])
    a.set_yticks([-60, -30, 0, 30, 60])
    a.tick_params(labelsize=7)
    return im
rows = [('u', 0, 24, 'Surface u, t+24 h [m/s]', -20, 20, 'RdBu_r'), ('T', 0, 24, 'Surface T, t+24 h [K]', 230, 310, 'turbo'), ('u', 3, 12, 'u at 307 km, t+12 h [m/s]', -250, 250, 'RdBu_r'), ('T', 3, 12, 'T at 307 km, t+12 h [K]', 600, 1000, 'turbo')]
fig, ax = plt.subplots(4, 3, figsize=(7.2, 7.6), sharex=True, sharey=True)
for i, (v, k, ld, ttl, vmin, vmax, cm) in enumerate(rows):
    li = leads.index(ld)
    tr = f5['truth_' + v][li, k]
    if i >= 2:
        vmin, vmax = (np.percentile(tr, 1), np.percentile(tr, 99)) if v == 'T' else (vmin, vmax)
    for j, (F, nm) in enumerate(((tr, 'Truth'), (f5['pred_' + v][li, k], 'M5'), (fs['pred_' + v][li, k], 'M5-SZA'))):
        im = panel(ax[i, j], F, vmin, vmax, cm, f"({'abcdefghijkl'[i * 3 + j]}) {nm}")
    cb = fig.colorbar(im, ax=ax[i, :], fraction=0.025, pad=0.02)
    cb.set_label(ttl, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
for a in ax[-1]:
    a.set_xlabel('Longitude [°E]', fontsize=8)
for a in ax[:, 0]:
    a.set_ylabel('Latitude [°]', fontsize=8)
save(fig, 'figure5')
th = list(np.round(f5['tide_heights_km'], 1))
print(th)
fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.4), sharex=True, sharey=True)
hrs = np.arange(1, 73)
for i, k in enumerate((0, 3)):
    tr = f5['tide_truth_T_eq'][k, :72]
    m = tr.mean()
    a_ = np.percentile(abs(tr - m), 99)
    for j, (F, nm) in enumerate(((tr, 'Truth'), (f5['tide_pred_T_eq'][k, :72], 'M5'), (fs['tide_pred_T_eq'][k, :72], 'M5-SZA'))):
        im = ax[i, j].pcolormesh(lon, hrs, F - F.mean(axis=1, keepdims=True), vmin=-a_, vmax=a_, cmap='RdBu_r', shading='auto', rasterized=True)
        ax[i, j].set_title(f"({'abcdef'[i * 3 + j]}) {nm}, {th[k]:.1f} km", loc='left', fontsize=8)
        ax[i, j].set_xticks([0, 90, 180, 270])
        ax[i, j].set_yticks(range(0, 73, 12))
        ax[i, j].tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax[i, :], fraction=0.025, pad=0.02)
    cb.set_label('Deviation of T from the zonal mean [K]', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
for a in ax[-1]:
    a.set_xlabel('Longitude [°E]', fontsize=8)
for a in ax[:, 0]:
    a.set_ylabel('Lead time [h]', fontsize=8)
save(fig, 'figure8')
print('ok')
