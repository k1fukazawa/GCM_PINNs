#!/usr/bin/env python3
"""Figure 2 of the paper (schematic of the model).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
fig, ax = plt.subplots(figsize=(6.5, 7.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 108)
ax.axis('off')

def box(x, y, w, h, fc, ec, lines, bold=False, color='k', fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=1.4))
    n = len(lines)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n - 1 - 2 * i) * fs * 0.155, t, ha='center', va='center', fontsize=fs, color=color, fontweight='bold' if (bold if isinstance(bold, bool) else bold[i]) else 'normal')

def arr(x0, y0, x1, y1, **k):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.3, **k))
B = '#DCE6F1'
BE = '#2F4B7C'
Y = '#FDF0CC'
YE = '#C9A227'
box(10, 97, 84, 9, B, BE, ['Input state $X(t)$: u, v, ω, T, P$_s$', '(global fields, 128 × 64 × 48 levels)'], bold=True)
box(10, 84, 84, 9, B, BE, ['Level-wise normalization (mean and SD of each level)', 'Input noise in training (normalized SD 0.05)'])
box(10, 68, 84, 12, '#2E7D73', '#1D4F49', ['3D CNN $G_W$ (four convolutional layers)', 'Channels 48 → 96 → 64 → 32', 'Time-evolution mapping from spatial correlations'], bold=True, color='w')
box(10, 56, 84, 8.5, Y, YE, ['Increment prediction $\\Delta\\hat{X}$ (normalized space;', 'horizontal smoothing in training)'])
box(10, 43.5, 84, 8.5, Y, YE, ['State update $\\hat{X}(t+\\Delta t)=X(t)+\\Delta\\hat{X}$', 'Increment clip (±1.0) and state clamps'])
box(3, 25, 45, 12, B, BE, ['Predicted field $\\hat{X}(t+\\Delta t)$', 'compared with truth $X(t+\\Delta t)$'])
box(51, 25, 46, 12, '#FBE3D6', '#C8602A', ['Pressure $p_k$ (hypsometric eq.)', '→ residual of primitive', 'equations $L_{\\mathrm{phys}}$'])
box(3, 11.5, 94, 10, '#E2EDDA', '#5B8C3A', ['Loss $L=L_{\\mathrm{data}}+\\lambda_{\\mathrm{phys}}L_{\\mathrm{phys}}+\\cdots$', 'level weight $\\eta(z_k)$, variable weights $\\gamma_{\\mathrm{var}}$ (Section 2.4)'])
box(3, 1, 45, 7.5, '#EAEAEA', '#666666', ['Curriculum learning', 'integration length 1 → 6 → 12 → 24 h'], fs=8.6)
box(51, 1, 46, 7.5, '#EAEAEA', '#666666', ['Prediction: 1-h steps to t+143 h'], bold=True, fs=8.6)
for y0, y1 in ((97, 93), (84, 80), (68, 64.5), (56, 52)):
    arr(52, y0, 52, y1)
arr(31, 43.5, 26, 37)
arr(73, 43.5, 74, 37)
arr(25.5, 25, 25.5, 21.5)
arr(74, 25, 74, 21.5)
arr(25.5, 11.5, 25.5, 8.5)
arr(74, 11.5, 74, 8.5)
ax.plot([6, 6], [37, 101.5], ls='--', color=BE, lw=1.3)
ax.annotate('', xy=(10, 101.5), xytext=(6, 101.5), arrowprops=dict(arrowstyle='-|>', color=BE, lw=1.3, ls='--'))
ax.text(3, 70, 'Autoregressive integration', rotation=90, ha='center', va='center', color=BE, fontsize=9.5)
fig.savefig('../figs/figure2.png', dpi=300, bbox_inches='tight')
fig.savefig('../figs/figure2.pdf', bbox_inches='tight')
