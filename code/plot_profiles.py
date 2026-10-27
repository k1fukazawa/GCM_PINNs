#!/usr/bin/env python3
"""\
Draw Figures 7 and 8 from results.json files written by evaluate.py.
"""


import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--jsons", nargs="+", required=True)
ap.add_argument("--labels", nargs="+", default=None)
ap.add_argument("--outdir", default="./figs/fig7_fig8")
ap.add_argument("--lead", type=int, default=72, help=" 7  [h]")
ap.add_argument("--var", default="T", choices=["u", "v", "T"])
ap.add_argument("--fig8_level_km", default="300", help=" 8  300")
ap.add_argument("--fig8_surface_var", default="T")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
labels = args.labels or [os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p)))) for p in args.jsons]
assert len(labels) == len(args.jsons), "--labels の数が --jsons と一致しない"

D = [json.load(open(p, encoding="utf-8")) for p in args.jsons]
colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(D), 1)))

z_ref = None
fig, ax = plt.subplots(1, 3, figsize=(11.5, 5.2), sharey=True)
for d, lab, c in zip(D, labels, colors):
    pr = d.get("profile")
    if not pr:
        print(f"[skip] {lab}: profile がない（--profile_leads を付けて eval_extra3 を実行）")
        continue
    ent = pr["by_lead"].get(str(args.lead))
    if ent is None:
        print(f"[skip] {lab}: lead {args.lead} がない（ある lead: {list(pr['by_lead'])})")
        continue
    z = np.asarray(pr["heights_km"]); z_ref = z
    m = ent["pred"][args.var]; s = ent["persist"][args.var]
    ax[0].plot(m["bias"], z, "-", color=c, lw=1.4, label=lab)
    ax[1].plot(np.maximum(m["crmse"], 1e-3), z, "-", color=c, lw=1.4, label=lab)
    ax[2].plot(m["acc_clim"], z, "-", color=c, lw=1.4, label=lab)
    ax[0].plot(s["bias"], z, ":", color=c, lw=1.0)
    ax[1].plot(np.maximum(s["crmse"], 1e-3), z, ":", color=c, lw=1.0)
    ax[2].plot(s["acc_clim"], z, ":", color=c, lw=1.0)
    if lab == labels[0]:
        ax[0].plot(s["bias"], z, ":", color="k", lw=1.6, label="persistence")
ax[0].axvline(0, color="k", lw=0.8)
ax[1].set_xscale("log")
ax[2].axvline(0, color="k", lw=0.8)
ax[2].set_xlim(-0.2, 1.0)
ax[0].set_title(f"{args.var} bias [{'K' if args.var == 'T' else 'm/s'}]")
ax[1].set_title(f"{args.var} centered RMSE [{'K' if args.var == 'T' else 'm/s'}]")
ax[2].set_title(f"{args.var} ACC (climatological anomaly)")
for a in ax:
    a.grid(alpha=0.3); a.set_xlabel(a.get_title()); a.set_title("")
ax[0].set_ylabel("height [km]")
ax[0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"Vertical profile at t+{args.lead}h (48 levels, before calibration, EMA applied)", fontsize=10)
fig.tight_layout()
p1 = os.path.join(args.outdir, f"fig7_{args.var}_profile_t{args.lead}h.png")
fig.savefig(p1, dpi=150); plt.close(fig)
print("saved", p1)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for d, lab, c in zip(D, labels, colors):
    lv = f"{args.fig8_level_km}km"
    if lv in d.get("native", {}) and "u" in d["native"][lv]:
        s = d["native"][lv]["u"]["pred"]["series"]; sp = d["native"][lv]["u"]["persist"]["series"]
        L = np.asarray(d["native"][lv]["u"]["leads_h"])
        axes[0].plot(L, np.abs(s["bias"]), "-", color=c, lw=1.5, label=lab)
        axes[0].plot(L, s["crmse"], "--", color=c, lw=1.2)
        axes[0].plot(L, sp["rmse"], ":", color=c, lw=1.0)
    s = d["native"]["0km"][args.fig8_surface_var]["pred"]["series"]
    sp = d["native"]["0km"][args.fig8_surface_var]["persist"]["series"]
    L = np.asarray(d["native"]["0km"][args.fig8_surface_var]["leads_h"])
    axes[1].plot(L, np.abs(s["bias"]), "-", color=c, lw=1.5, label=lab)
    axes[1].plot(L, s["crmse"], "--", color=c, lw=1.2)
    axes[1].plot(L, sp["rmse"], ":", color=c, lw=1.0)
for a, t, u in ((axes[0], f"u at {args.fig8_level_km} km", "m/s"),
                (axes[1], f"{args.fig8_surface_var} at surface", "K" if args.fig8_surface_var == "T" else "hPa")):
    a.set_title(f"{t} [{u}]"); a.set_xlabel("lead time [h]"); a.set_yscale("log")
    a.set_xticks(range(0, 145, 24)); a.grid(alpha=0.3)
axes[0].plot([], [], "-", color="0.3", label="|bias|")
axes[0].plot([], [], "--", color="0.3", label="centered RMSE")
axes[0].plot([], [], ":", color="0.3", label="persistence RMSE")
axes[0].legend(fontsize=7.5)
fig.tight_layout()
p2 = os.path.join(args.outdir, "fig8_bias_decomposition.png")
fig.savefig(p2, dpi=150); plt.close(fig)
print("saved", p2)
