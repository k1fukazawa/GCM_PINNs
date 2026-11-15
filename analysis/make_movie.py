#!/usr/bin/env python3
"""Movie of the time evolution of the GCM (truth), the surrogate, and their difference.

Input : fields.npz written by  code/evaluate.py --dump_fields 1 --dump_leads 1 2 ... 143
Output: an MP4 file (needs ffmpeg) or, with --format gif or when ffmpeg is absent, a GIF.
Rows  : one per (variable, height); columns: GCM, surrogate, surrogate minus GCM.

Example
  python make_movie.py --fields runs/M5_seed0/extra3/fields.npz --label M5 --out M5.mp4
  python make_movie.py --fields ... --rows u:0 T:0 u:300 T:300 --fps 12
"""
import argparse, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

ap = argparse.ArgumentParser()
ap.add_argument("--fields", required=True)
ap.add_argument("--out", default="movie.mp4")
ap.add_argument("--label", default="PINN", help="name of the surrogate shown in the panel titles")
ap.add_argument("--rows", nargs="+", default=["u:0", "T:0", "u:300", "T:300"],
                help="variable:height_km for each row (variables u, v, T; height 0 = surface)")
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--dpi", type=int, default=120)
ap.add_argument("--format", choices=["auto", "mp4", "gif"], default="auto")
ap.add_argument("--ut0", type=float, default=0.0, help="UT [h] of the initial state of the prediction")
ap.add_argument("--max_frames", type=int, default=0)
args = ap.parse_args()

f = np.load(args.fields)
leads = [int(x) for x in f["leads"]]
hk = [float(x) for x in f["heights_km"]]
lat = f["lat_deg"]; lon = f["lon_deg"]
if args.max_frames > 0:
    leads = leads[:args.max_frames]
UNIT = {"u": "m/s", "v": "m/s", "T": "K"}
rows = []
for r in args.rows:
    v, h = r.split(":")
    k = int(np.argmin(np.abs(np.array(hk) - float(h))))
    tr = f["truth_" + v][:len(leads), k].astype(np.float32); pr = f["pred_" + v][:len(leads), k].astype(np.float32)
    if v == "T":
        lo, hi = np.percentile(tr, [1, 99]); cmap = "turbo"
    else:
        a = float(np.percentile(np.abs(tr), 99)); lo, hi = -a, a; cmap = "RdBu_r"
    d = float(np.percentile(np.abs(tr - tr.mean(axis=0)), 99))          # scale of the temporal variability of the truth
    name = ("Surface " if hk[k] < 2 else "") + v + ("" if hk[k] < 2 else f" at {hk[k]:.0f} km")
    rows.append(dict(v=v, tr=tr, pr=pr, lo=lo, hi=hi, cmap=cmap, d=d, name=f"{name} [{UNIT[v]}]"))

n = len(rows)
fig, ax = plt.subplots(n, 3, figsize=(10.4, 2.0 * n + 0.6), sharex=True, sharey=True, squeeze=False, layout="constrained")
ims = []
for i, r in enumerate(rows):
    row = []
    for j, (F, ttl, kw) in enumerate(((r["tr"][0], "GCM", dict(vmin=r["lo"], vmax=r["hi"], cmap=r["cmap"])),
                                      (r["pr"][0], args.label, dict(vmin=r["lo"], vmax=r["hi"], cmap=r["cmap"])),
                                      (r["pr"][0] - r["tr"][0], f"{args.label} − GCM", dict(vmin=-r["d"], vmax=r["d"], cmap="PuOr_r")))):
        im = ax[i, j].pcolormesh(lon, lat, F, shading="auto", **kw)
        ax[i, j].set_title(ttl, fontsize=9, loc="left")
        ax[i, j].set_xticks([0, 90, 180, 270]); ax[i, j].set_yticks([-60, -30, 0, 30, 60]); ax[i, j].tick_params(labelsize=7)
        row.append(im)
    cb = fig.colorbar(row[1], ax=ax[i, 1], fraction=0.06, pad=0.03); cb.set_label(r["name"], fontsize=8); cb.ax.tick_params(labelsize=7)
    cb = fig.colorbar(row[2], ax=ax[i, 2], fraction=0.06, pad=0.03); cb.set_label(f"Difference [{UNIT[r['v']]}]", fontsize=8); cb.ax.tick_params(labelsize=7)
    ims.append(row)
for a in ax[-1]:
    a.set_xlabel("Longitude [°E]", fontsize=8)
for a in ax[:, 0]:
    a.set_ylabel("Latitude [°]", fontsize=8)
sup = fig.suptitle("Lead time t+000 h  (day 0, 00 UT)", fontsize=11, x=0.02, ha="left")


def update(t):
    for r, row in zip(rows, ims):
        row[0].set_array(r["tr"][t].ravel()); row[1].set_array(r["pr"][t].ravel()); row[2].set_array((r["pr"][t] - r["tr"][t]).ravel())
    l = leads[t]
    sup.set_text(f"Lead time t+{l:d} h  (day {l // 24}, {int((args.ut0 + l) % 24):02d} UT)")
    return [im for row in ims for im in row] + [sup]


anim = animation.FuncAnimation(fig, update, frames=len(leads), blit=False)
fmt = args.format
if fmt == "auto":
    fmt = "mp4" if shutil.which("ffmpeg") else "gif"
out = os.path.splitext(args.out)[0] + "." + fmt
if fmt == "mp4":
    anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=4000, extra_args=["-pix_fmt", "yuv420p"]), dpi=args.dpi)
else:
    anim.save(out, writer=animation.PillowWriter(fps=args.fps), dpi=min(args.dpi, 90))
print(f"saved {out} ({len(leads)} frames, {args.fps} fps)")
