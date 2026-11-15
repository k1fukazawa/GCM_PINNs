#!/usr/bin/env python3
"""Pack the checkpoints of the runs for release.

For every run directory <runs>/<internal name>_s<seed>/final.pt, write
weights/<paper name>_seed<seed>.pt with the entries
  config_id : paper name and seed
  args      : hyperparameters of the run (paths and machine settings removed)
  model     : trained weights (unchanged)
and print a table of sizes and SHA-256 digests (also written to weights/SHA256.md).

Usage:  python pack_weights.py --runs ./out/v18 --out ./weights
"""
import argparse, hashlib, os
import torch

MAP = {"M0": "M0w", "M1": "M1w", "M2": "M2w", "M3": "B1_M3w_nospec", "M4": "B5_M8w_nospec", "M5": "B6_M9w_nospec",
       "M2+S": "B4_M2w_spec", "M3+S": "M3w", "M4+S": "M8w", "M5+S": "M9w",
       "M2-P0": "M2w_lp0", "M2-P1": "A3_M2w_lp1", "M2-P10": "A3_M2w_lp10", "M5-P0": "B6_M9w_nospec_lp0",
       "M2-2E": "A2_M2w_long", "M5-2E": "B7_M9w_nospec_long", "M2-SZA": "M2w_sza", "M5-SZA": "B6_M9w_nospec_sza"}
DROP = ("dir", "outdir", "tag", "threads", "resume", "compile", "amp", "channels_last", "save_every", "quick", "rollout_start")

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="./out/v18")
ap.add_argument("--out", default="./weights")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rows = []
for name, d in MAP.items():
    for s in range(3):
        src = f"{args.runs}/{d}_s{s}/final.pt"
        if not os.path.exists(src):
            print(f"[missing] {src}"); continue
        ck = torch.load(src, map_location="cpu", weights_only=False)
        a = {k: v for k, v in ck["args"].items() if k not in DROP}
        out = {"config_id": f"{name}_seed{s}", "args": a, "model": ck["model"]}
        dst = f"{args.out}/{name}_seed{s}.pt"
        torch.save(out, dst)
        back = torch.load(dst, map_location="cpu", weights_only=False)
        assert all(torch.equal(back["model"][k], ck["model"][k]) for k in ck["model"]), dst
        h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        rows.append(f"| `{name}_seed{s}.pt` | {name} | {s} | {os.path.getsize(dst) / 1e6:.2f} MB | `{h[:16]}` |")
txt = "| file | configuration | seed | size | SHA-256 (first 16 hex) |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
open(f"{args.out}/SHA256.md", "w").write(txt)
print(txt); print(f"{len(rows)} checkpoints written to {args.out}")
