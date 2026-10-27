#!/usr/bin/env python3
"""\
Build Table 4 (lambda_phys control experiment) from results.json files.
The table uses the uncorrected series surface_rmse_raw; the calibrated series
surface_rmse is read only for reference.
"""


import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--group", nargs="+", action="append", required=True,
                help="")
ap.add_argument("--lead", default="t+144h")
ap.add_argument("--vars", nargs="+", default=["u", "T", "Ps"])
ap.add_argument("--out", default=None, help="CSV ")
a = ap.parse_args()

missing = []
rows = []
for g in a.group:
    label, dirs = g[0], g[1:]
    per = {}
    for d in dirs:
        p = os.path.join(d, "results.json")
        if not os.path.exists(p):
            missing.append(p); continue
        r = json.load(open(p, encoding="utf-8"))
        c = r.get("config") or {}
        lam = float(c.get("lambda_phys") or 0)
        seed = c.get("seed")
        raw = (r.get("surface_rmse_raw") or {}).get(a.lead) or (r.get("surface_rmse") or {}).get(a.lead) or {}
        cal = (r.get("surface_rmse") or {}).get(a.lead) or {}
        lst = per.setdefault(lam, [])
        dup = [x for x in lst if x["seed"] == seed]
        if dup:
            print(f"[dup] λ={lam} seed={seed}: {d} は {dup[0]['dir']} と同じ条件なので除外")
            continue
        lst.append(dict(dir=d, seed=seed, **{f"raw_{k}": raw.get(k) for k in a.vars},
                                           **{f"cal_{k}": cal.get(k) for k in a.vars}))
    for lam, lst in sorted(per.items(), key=lambda x: -x[0]):
        row = dict(label=label, lam=lam, n=len(lst), seeds=[x["seed"] for x in lst])
        for k in a.vars:
            v = [x[f"raw_{k}"] for x in lst if x[f"raw_{k}"] is not None]
            row[k] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (None, None)
            vc = [x[f"cal_{k}"] for x in lst if x[f"cal_{k}"] is not None]
            row[f"cal_{k}"] = (float(np.mean(vc)), float(np.std(vc, ddof=1)) if len(vc) > 1 else 0.0) if vc else (None, None)
        rows.append(row)

if missing:
    print("\n[!] 存在しない results.json（集計から除外した）:")
    for m in missing:
        print("    " + m)
    print(f"    計 {len(missing)} 件。この条件は n が不足する。\n")
print(f"\n== 表 4 素材（{a.lead}、補正前 = surface_rmse_raw、括弧は標準偏差）\n")
hdr = ["構成", "λphys", "n"] + [f"{k}" for k in a.vars] + [f"{k}（補正後）" for k in a.vars]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
fmt = lambda t: "—" if t[0] is None else (f"{t[0]:.1f}" if t[1] == 0 else f"{t[0]:.1f} ± {t[1]:.1f}")
for r in rows:
    cells = [r["label"], f"{r['lam']:g}", str(r["n"]) + ("  ⚠" if r["n"] < 3 else "")] + [fmt(r[k]) for k in a.vars] + [fmt(r[f"cal_{k}"]) for k in a.vars]
    print("| " + " | ".join(cells) + " |")

short = [r for r in rows if r["n"] < 3]
if short:
    print("\n[!] n < 3 の条件（3 シード比較に不足）: " +
          ", ".join(f"{r['label']} λ={r['lam']:g} (n={r['n']})" for r in short))

print("\n== λphys 0.03 → 0 の差（補正前）")
for label in dict.fromkeys(r["label"] for r in rows):
    hi = next((r for r in rows if r["label"] == label and r["lam"] > 0), None)
    lo = next((r for r in rows if r["label"] == label and r["lam"] == 0), None)
    if not hi or not lo:
        print(f"  {label}: 両方の λ が揃っていない"); continue
    s = []
    for k in a.vars:
        if hi[k][0] is None or lo[k][0] is None:
            continue
        d = lo[k][0] - hi[k][0]
        pooled = np.sqrt((hi[k][1] ** 2 + lo[k][1] ** 2) / 2) if (hi[k][1] or lo[k][1]) else None
        eff = f"  (d={d / pooled:+.2f})" if pooled else ""
        s.append(f"{k} {hi[k][0]:.1f} → {lo[k][0]:.1f} (Δ {d:+.1f}){eff}")
    print(f"  {label}: " + " / ".join(s))

if a.out:
    import csv
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "lambda_phys", "n", "seeds"] + [f"{k}_mean" for k in a.vars]
                   + [f"{k}_sd" for k in a.vars] + [f"cal_{k}_mean" for k in a.vars] + [f"cal_{k}_sd" for k in a.vars])
        for r in rows:
            w.writerow([r["label"], r["lam"], r["n"], " ".join(str(s) for s in r["seeds"])]
                       + [r[k][0] for k in a.vars] + [r[k][1] for k in a.vars]
                       + [r[f"cal_{k}"][0] for k in a.vars] + [r[f"cal_{k}"][1] for k in a.vars])
    print(f"\n[i] {a.out} に保存")
