#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation of M2-SZA: the values of Tables S1, A1 and A2-A3 from one rollout.

- Normalization as in training (level-wise mean and standard deviation of the first train_days*24 h).
- Rollout: the input of step k is at time index rs+k and the truth at rs+k+1 (as in the paper).
- The three solar channels are computed from the time of the input state of each step.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_sza import Full3DStepper, smooth2d          # noqa: E402
from solar_geometry import geometry as solar_geometry   # noqa: E402

T_MIN_K, T_MAX_K = 120.0, 1500.0
PS_MIN_HPA, PS_MAX_HPA = 300.0, 1200.0
COMP = {"D0": (0, 1), "DW1": (1, 1), "SW2": (2, 2), "DE3": (-3, 1)}


def tide_analysis(F):
    """F:[T,nlat,nlon] -> amplitude [nlat], time of maximum [h] (2D FFT after removal of a linear trend)"""
    Tn = F.shape[0]
    tt = np.arange(Tn, dtype=np.float64)
    tt_c = tt - tt.mean()
    den = float((tt_c * tt_c).sum())
    A, PH = {}, {}
    for name, (s, om) in COMP.items():
        amp = np.zeros(F.shape[1])
        tmax = np.zeros(F.shape[1])
        for la in range(F.shape[1]):
            y = F[:, la, :].astype(np.float64)
            slope = (y * tt_c[:, None]).sum(axis=0) / den
            inter = y.mean(axis=0)
            y = y - (inter[None, :] + slope[None, :] * tt_c[:, None])
            Ff = np.fft.fft2(y) / (Tn * y.shape[1])
            kx_s = s % y.shape[1]
            kt = int(round(om * Tn / 24.0)) % Tn
            Fp = Ff[kt, kx_s]
            Fm = Ff[(Tn - kt) % Tn, kx_s]
            Fc = Fp if abs(Fp) >= abs(Fm) else Fm
            amp[la] = 2.0 * abs(Fc)
            tmax[la] = np.mod(-np.angle(Fc) / (2 * np.pi * om), 1.0 / om) * 24.0
        A[name], PH[name] = amp, tmax
    return A, PH


def circ_mean_h(x, period):
    a = np.asarray(x) * 2 * np.pi / period
    return float(np.mod(np.angle(np.exp(1j * a).mean()) * period / (2 * np.pi), period))


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / den if den > 0 else 0.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="directory that holds final.pt (e.g. weights/M2SZA_s0)")
    ap.add_argument("--data_dir", default="./npy")
    ap.add_argument("--threads", type=int, default=112)
    ap.add_argument("--t_train", type=int, default=600, help="time index of the initial state")
    ap.add_argument("--horizon", type=int, default=0, help="0: H-rs-1 (143, as in the paper)")
    ap.add_argument("--heights", type=float, nargs="+", default=[0.4, 99.8, 195.2, 307.3])
    ap.add_argument("--tide_heights", type=float, nargs="+", default=[99.8, 123.1, 136.5, 148.7])
    ap.add_argument("--lead_upper", type=int, default=72)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    run = a.run_dir.rstrip("/")
    _rj = os.path.join(run, "results.json")
    if os.path.exists(_rj):
        cfg = json.load(open(_rj))["config"]
    else:
        cfg = (torch.load(os.path.join(run, "final.pt"), map_location="cpu",
                          weights_only=False).get("args") or {})
        print("[i] no results.json; the args stored in final.pt are used")
    out_path = a.out or os.path.join(run, "eval_m2sza.json")

    gu = np.load(f"{a.data_dir}/gu.npy", mmap_mode="r")
    gv = np.load(f"{a.data_dir}/gv.npy", mmap_mode="r")
    gw = np.load(f"{a.data_dir}/gw.npy", mmap_mode="r")
    gt = np.load(f"{a.data_dir}/gt.npy", mmap_mode="r")
    gp = np.load(f"{a.data_dir}/gp.npy", mmap_mode="r")
    gz = np.load(f"{a.data_dir}/multilayer_gz.npy", mmap_mode="r")
    LEV, LAT, LON = gu.shape[1], gu.shape[2], gu.shape[3]
    H = gu.shape[0]
    rs = a.t_train
    horizon = a.horizon if a.horizon > 0 else H - rs - 1
    horizon = min(horizon, H - rs - 1)
    assert horizon > 0, f"horizon={horizon} (H={H}, t_train={rs})"

    lev_idx = np.unique(np.round(np.linspace(0, LEV - 1, int(cfg.get("levels", 48))))).astype(int)
    stride = int(cfg.get("stride", 1))
    lat_idx = np.arange(0, LAT, stride)
    lon_idx = np.arange(0, LON, stride)
    L, nlat, nlon = len(lev_idx), len(lat_idx), len(lon_idx)
    Z = np.asarray(gz[:, lev_idx, :, :].mean(axis=(0, 2, 3)), dtype=np.float64)
    zfeat_t = torch.tensor(((Z - Z.mean()) / (Z.std() + 1e-9)).astype(np.float32))

    ck = torch.load(os.path.join(run, "final.pt"), map_location="cpu", weights_only=False)
    state = ck["ema"] if ck.get("ema") is not None else ck["model"]
    in_ch = int(state["enc.0.conv.weight"].shape[1])
    use_terrain = bool(cfg.get("use_terrain", False))
    solar = bool(cfg.get("solar", True))
    expect = 6 + (1 if use_terrain else 0) + (3 if solar else 0)
    assert in_ch == expect, f"in_ch={in_ch}, but the settings give {expect}"
    model = Full3DStepper(in_ch=in_ch, ch=tuple(cfg.get("ch", [48, 96, 64, 32])))
    model.load_state_dict(state)
    model.eval()

    T_tr = int(cfg.get("train_days", 25)) * 24
    def ind(a_):
        return np.asarray(a_[:, lev_idx][:, :, lat_idx][:, :, :, lon_idx], dtype=np.float64)
    def stats_of(arr):
        m = np.zeros(L); s2 = np.zeros(L); n = 0
        for s0 in range(0, T_tr, 24):
            e = min(s0 + 24, T_tr)
            x = ind(arr[s0:e])
            m += x.mean(axis=(0, 2, 3)) * (e - s0)
            s2 += (x * x).mean(axis=(0, 2, 3)) * (e - s0)
            n += e - s0
        m /= n
        var = np.maximum(s2 / n - m * m, 0.0)
        return m.astype(np.float32), (np.sqrt(var) + 1e-8).astype(np.float32)
    mU, sU = stats_of(gu); mV, sV = stats_of(gv); mW, sW = stats_of(gw); mT, sT = stats_of(gt)
    lps_sum = 0.0; lps_sq = 0.0; nps = 0
    for s0 in range(0, T_tr, 24):
        e = min(s0 + 24, T_tr)
        x = np.log(np.clip(np.asarray(gp[s0:e][:, lat_idx][:, :, lon_idx], dtype=np.float64), 1.0, None))
        lps_sum += x.sum(); lps_sq += (x * x).sum(); nps += x.size
    mP = float(lps_sum / nps)
    sP = float(math.sqrt(max(lps_sq / nps - mP * mP, 0.0))) + 1e-8
    climPS = np.zeros((nlat, nlon), np.float64)
    for s0 in range(0, T_tr, 24):
        e = min(s0 + 24, T_tr)
        climPS += np.asarray(gp[s0:e][:, lat_idx][:, :, lon_idx], dtype=np.float64).sum(axis=0)
    climPS = (climPS / T_tr)
    climPS = (climPS / 100.0 if climPS.mean() > 2000 else climPS).astype(np.float32)

    def T3(x): return torch.tensor(x, dtype=torch.float32).reshape(1, 1, L, 1, 1)
    mUt, sUt, mVt, sVt = T3(mU), T3(sU), T3(mV), T3(sV)
    mWt, sWt, mTt, sTt = T3(mW), T3(sW), T3(mT), T3(sT)
    TMIN = (T_MIN_K - mTt) / sTt
    TMAX = (T_MAX_K - mTt) / sTt
    PMIN = (math.log(PS_MIN_HPA) - mP) / sP
    PMAX = (math.log(PS_MAX_HPA) - mP) / sP

    # Climatology (time mean of the training period)
    clim = {"u": np.zeros((L, nlat, nlon), np.float64), "v": np.zeros((L, nlat, nlon), np.float64),
            "T": np.zeros((L, nlat, nlon), np.float64)}
    for s0 in range(0, T_tr, 24):
        e = min(s0 + 24, T_tr)
        for nm, arr in (("u", gu), ("v", gv), ("T", gt)):
            clim[nm] += ind(arr[s0:e]).sum(axis=0)
    for nm in clim:
        clim[nm] = (clim[nm] / T_tr).astype(np.float32)

    lat_deg = 90.0 - 180.0 * (lat_idx + 0.5) / LAT
    lon_deg = 360.0 * (lon_idx + 0.5) / LON
    origin = cfg.get("time_origin", "2020-01-01T00:00:00+00:00")
    dsh = float(cfg.get("data_start_hour", 0.0))
    cache = {}
    def solar_t(hour):
        if hour not in cache:
            cache[hour] = torch.tensor(solar_geometry(dsh + hour, lat_deg, lon_deg, origin), dtype=torch.float32)
        return cache[hour]

    def load(j):
        out = []
        for arr in (gu, gv, gw, gt):
            out.append(np.ascontiguousarray(arr[j][lev_idx][:, lat_idx][:, :, lon_idx], dtype=np.float32))
        out.append(np.ascontiguousarray(gp[j][lat_idx][:, lon_idx], dtype=np.float32))
        return out

    def build_input(u, v, w, t, p, hour):
        chans = [u, v, w, t, p[:, :, None].expand(1, 1, L, nlat, nlon),
                 zfeat_t[None, None, :, None, None].expand(1, 1, L, nlat, nlon)]
        if use_terrain:
            raise SystemExit("checkpoints with terrain input are not supported (M2-SZA has terrain=False)")
        if solar:
            chans.append(solar_t(hour)[None, :, None].expand(1, 3, L, nlat, nlon))
        return torch.cat(chans, dim=1)

    hypervisc = float(cfg.get("hypervisc", 0.05) or 0.0)
    inf_hv = bool(cfg.get("inference_hypervisc", True)) and hypervisc > 0
    clip_inc = float(cfg.get("clip_inc", 1.0))
    mass_fix = bool(cfg.get("mass_fix", False))

    ktide = []
    for h in a.tide_heights:
        k = int(np.argmin(np.abs(Z - h * 1000.0)))
        if k not in [x for _, x in ktide]:
            ktide.append((float(h), k))
    kspot = {float(h): int(np.argmin(np.abs(Z - h * 1000.0))) for h in a.heights}

    print(f"[i] run={run}")
    print(f"[i] in_ch={in_ch} solar={solar} terrain={use_terrain} L={L} grid={nlat}x{nlon} H={H} rs={rs} horizon={horizon}")
    print(f"[i] Z: {Z[0]/1000:.2f}-{Z[-1]/1000:.2f} km")
    print(f"[i] tide altitude -> level: {[(h, k, round(Z[k]/1000, 1)) for h, k in ktide]}")
    print(f"[i] altitude -> level: {[(h, k, round(Z[k]/1000, 1)) for h, k in kspot.items()]}")
    print(f"[i] mass_fix={mass_fix} inf_hv={inf_hv} clip={clip_inc}")

    surf = {k: np.zeros((horizon, nlat, nlon), np.float32) for k in ("u", "v", "T", "Ps")}
    up = {k: {n: np.zeros((horizon, nlat, nlon), np.float32) for n in ("u", "v", "T")} for k in kspot.values()}
    tpred = {k: np.zeros((horizon, nlat, nlon), np.float32) for _, k in ktide}

    u0, v0, w0, t0, p0 = load(rs)
    nrm = lambda x, m, s: (torch.tensor(x[None, None]) - m) / s
    un, vn, wn, tn = nrm(u0, mUt, sUt), nrm(v0, mVt, sVt), nrm(w0, mWt, sWt), nrm(t0, mTt, sTt)
    pn = (torch.tensor(np.log(np.clip(p0, 1.0, None))[None, None]) - mP) / sP

    with torch.inference_mode():
        for step in range(horizon):
            d3d, d2d = model(build_input(un, vn, wn, tn, pn, rs + step))
            if inf_hv:
                d3d = d3d - hypervisc * (d3d - smooth2d(d3d))
                d2d = d2d - hypervisc * (d2d - smooth2d(d2d))
            if clip_inc > 0:
                d3d = torch.clamp(d3d, -clip_inc, clip_inc)
                d2d = torch.clamp(d2d, -clip_inc, clip_inc)
            if mass_fix:
                d2d = d2d - d2d.mean()
            un = un + d3d[:, 0:1]; vn = vn + d3d[:, 1:2]; wn = wn + d3d[:, 2:3]
            tn = torch.clamp(tn + d3d[:, 3:4], min=TMIN, max=TMAX)
            pn = torch.clamp(pn + d2d, min=PMIN, max=PMAX)
            U = un[0, 0].numpy() * sU[:, None, None] + mU[:, None, None]
            V = vn[0, 0].numpy() * sV[:, None, None] + mV[:, None, None]
            TT = tn[0, 0].numpy() * sT[:, None, None] + mT[:, None, None]
            PS = np.exp(pn[0, 0].numpy().astype(np.float64) * sP + mP)
            PS = PS / 100.0 if PS.mean() > 2000 else PS
            surf["u"][step] = U[0]; surf["v"][step] = V[0]; surf["T"][step] = TT[0]; surf["Ps"][step] = PS
            for k in kspot.values():
                up[k]["u"][step] = U[k]; up[k]["v"][step] = V[k]; up[k]["T"][step] = TT[k]
            for _, k in ktide:
                tpred[k][step] = TT[k]

    # Truth and metrics (paper: truth at rs+step+1; comparison with the training log: truth at rs+step)
    stat = {k: {n: [] for n in ("rmse", "bias", "corr")} for k in ("u", "v", "T", "Ps")}
    upstat = {k: {n: {m: [] for m in ("rmse", "bias", "crmse", "corr")} for n in ("u", "v", "T")} for k in kspot.values()}
    logchk = {k: [] for k in ("u", "v", "T", "Ps")}
    ttruth = {k: np.zeros((horizon, nlat, nlon), np.float32) for _, k in ktide}
    for j in range(rs, rs + horizon + 1):
        uu, vv, ww, tt_, pp = load(j)
        if j - rs - 1 >= 0:
            s = j - rs - 1
            pp_h = pp / 100.0 if pp.mean() > 2000 else pp
            for nm, P, Tr, C in (("u", surf["u"][s], uu[0], clim["u"][0]),
                                 ("v", surf["v"][s], vv[0], clim["v"][0]),
                                 ("T", surf["T"][s], tt_[0], clim["T"][0]),
                                 ("Ps", surf["Ps"][s], pp_h, climPS)):
                d = P.astype(np.float64) - Tr.astype(np.float64)
                stat[nm]["rmse"].append(float(np.sqrt((d * d).mean())))
                stat[nm]["bias"].append(float(d.mean()))
                stat[nm]["corr"].append(corr(P - C, Tr - C))
            for k in kspot.values():
                for nm, P, Tr, C in (("u", up[k]["u"][s], uu[k], clim["u"][k]),
                                     ("v", up[k]["v"][s], vv[k], clim["v"][k]),
                                     ("T", up[k]["T"][s], tt_[k], clim["T"][k])):
                    d = P.astype(np.float64) - Tr.astype(np.float64)
                    rm = float(np.sqrt((d * d).mean())); bi = float(d.mean())
                    upstat[k][nm]["rmse"].append(rm)
                    upstat[k][nm]["bias"].append(bi)
                    upstat[k][nm]["crmse"].append(math.sqrt(max(rm * rm - bi * bi, 0.0)))
                    upstat[k][nm]["corr"].append(corr(P - C, Tr - C))
            for _, k in ktide:
                ttruth[k][s] = tt_[k]
        if j - rs < horizon:
            s2 = j - rs
            pp_h = pp / 100.0 if pp.mean() > 2000 else pp
            for nm, P, Tr in (("u", surf["u"][s2], uu[0]), ("v", surf["v"][s2], vv[0]),
                              ("T", surf["T"][s2], tt_[0]), ("Ps", surf["Ps"][s2], pp_h)):
                d = P.astype(np.float64) - Tr.astype(np.float64)
                logchk[nm].append(float(np.sqrt((d * d).mean())))

    _rj2 = os.path.join(run, "results.json")
    cal_raw = (json.load(open(_rj2)).get("calibration") if os.path.exists(_rj2) else None) or {}
    cal = {}
    for _k, _v in cal_raw.items():
        cal[_k] = float(_v["factor"]) if isinstance(_v, dict) else float(_v)
    step0_ut = (rs + 1 + dsh) % 24.0

    def tide_block(F):
        A, PH = tide_analysis(F)
        band = np.abs(lat_deg) <= 21.5
        out = {}
        for nm, (_, om) in COMP.items():
            period = 24.0 / om
            ut = np.mod(PH[nm] + step0_ut, period)
            ent = {"amp_eq": float(A[nm][band].mean()), "phase_ut_eq_h": circ_mean_h(ut[band], period)}
            lt = np.mod(ut + lon_deg[0] / 15.0, period)
            ent["phase_lt_eq_h"] = circ_mean_h(lt[band], period)
            out[nm] = ent
        return out

    result = {"run": run, "t_train": rs, "horizon": horizon, "in_ch": in_ch,
              "levels": L, "grid": [nlat, nlon], "Z_km": (Z / 1000).round(3).tolist(),
              "tide": {}, "surface": {}, "upper": {}, "log_check": {}, "config_used": {
                  "mass_fix": mass_fix, "inference_hypervisc": inf_hv, "hypervisc": hypervisc,
                  "clip_inc": clip_inc, "solar": solar, "time_origin": origin, "data_start_hour": dsh}}

    print("\n===== Tables A2/A3 (tides, equatorial band +-21.5 deg, amplitude K / phase h) =====")
    for h, k in ktide:
        tp, tt_ = tide_block(tpred[k]), tide_block(ttruth[k])
        result["tide"][f"{Z[k]/1000:.1f}km"] = {"lev": int(k), "requested_km": h,
                                                "pred": tp, "truth": tt_}
        print(f"-- {Z[k]/1000:.1f} km (requested {h}) --")
        for nm in ("D0", "DW1", "SW2", "DE3"):
            print(f"  {nm:3s} pred {tp[nm]['amp_eq']:6.2f} K  phase_lt {tp[nm]['phase_lt_eq_h']:5.2f} h "
                  f"| truth {tt_[nm]['amp_eq']:6.2f} K  phase_lt {tt_[nm]['phase_lt_eq_h']:5.2f} h "
                  f"| ratio {tp[nm]['amp_eq']/max(tt_[nm]['amp_eq'],1e-9):.2f}")

    print("\n===== Table S1 (surface, t+%d h, uncalibrated / calibrated) =====" % horizon)
    for nm in ("u", "v", "T", "Ps"):
        r = stat[nm]["rmse"][-1]; c = stat[nm]["corr"][-1]
        f = float(cal.get(nm, 1.0)) if nm in ("u", "v", "T") else 1.0
        rc = r * f if nm in ("u", "v", "T") else r
        result["surface"][nm] = {"rmse": r, "bias": stat[nm]["bias"][-1], "acc": c,
                                 "rmse_calibrated": rc, "cal_factor": f, "lead_h": horizon}
        print(f"  {nm:2s}  RMSE {r:6.2f} -> calibrated {rc:6.2f}   ACC {c:+.3f}")

    lead = a.lead_upper
    i = min(lead, horizon) - 1
    print(f"\n===== Table A1 (by altitude, t+{min(lead,horizon)} h) =====")
    for k, (h, kk) in zip(kspot.values(), kspot.items()):
        ent = {}
        row = []
        for nm in ("u", "v", "T"):
            d = upstat[k][nm]
            ent[nm] = {m: d[m][i] for m in ("rmse", "bias", "crmse", "corr")}
            row.append(f"{nm}: RMSE {ent[nm]['rmse']:6.2f} bias {ent[nm]['bias']:+7.2f} "
                       f"cRMSE {ent[nm]['crmse']:6.2f} ACC {ent[nm]['corr']:+.3f}")
        result["upper"][f"{h:.1f}km"] = {"lev": int(k), "height_km": float(Z[k]/1000), **ent}
        print(f"  {h:5.1f} km (level {k}, {Z[k]/1000:.1f} km)")
        for nm in ("u", "v", "T"):
            e = ent[nm]
            print(f"     {nm}: RMSE {e['rmse']:7.2f}  bias {e['bias']:+8.2f}  cRMSE {e['crmse']:7.2f}  ACC {e['corr']:+.3f}")

    print("\n===== Check against the training log (truth at rs+step, calibrated; should match the surface table of the .out file) =====")
    ok = True
    for lead_ in (12, 24, 48, 72, 96, 120, 144):
        if lead_ > horizon:
            continue
        vals = []
        for nm in ("u", "v", "T", "Ps"):
            f = float(cal.get(nm, 1.0)) if nm in ("u", "v", "T") else 1.0
            vals.append(logchk[nm][lead_ - 1] * f)
        result["log_check"][f"t+{lead_}h"] = dict(zip(("u", "v", "T", "Ps"), vals))
        print("  t+%3dh  u=%5.2f v=%5.2f T=%5.2f Ps=%5.2f" % (lead_, *vals))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[i] saved: {out_path}")


if __name__ == "__main__":
    main()
