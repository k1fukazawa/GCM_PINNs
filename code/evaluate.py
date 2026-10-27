#!/usr/bin/env python3
"""\
Evaluation code for every metric reported in the paper: 143 h autoregressive
rollout from a checkpoint, RMSE, bias, centered RMSE, ACC, pattern correlation,
vertical profiles, persistence and climatology baselines, mass drift and the
two-dimensional Fourier decomposition of atmospheric tides. See README.md.
"""


import argparse, json, math, os, time
import numpy as np
import torch
from eval_extra import Full3DStepper

def smooth2d(f):

    # Autoregressive rollout over R steps; the loss is accumulated over all steps.
    left = torch.roll(f, 1, dims=-1)
    right = torch.roll(f, -1, dims=-1)
    up = torch.cat([f[..., 1:, :], f[..., -1:, :]], dim=-2)
    dn = torch.cat([f[..., :1, :], f[..., :-1, :]], dim=-2)
    return 0.25 * (left + right + up + dn)

def build_argparse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./npy")
    ap.add_argument("--versions", type=int, nargs="+", default=[])
    ap.add_argument("--run_dirs", nargs="+", default=[],
                    help="full3d.pt|final.pt  results.json ")
    # Longitude-time Fourier decomposition used for the atmospheric-tide diagnostics.
    ap.add_argument("--tide_heights", type=float, nargs="+", default=[100.0, 120.0, 130.0, 150.0])
    ap.add_argument("--t0_ut", type=float, default=0.0, help=" index 0  UT [h]")
    ap.add_argument("--lon0_deg", type=float, default=0.0, help=" index 0  [deg E]")
    ap.add_argument("--clamp_diag", type=int, default=1)
    ap.add_argument("--ckpt_prefer", choices=["final", "full3d"], default="final",
                    help="final  EMA "
                         ""
                         "")
    ap.add_argument("--timing_repeats", type=int, default=5)
    ap.add_argument("--profile_leads", type=int, nargs="+", default=[24, 72, 143],
                    help=" 48  [h]0 ")
    ap.add_argument("--ckpt_tpl", default="./out/v{V}/full3d.pt")
    ap.add_argument("--results_tpl", default="./out/v{V}/results.json")
    ap.add_argument("--out_tpl", default="./out/v{V}/extra3")
    ap.add_argument("--t_train", type=int, default=600)
    ap.add_argument("--heights", type=float, nargs="+", default=[0.0, 100.0, 120.0, 150.0, 200.0, 300.0])
    ap.add_argument("--tide_height", type=float, default=100.0)
    ap.add_argument("--horizon", type=int, default=143,
                    help=" [h]0 = ")
    ap.add_argument("--max_steps", type=int, default=-1, help="rollout ")
    # Remove the global mean of the surface-pressure increment at every step.
    ap.add_argument("--drift_series", type=int, default=1, help="mass fixer ")
    ap.add_argument("--plot", type=int, default=1)
    ap.add_argument("--threads", type=int, default=112)
    return ap

def main():
    args = build_argparse().parse_args()
    torch.set_num_threads(args.threads)
    gu = np.load(f"{args.dir}/gu.npy", mmap_mode="r")
    gv = np.load(f"{args.dir}/gv.npy", mmap_mode="r")
    gw = np.load(f"{args.dir}/gw.npy", mmap_mode="r")
    gt = np.load(f"{args.dir}/gt.npy", mmap_mode="r")
    gp = np.load(f"{args.dir}/gp.npy", mmap_mode="r")
    gz = np.load(f"{args.dir}/multilayer_gz.npy", mmap_mode="r")
    LEV, LAT, LON = gu.shape[1], gu.shape[2], gu.shape[3]
    H = gu.shape[0]
    rs = args.t_train
    horizon = args.horizon if args.horizon > 0 else H - rs - 1
    horizon = min(horizon, H - rs - 1)
    if args.max_steps > 0:
        horizon = min(horizon, args.max_steps)
    assert horizon > 0, f"horizon={horizon}H={H}, t_train={rs}--t_train "
    print(f"[i] H={H}, t_train={rs}, horizon={horizon}")

    levels = 48
    lev_idx = np.unique(np.round(np.linspace(0, LEV - 1, levels))).astype(int)
    L = len(lev_idx); nlat, nlon = LAT, LON
    Z = np.asarray(gz[:, lev_idx, :, :].mean(axis=(0, 2, 3)), dtype=np.float64)
    zs_full = np.asarray(gz[0, 0, :, :], dtype=np.float32)
    zs_mean, zs_std = float(zs_full.mean()), float(zs_full.std() + 1e-9)
    zsfc_t = torch.tensor(((zs_full - zs_mean) / zs_std).astype(np.float32))[None, None, None]
    zfeat = (Z - Z.mean()) / (Z.std() + 1e-9)
    zfeat_t = torch.tensor(zfeat.astype(np.float32))
    z_t = torch.tensor(Z.astype(np.float32))
    phi = -np.radians(180 * ((np.arange(nlat, dtype=np.float64) + 0.5 - 32) / 64))
    T_MIN_K, T_MAX_K = 120.0, 1500.0
    PS_MIN_HPA, PS_MAX_HPA = 300.0, 1200.0

    T_tr = min(rs, H)
    def chunk_stats(getter, nlev):
        m = np.zeros(nlev); s2 = np.zeros(nlev); n = 0
        for s0 in range(0, T_tr, 48):
            e = min(s0 + 48, T_tr)
            a = np.stack([getter(h) for h in range(s0, e)]).astype(np.float64)
            m += a.mean(axis=(0, 2, 3)) * (e - s0)
            s2 += (a * a).mean(axis=(0, 2, 3)) * (e - s0)
            n += e - s0
        m /= n; var = np.maximum(s2 / n - m * m, 0.0)
        return m.astype(np.float32), (np.sqrt(var) + 1e-8).astype(np.float32)
    def sub(h, arr): return np.ascontiguousarray(arr[h][lev_idx])
    mU, sU = chunk_stats(lambda h: sub(h, gu), L)
    mV, sV = chunk_stats(lambda h: sub(h, gv), L)
    mW, sW = chunk_stats(lambda h: sub(h, gw), L)
    mT, sT = chunk_stats(lambda h: sub(h, gt), L)
    # Clip the normalized state increment to keep the autoregressive rollout stable.
    PSin = lambda h: np.log(np.clip(np.ascontiguousarray(gp[h]), 1.0, None))
    mPs_n = 0.0; sPs_n = 0.0; n = 0
    for s0 in range(0, T_tr, 48):
        e = min(s0 + 48, T_tr)
        a = np.stack([PSin(h) for h in range(s0, e)]).astype(np.float64)
        mPs_n += a.mean() * (e - s0); sPs_n += (a * a).mean() * (e - s0); n += e - s0
    mP = float(mPs_n / n); sP = float(math.sqrt(max(sPs_n / n - mP * mP, 0.0)) + 1e-8)

    def to_global(m, s):
        mg = float(m.mean()); vg = float((s.astype(np.float64) ** 2 + m.astype(np.float64) ** 2).mean() - mg * mg)
        sg = math.sqrt(max(vg, 0.0)) + 1e-8
        return np.full_like(m, mg), np.full_like(s, sg)
    mU_g, sU_g = to_global(mU, sU); mV_g, sV_g = to_global(mV, sV)
    mW_g, sW_g = to_global(mW, sW); mT_g, sT_g = to_global(mT, sT)

    t0 = time.time()
    def time_mean(getter):
        m = np.zeros((L, nlat, nlon), dtype=np.float64)
        for s0 in range(0, T_tr, 48):
            e = min(s0 + 48, T_tr)
            a = np.stack([getter(h) for h in range(s0, e)]).astype(np.float64)
            m += a.sum(axis=0)
        return (m / T_tr).astype(np.float32)
    clim_u = time_mean(lambda h: sub(h, gu)); clim_v = time_mean(lambda h: sub(h, gv))
    clim_T = time_mean(lambda h: sub(h, gt))
    print(f"[i]  {time.time()-t0:.1f}s")

    wcos = np.cos(phi).astype(np.float64); wcos /= wcos.sum()
    wcos2 = wcos[:, None]
    kspot_nat = [int(np.argmin(np.abs(Z - h * 1000.0))) for h in args.heights]
    ktide_nat = int(np.argmin(np.abs(Z - args.tide_height * 1000.0)))
    ktide_map = []
    for _h in args.tide_heights:
        _k = int(np.argmin(np.abs(Z - _h * 1000.0)))
        if _k not in [k for _, k in ktide_map]:
            ktide_map.append((_h, _k))
    ktides = [k for _, k in ktide_map]
    ut_off = float((rs + 1 + args.t0_ut) % 24.0)
    print(f"[i] tide levels: {[(k, round(Z[k]/1000,1)) for k in ktides]}, step0 UT={ut_off}")
    zmeso = (Z > 70e3)[:, None, None]
    print(f"[i] native spot levels: {list(zip(args.heights, kspot_nat))}, "
          f"tide level k={ktide_nat} ({Z[ktide_nat]/1000:.1f} km)")

    def load48(h):
        u = np.ascontiguousarray(gu[h][lev_idx], dtype=np.float32)
        v = np.ascontiguousarray(gv[h][lev_idx], dtype=np.float32)
        t = np.ascontiguousarray(gt[h][lev_idx], dtype=np.float32)
        p = np.ascontiguousarray(gp[h], dtype=np.float32)
        return u, v, t, p

    def acc(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        return float((a * b).sum()) / d if d > 0 else 0.0

    def mset(P, Tr, B0, Bc):

        if P.ndim == 2:
            P = np.broadcast_to(P, Tr.shape)
        out = {k: [] for k in ("rmse", "bias", "crmse", "acc_gm", "acc_clim")}
        for i in range(Tr.shape[0]):
            p_ = np.asarray(P[i], dtype=np.float64); t_ = np.asarray(Tr[i], dtype=np.float64)
            d = p_ - t_
            rm = float(np.sqrt((d * d).mean())); bi = float(d.mean())
            out["rmse"].append(rm); out["bias"].append(bi)
            out["crmse"].append(math.sqrt(max(rm * rm - bi * bi, 0.0)))
            out["acc_gm"].append(acc(p_, t_))
            out["acc_clim"].append(acc(p_ - B0, t_ - B0))
        return out

    def tmean(d): return {k: float(np.mean(v)) for k, v in d.items()}
    def pick(d, lead):
        i = min(lead, len(d["rmse"])) - 1
        return {k: float(v[i]) for k, v in d.items()}

    def tide_analysis(F):

        Tn = F.shape[0]; tt = np.arange(Tn, dtype=np.float64)
        A = {}; PH = {}
        comps = {"D0": (0, 1), "DW1": (1, 1), "SW2": (2, 2), "DE3": (-3, 1)}
        for name, (s, om) in comps.items():
            amp = np.zeros(F.shape[1]); tmax = np.zeros(F.shape[1])
            for la in range(F.shape[1]):
                y = F[:, la, :].astype(np.float64)

                tt_c = tt - tt.mean()
                den = float((tt_c * tt_c).sum())
                slope = (y * tt_c[:, None]).sum(axis=0) / den
                inter = y.mean(axis=0)
                y = y - (inter[None, :] + slope[None, :] * tt_c[:, None])
                Ff = np.fft.fft2(y) / (Tn * y.shape[1])
                kx_s = s % y.shape[1]
                kt = int(round(om * Tn / 24.0)) % Tn
                Fp = Ff[kt, kx_s]; Fm = Ff[(Tn - kt) % Tn, kx_s]
                Fc = Fp if abs(Fp) >= abs(Fm) else Fm
                amp[la] = 2.0 * abs(Fc)
                psi = np.angle(Fc)
                tmax[la] = np.mod(-psi / (2 * np.pi * om), 1.0 / om) * 24.0
            A[name] = amp; PH[name] = tmax
        return A, PH

    gser_truth = [float((np.asarray(gp[rs + s + 1], dtype=np.float64) * wcos2).sum())
                  for s in range(horizon)]

    runs = [(V, args.ckpt_tpl.format(V=V), args.results_tpl.format(V=V), args.out_tpl.format(V=V))
            for V in args.versions]
    for dr in args.run_dirs:
        dr = dr.rstrip("/")
        runs.append((os.path.basename(dr), f"{dr}/full3d.pt", f"{dr}/results.json", f"{dr}/extra3"))
    assert runs, "--versions  --run_dirs "
    import socket, statistics
    envinfo = dict(host=socket.gethostname(), torch_threads=torch.get_num_threads(),
                   cpu_count=os.cpu_count(),
                   **{k: os.environ.get(k) for k in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES", "SLURM_NTASKS",
                                                     "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS", "SLURM_JOB_ID")})
    print("[i] env:", envinfo)
    results = {}
    for V, ck_tpl, res_tpl, out_dir in runs:
        print(f"\n########## {V} ##########")
        cfg = {}
        if os.path.exists(res_tpl):
            cfg = json.load(open(res_tpl)).get("config", {})
        ck_path = None
        _cands = [ck_tpl, ck_tpl.replace("full3d.pt", "final.pt")]
        if args.ckpt_prefer == "final":
            _cands.reverse()
        for cand in _cands:
            if os.path.exists(cand):
                ck_path = cand; break
        if ck_path is None:
            print(f"[skip] v{V}: checkpoint "); continue
        try:
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(ck_path, map_location="cpu")
        # Exponential moving average of the weights; M9 is evaluated with these weights.
        state = ck["ema"] if (isinstance(ck, dict) and ck.get("ema") is not None) else\
                (ck["model"] if isinstance(ck, dict) and "model" in ck else ck)

        try:
            in_ch = int(state["enc.0.conv.weight"].shape[1])
        except Exception:
            in_ch = 7 if bool(cfg.get("use_terrain", True)) else 6
        level_norm = "level_weight_scale_km" in cfg
        mUs, sUs = (mU, sU) if level_norm else (mU_g, sU_g)
        mVs, sVs = (mV, sV) if level_norm else (mV_g, sV_g)
        mWs, sWs = (mW, sW) if level_norm else (mW_g, sW_g)
        mTs, sTs = (mT, sT) if level_norm else (mT_g, sT_g)
        weights = "unknown"
        model = Full3DStepper(in_ch=in_ch, ch=tuple(cfg.get("ch", [24, 48, 32])))
        if isinstance(ck, dict) and "ema" in ck and float(cfg.get("ema_decay", 0.0) or 0.0) > 0 and ck["ema"] is not None:
            model.load_state_dict(ck["ema"]); weights = "ema"
        elif isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"]); weights = "model"
        else:
            model.load_state_dict(ck); weights = "state_dict"
        print(f"[i] checkpoint={ck_path} weights={weights}")
        model.eval()
        mass_fix = bool(cfg.get("mass_fix", False))
        hypervisc = float(cfg.get("hypervisc", 0.05) or 0.0)
        inf_hv = (True if "inference_hypervisc" not in cfg else bool(cfg["inference_hypervisc"])) and hypervisc > 0
        clip_inc = float(cfg.get("clip_inc", 1.0))
        mUt = torch.tensor(mUs).reshape(1, 1, L, 1, 1); sUt = torch.tensor(sUs).reshape(1, 1, L, 1, 1)
        mVt = torch.tensor(mVs).reshape(1, 1, L, 1, 1); sVt = torch.tensor(sVs).reshape(1, 1, L, 1, 1)
        mWt = torch.tensor(mWs).reshape(1, 1, L, 1, 1); sWt = torch.tensor(sWs).reshape(1, 1, L, 1, 1)
        mTt = torch.tensor(mTs).reshape(1, 1, L, 1, 1); sTt = torch.tensor(sTs).reshape(1, 1, L, 1, 1)
        TMiN = (T_MIN_K - mTt) / sTt; TMaX = (T_MAX_K - mTt) / sTt
        PMiN = (math.log(PS_MIN_HPA) - mP) / sP; PMaX = (math.log(PS_MAX_HPA) - mP) / sP
        use_terrain = in_ch >= 7

        def build_input(u, v, w, t, p):
            ps = p[:, :, None].expand(1, 1, L, nlat, nlon)
            zf = zfeat_t[None, None, :, None, None].expand(1, 1, L, nlat, nlon)
            chans = [u, v, w, t, ps, zf]
            if use_terrain:
                chans.append(zsfc_t.expand(1, 1, L, nlat, nlon))
            return torch.cat(chans, dim=1)

        def lev_metrics(P, T_, C):

            d = P - T_
            n = P.shape[1] * P.shape[2]
            rmse = np.sqrt((d * d).mean(axis=(1, 2))); bias = d.mean(axis=(1, 2))
            crmse = np.sqrt(np.maximum(rmse * rmse - bias * bias, 0.0))
            def corr(a, b):
                a = a - a.mean(axis=(1, 2), keepdims=True); b = b - b.mean(axis=(1, 2), keepdims=True)
                den = np.sqrt((a * a).sum(axis=(1, 2)) * (b * b).sum(axis=(1, 2)))
                out = np.zeros(a.shape[0]); ok = den > 0
                out[ok] = (a[ok] * b[ok]).sum(axis=(1, 2)) / den[ok]
                return out
            return dict(rmse=rmse.tolist(), bias=bias.tolist(), crmse=crmse.tolist(),
                        acc_gm=corr(P, T_).tolist(), acc_clim=corr(P - C, T_ - C).tolist())
        prof = {} if (args.profile_leads and args.profile_leads[0] > 0) else None
        u_init48 = v_init48 = t_init48 = None

        @torch.inference_mode()
        def rollout(apply_mass_fix, collect=True, diag=False):

            u0, v0, t0_, p0 = load48(rs)
            nonlocal u_init48, v_init48, t_init48
            if prof is not None and u_init48 is None:
                u_init48, v_init48, t_init48 = (a.astype(np.float64) for a in (u0, v0, t0_))
            nrm = lambda x, mm, ss: (torch.tensor(x[None, None]) - mm) / ss
            u_n = nrm(u0, mUt, sUt); v_n = nrm(v0, mVt, sVt)
            w_n = nrm(gw[rs][lev_idx], mWt, sWt)
            t_n = nrm(t0_, mTt, sTt)
            p_n = (torch.tensor(np.log(np.clip(p0, 1.0, None))[None, None]) - mP) / sP
            gser = []
            fields = ({k: {"u": [], "v": [], "T": []} for k in kspot_nat} if collect else None)
            Ttide = ({k: np.zeros((horizon, nlat, nlon), dtype=np.float32) for k in ktides} if collect else None)
            Pser = [] if collect else None
            dg = dict(clamp_frac=[], inc1=None) if diag else None
            clamp_step = None; clamp_lev = None
            for step in range(horizon):
                x = build_input(u_n, v_n, w_n, t_n, p_n)
                d3d, d2d = model(x)
                if inf_hv:
                    d3d = d3d - hypervisc * (d3d - smooth2d(d3d))
                    d2d = d2d - hypervisc * (d2d - smooth2d(d2d))
                if diag and step == 0:
                    dT = d3d[0, 3].detach().numpy().astype(np.float64)
                    dg["inc1"] = dict(
                        min_sigma=dT.min(axis=(1, 2)).tolist(), mean_sigma=dT.mean(axis=(1, 2)).tolist(),
                        rms_sigma=np.sqrt((dT * dT).mean(axis=(1, 2))).tolist(),
                        frac_clip_lo=(dT <= -clip_inc).mean(axis=(1, 2)).tolist() if clip_inc > 0 else None,
                        min_K=(dT.min(axis=(1, 2)) * sTs).tolist())
                if clip_inc > 0:
                    d3d = torch.clamp(d3d, -clip_inc, clip_inc); d2d = torch.clamp(d2d, -clip_inc, clip_inc)
                if apply_mass_fix and mass_fix:
                    d2d = d2d - d2d.mean()
                u_n = u_n + d3d[:, 0:1]; v_n = v_n + d3d[:, 1:2]; w_n = w_n + d3d[:, 2:3]
                t_n = torch.clamp(t_n + d3d[:, 3:4], min=TMiN, max=TMaX)
                p_n = torch.clamp(p_n + d2d, min=PMiN, max=PMaX)
                U48 = u_n[0, 0].detach().numpy() * sUs[:, None, None] + mUs[:, None, None]
                V48 = v_n[0, 0].detach().numpy() * sVs[:, None, None] + mVs[:, None, None]
                T48 = t_n[0, 0].detach().numpy() * sTs[:, None, None] + mTs[:, None, None]
                ps = np.exp(p_n[0, 0].detach().numpy().astype(np.float64) * sP + mP)
                gser.append(float((ps * wcos2).sum()))
                if collect:
                    for k in kspot_nat:
                        fields[k]["u"].append(U48[k].astype(np.float32))
                        fields[k]["v"].append(V48[k].astype(np.float32))
                        fields[k]["T"].append(T48[k].astype(np.float32))
                    for k in ktides:
                        Ttide[k][step] = T48[k]
                    Pser.append((ps / 100.0 if ps.mean() > 2000 else ps).astype(np.float32))
                if diag:
                    dg["clamp_frac"].append((T48 <= T_MIN_K + 1e-3).mean(axis=(1, 2)).tolist())
                if prof is not None and collect and (step + 1) in args.profile_leads:
                    tu, tv, tt_, _ = load48(rs + step + 1)
                    ent = {"lead_h": step + 1, "pred": {}, "persist": {}, "clim": {}}
                    for var, P48, I48, C48 in (("u", U48, u_init48, clim_u), ("v", V48, v_init48, clim_v),
                                               ("T", T48, t_init48, clim_T)):
                        Td = tt_ if var == "T" else (tu if var == "u" else tv)
                        ent["pred"][var] = lev_metrics(P48.astype(np.float64), Td.astype(np.float64),
                                                       np.asarray(C48, dtype=np.float64))
                        ent["persist"][var] = lev_metrics(I48, Td.astype(np.float64), np.asarray(C48, dtype=np.float64))
                        ent["clim"][var] = lev_metrics(np.asarray(C48, dtype=np.float64), Td.astype(np.float64),
                                                       np.asarray(C48, dtype=np.float64))
                    prof[step + 1] = ent
                if clamp_step is None:
                    hit = np.where(zmeso & (T48 <= T_MIN_K + 1e-3))[0]
                    if hit.size:
                        clamp_step = step + 1; clamp_lev = int(hit[0])
            return gser, fields, Ttide, clamp_step, clamp_lev, Pser, dg

        t0 = time.time()
        gser_on, spot_pred, Ttide_pred, clamp_step, clamp_lev, Ps_pred, dg = rollout(
            True, collect=True, diag=bool(args.clamp_diag))
        gser_off = None
        if args.drift_series and mass_fix:
            gser_off = rollout(False, collect=False)[0]

        timing = None
        if args.timing_repeats > 0:
            ts_ = []
            for _ in range(args.timing_repeats):
                t1 = time.perf_counter(); rollout(True, collect=False); ts_.append(time.perf_counter() - t1)
            timing = dict(seconds=ts_, median=float(statistics.median(ts_)),
                          per_step_median=float(statistics.median(ts_)) / horizon,
                          note="rollout  Ps ", **envinfo)
            print(f"[i] timing median {timing['median']:.2f}s ({args.timing_repeats} )")
        print(f"[i] rollout  {time.time()-t0:.1f}s (mass_fix={mass_fix}, inf_hv={inf_hv}, in_ch={in_ch})")

        t0 = time.time()
        spot_truth = {k: {"u": [], "v": [], "T": []} for k in kspot_nat}
        Ttide_truth = {k: np.zeros((horizon, nlat, nlon), dtype=np.float32) for k in ktides}
        Ps_truth = []
        for step in range(horizon):
            u, v, t, p = load48(rs + step + 1)
            for k in kspot_nat:
                spot_truth[k]["u"].append(u[k]); spot_truth[k]["v"].append(v[k]); spot_truth[k]["T"].append(t[k])
            for k in ktides:
                Ttide_truth[k][step] = t[k]
            Ps_truth.append(p / 100.0 if p.mean() > 2000 else p)
        u0, v0, t0_, p0_ = load48(rs)
        init_field = {k: {"u": u0[k], "v": v0[k], "T": t0_[k]} for k in kspot_nat}
        print(f"[i]  {time.time()-t0:.1f}s")

        native = {}
        for h, k in zip(args.heights, kspot_nat):
            native[f"{h:.0f}km"] = dict(lev=k, height_km=float(Z[k] / 1000.0))
            for var, clim in (("u", clim_u), ("v", clim_v), ("T", clim_T)):
                P = np.stack(spot_pred[k][var]).astype(np.float64)
                Tr = np.stack(spot_truth[k][var]).astype(np.float64)
                B0 = clim[k].astype(np.float64)
                mp_ = mset(P, Tr, B0, None)
                ms_ = mset(init_field[k][var], Tr, B0, None)
                mc_ = mset(B0, Tr, B0, None)
                summary = {}
                for tag, lead in (("t24h", 24), ("t72h", 72), ("tlast", horizon)):
                    summary[tag] = {"pred": pick(mp_, lead), "persist": pick(ms_, lead), "clim": pick(mc_, lead)}
                summary["tmean"] = {"pred": tmean(mp_), "persist": tmean(ms_), "clim": tmean(mc_)}
                native[f"{h:.0f}km"][var] = dict(
                    leads_h=list(range(1, horizon + 1)),
                    pred={"series": mp_, "summary": {t: summary[t]["pred"] for t in summary}},
                    persist={"series": ms_, "summary": {t: summary[t]["persist"] for t in summary}},
                    clim={"series": mc_, "summary": {t: summary[t]["clim"] for t in summary}})

        P = np.stack(Ps_pred).astype(np.float64); Tr = np.stack(Ps_truth).astype(np.float64)
        p0h = (p0_ / 100.0 if p0_.mean() > 2000 else p0_).astype(np.float64)
        climP = np.mean([np.asarray(gp[h], dtype=np.float64) for h in range(0, T_tr)], axis=0)
        climP = climP / 100.0 if climP.mean() > 2000 else climP
        mp_ = mset(P, Tr, climP, None); ms_ = mset(p0h, Tr, climP, None); mc_ = mset(climP, Tr, climP, None)
        if "0km" in native: native["0km"]["Ps"] = dict(leads_h=list(range(1, horizon + 1)),
            pred={"series": mp_, "summary": {t: pick(mp_, l) for t, l in (("t24h", 24), ("t72h", 72), ("tlast", horizon))}},
            persist={"series": ms_, "summary": {t: pick(ms_, l) for t, l in (("t24h", 24), ("t72h", 72), ("tlast", horizon))}},
            clim={"series": mc_, "summary": {t: pick(mc_, l) for t, l in (("t24h", 24), ("t72h", 72), ("tlast", horizon))}})

        clampdiag = None
        if dg is not None:
            kk = np.where(Z > 70e3)[0]
            tmin_tr = np.full(L, np.inf); tmin_va = np.full(L, np.inf); incr = np.zeros(L); n_ = 0
            prev = None
            for h in range(0, rs + horizon + 1):
                tt_ = np.asarray(gt[h][lev_idx], dtype=np.float64)
                if h < rs: tmin_tr = np.minimum(tmin_tr, tt_.min(axis=(1, 2)))
                else: tmin_va = np.minimum(tmin_va, tt_.min(axis=(1, 2)))
                if prev is not None and h <= rs:
                    d_ = tt_ - prev; incr += (d_ * d_).mean(axis=(1, 2)); n_ += 1
                prev = tt_
            incr_rms = np.sqrt(incr / max(n_, 1))
            cf = np.array(dg["clamp_frac"])
            clampdiag = dict(levels_km=(Z[kk] / 1000.0).tolist(), lev=kk.tolist(),
                truth_Tmin_train_K=tmin_tr[kk].tolist(), truth_Tmin_valid_K=tmin_va[kk].tolist(),
                # Normalization with statistics of the training period.
                norm_mean_K=np.asarray(mTs)[kk].tolist(), norm_std_K=np.asarray(sTs)[kk].tolist(),
                dist_to_Tmin_sigma=((np.asarray(mTs) - T_MIN_K) / np.asarray(sTs))[kk].tolist(),
                truth_inc_rms_K=incr_rms[kk].tolist(), truth_inc_rms_sigma=(incr_rms / np.asarray(sTs))[kk].tolist(),
                step1_inc={k_: (np.asarray(v_)[kk].tolist() if v_ is not None else None) for k_, v_ in dg["inc1"].items()},
                clamp_frac_by_step={str(s_ + 1): cf[s_, kk].tolist() for s_ in list(range(min(24, horizon))) + [min(72, horizon) - 1, horizon - 1]},
                clip_inc=clip_inc,
                note="dist_to_Tmin_sigma = ( − 120 K)/σtruth_inc_rms_sigma  clip_inc "
                     "")

        clamp = dict(step=clamp_step, height_km=float(Z[clamp_lev] / 1000.0) if clamp_lev is not None else None,
                     note=" T  120 K  step1 ")

        band = np.abs(np.degrees(phi)) <= 21.5
        def circ_mean_h(x, period):
            a = np.asarray(x) * 2 * np.pi / period
            return float(np.mod(np.angle(np.exp(1j * a).mean()) * period / (2 * np.pi), period))
        per = {"D0": 24.0, "DW1": 24.0, "SW2": 12.0, "DE3": 24.0}
        tides = {}
        for req_h, k in ktide_map:
            Atr, Ptr = tide_analysis(Ttide_truth[k]); Apr, Ppr = tide_analysis(Ttide_pred[k])
            td = {"height_km": float(Z[k] / 1000.0), "requested_km": float(req_h), "clamp_frac_last": (None if dg is None else float(np.array(dg["clamp_frac"])[-1, k]))}
            for src, (A, Pp) in (("truth", (Atr, Ptr)), ("pred", (Apr, Ppr))):
                td[src] = {}
                for name in A:
                    rel = Pp[name]
                    ut = np.mod(rel + ut_off, per[name])
                    ent = dict(amp_eq=float(A[name][band].mean()),
                               tmax_ut_lon0_eq_h=circ_mean_h(ut[band], per[name]),
                               amp_lat=A[name].tolist(), tmax_ut_lon0_lat=ut.tolist())
                    if name in ("DW1", "SW2"):
                        lt = np.mod(ut + args.lon0_deg / 15.0, per[name])
                        ent["tmax_lt_eq_h"] = circ_mean_h(lt[band], per[name]); ent["tmax_lt_lat"] = lt.tolist()
                    td[src][name] = ent
            tides[f"{req_h:.0f}km"] = td
        tide = tides.get(f"{Z[ktide_nat]/1000:.0f}km")
        drift = {"pred": gser_on, "truth": gser_truth, "mass_fix": mass_fix}
        if gser_off is not None:
            drift["pred_nofix"] = gser_off
        out = dict(version=V, ckpt=dict(path=str(ck_path), weights=weights),
                   config_used={k: cfg.get(k) for k in ("levels", "stride", "ch", "mass_fix", "inference_hypervisc",
                                                        "ema_decay", "use_terrain", "seed", "tag")},
                   in_ch=in_ch, level_norm=level_norm,
                   native_levels_lev_idx=lev_idx.tolist(),
                   native_heights_km=(Z[kspot_nat] / 1000.0).tolist(),
                   native=native, clamp=clamp, clamp_diag=clampdiag, tide=tide, tides=tides, timing=timing,
                   profile=(dict(levels=len(Z), heights_km=(Z / 1000.0).tolist(),
                                 lev_idx=lev_idx.tolist(), by_lead={str(k): v for k, v in prof.items()},
                                 note="48 RMSE RMSEACC()ACC()")
                            if prof else None),
                   tide_convention=dict(step0_ut=ut_off, t0_ut=args.t0_ut, lon0_deg=args.lon0_deg,
                                        note="tmax_ut_lon0  index 0  UTDW1/SW2  LT "
                                             "D0/DE3  UT "),
                   drift_series=drift,
                   tide_level_km=float(Z[ktide_nat] / 1000.0), horizon=horizon)
        outdir = out_dir
        os.makedirs(outdir, exist_ok=True)
        json.dump(out, open(f"{outdir}/results_extra3.json", "w"), ensure_ascii=False, indent=1)
        print(f"[i] saved {outdir}/results_extra3.json")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            lat = np.degrees(phi)
            fig, axes = plt.subplots(2, len(tides), figsize=(3.6 * len(tides), 6.2), squeeze=False)
            for j, (lk, td) in enumerate(tides.items()):
                for comp, c in (("DW1", "tab:red"), ("SW2", "tab:blue")):
                    axes[0, j].plot(lat, td["truth"][comp]["amp_lat"], '-', color=c, label=f'{comp} truth')
                    axes[0, j].plot(lat, td["pred"][comp]["amp_lat"], '--', color=c, label=f'{comp} pred')
                    axes[1, j].plot(lat, td["truth"][comp]["tmax_lt_lat"], '-', color=c)
                    axes[1, j].plot(lat, td["pred"][comp]["tmax_lt_lat"], '--', color=c)
                axes[0, j].set_title(f'{lk} amplitude [K]'); axes[1, j].set_title(f'{lk} max LT [h]')
                axes[1, j].set_xlabel('latitude'); axes[1, j].set_ylim(0, 24)
                for a in axes[:, j]: a.grid(alpha=0.3)
            axes[0, 0].legend(fontsize=7)
            fig.tight_layout(); fig.savefig(f"{outdir}/tide_DW1_SW2_levels.png", dpi=130); plt.close(fig)
            if clampdiag is not None:
                fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
                zk = clampdiag["levels_km"]
                ax[0].plot(clampdiag["truth_Tmin_train_K"], zk, 'k-', label='truth min (train)')
                ax[0].plot(clampdiag["norm_mean_K"], zk, 'b-', label='norm mean')
                ax[0].axvline(T_MIN_K, color='r', ls=':'); ax[0].set_xlabel('T [K]'); ax[0].legend(fontsize=7)
                ax[1].plot(clampdiag["truth_inc_rms_sigma"], zk, 'k-', label='truth 1h inc RMS')
                ax[1].plot(np.abs(clampdiag["step1_inc"]["min_sigma"]), zk, 'r-', label='|step1 min inc|')
                ax[1].plot(clampdiag["step1_inc"]["rms_sigma"], zk, 'r--', label='step1 inc RMS')
                ax[1].set_xscale('log'); ax[1].set_xlabel('[sigma]'); ax[1].legend(fontsize=7)
                for s_ in ("1", "6", "24", str(horizon)):
                    if s_ in clampdiag["clamp_frac_by_step"]:
                        ax[2].plot(clampdiag["clamp_frac_by_step"][s_], zk, label=f'step {s_}')
                ax[2].set_xlabel('clamped fraction'); ax[2].legend(fontsize=7)
                for a in ax: a.set_ylabel('height [km]'); a.grid(alpha=0.3)
                fig.tight_layout(); fig.savefig(f"{outdir}/clamp_diag.png", dpi=130); plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(1, horizon + 1), gser_truth, 'k-', label='GCM truth')
            ax.plot(range(1, horizon + 1), gser_on, 'r-',
                    label=f"PINN pred (mass fixer {'on' if mass_fix else 'off'})")
            if gser_off is not None:
                ax.plot(range(1, horizon + 1), gser_off, 'r--', alpha=0.6, label='PINN pred (mass fixer off)')
            ax.set_xlabel('lead time [h]'); ax.set_ylabel('global mean Ps [hPa]')
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(f"{outdir}/global_mean_Ps2.png", dpi=120); plt.close(fig)
        del model, spot_pred, spot_truth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
