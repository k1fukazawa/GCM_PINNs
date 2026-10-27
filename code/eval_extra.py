#!/usr/bin/env python3
"""\
Model definition shared by the training and evaluation scripts.
"""


import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R_D, OMEGA, R_EARTH, HOUR = 287.0, 7.2921e-5, 6.371e6, 3600.0
G, CP = 9.80665, 1004.0
COS_POLE_MIN = 5e-2
T_MIN_K, T_MAX_K = 120.0, 1500.0
PS_MIN_HPA, PS_MAX_HPA = 300.0, 1200.0

class ConvBlock3D(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv3d(cin, cout, k, padding=0)
    def forward(self, x):
        x = F.pad(x, (self.p, self.p, 0, 0, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.p, self.p, self.p, self.p), mode="replicate")
        x = x.to(memory_format=torch.channels_last_3d)
        return F.silu(self.conv(x))

class Full3DStepper(nn.Module):
    def __init__(self, in_ch=7, ch=(24, 48, 32)):
        super().__init__()
        layers, prev = [], in_ch
        for c in ch:
            layers.append(ConvBlock3D(prev, c, 3)); prev = c
        self.enc = nn.Sequential(*layers)
        self.head3d = nn.Conv3d(prev, 4, 1)
        self.head2d = nn.Sequential(nn.Conv3d(prev, 16, 1), nn.SiLU())
        self.head2d_conv = nn.Conv2d(16, 1, 3, padding=0)
    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        f = self.enc(x)
        d3d = self.head3d(f)
        g = self.head2d(f).mean(dim=2)
        g = F.pad(g, (1, 1, 0, 0), mode='circular')
        g = F.pad(g, (0, 0, 1, 1), mode='replicate')
        d2d = self.head2d_conv(g)
        return d3d.contiguous(), d2d.contiguous()

def grad_lon(f, dtheta):
    g = torch.empty_like(f)
    g[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2 * dtheta)
    g[..., 0] = (f[..., 1] - f[..., -1]) / (2 * dtheta)
    g[..., -1] = (f[..., 0] - f[..., -2]) / (2 * dtheta)
    return g

def grad_lat(f, dphi):
    g = torch.zeros_like(f)
    g[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * dphi)
    g[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / dphi
    g[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / dphi
    return g

def grad_z(f, z):
    g = torch.zeros_like(f)
    g[..., 1:-1, :, :] = (f[..., 2:, :, :] - f[..., :-2, :, :]) / (z[2:] - z[:-2])[None, None, :, None, None]
    g[..., 0, :, :] = (f[..., 1, :, :] - f[..., 0, :, :]) / (z[1] - z[0])
    g[..., -1, :, :] = (f[..., -1, :, :] - f[..., -2, :, :]) / (z[-1] - z[-2])
    return g

def hypsometric_p(p_s, T, z):
    T_layer = 0.5 * (T[:, :, :-1] + T[:, :, 1:])
    T_layer = T_layer.clamp(min=T_MIN_K, max=T_MAX_K)
    dz = (z[1:] - z[:-1])[None, None, :, None, None]
    dlnp = -(G / R_D) * dz / T_layer
    cum = torch.cumsum(dlnp, dim=2).clamp(min=-80.0, max=0.0)
    lnp_s = torch.log(p_s + 1e-6)
    lnp = torch.cat([lnp_s[:, :, None], lnp_s[:, :, None] + cum], dim=2)
    return torch.exp(lnp)

def interp_levels(field48, h48, h150):

    # Clip the normalized state increment to keep the autoregressive rollout stable.
    idx = np.clip(np.searchsorted(h48, h150), 1, len(h48) - 1)
    lo = idx - 1
    w = (h150 - h48[lo]) / (h48[idx] - h48[lo] + 1e-9)
    w = np.clip(w, 0.0, 1.0)[:, None, None].astype(field48.dtype)
    return (1.0 - w) * field48[lo] + w * field48[idx]

def acc2d(p_, t_, w=None):
    if w is None:
        p0 = p_ - p_.mean(); t0 = t_ - t_.mean()
        den = math.sqrt((p0 * p0).sum() * (t0 * t0).sum()) + 1e-12
    else:
        sw = w.sum()
        p0 = p_ - (p_ * w).sum() / sw; t0 = t_ - (t_ * w).sum() / sw
        den = math.sqrt((p0 * p0 * w).sum() * (t0 * t0 * w).sum()) + 1e-12
    return float((p0 * t0 * (w if w is not None else 1.0)).sum() / den)

def level_metrics(p, t, w2, c=None):

    p = p.astype(np.float64); t = t.astype(np.float64)
    d = p - t
    rmse = np.sqrt((d * d).mean(axis=(1, 2))); bias = d.mean(axis=(1, 2))
    crmse = np.sqrt(np.maximum(rmse ** 2 - bias ** 2, 0.0))
    def acc(a, b, w=None):
        if w is None:
            a0 = a - a.mean(axis=(1, 2), keepdims=True); b0 = b - b.mean(axis=(1, 2), keepdims=True)
            return (a0 * b0).sum(axis=(1, 2)) / (np.sqrt((a0 ** 2).sum(axis=(1, 2)) * (b0 ** 2).sum(axis=(1, 2))) + 1e-12)
        a0 = a - (a * w).sum(axis=(1, 2), keepdims=True); b0 = b - (b * w).sum(axis=(1, 2), keepdims=True)
        return (a0 * b0 * w).sum(axis=(1, 2)) / (np.sqrt((a0 ** 2 * w).sum(axis=(1, 2)) * (b0 ** 2 * w).sum(axis=(1, 2))) + 1e-12)
    out = dict(rmse=rmse, bias=bias, crmse=crmse, acc_gm=acc(p, t),
               rmse_w=np.sqrt((d * d * w2).sum(axis=(1, 2))), bias_w=(d * w2).sum(axis=(1, 2)),
               acc_gm_w=acc(p, t, w2))
    if c is not None:
        c = c.astype(np.float64)
        out["acc_clim"] = acc(p - c, t - c, None); out["acc_clim_w"] = acc(p - c, t - c, w2)
    return {k: np.asarray(v).tolist() for k, v in out.items()}

def pick(md, k):
    return {kk: vv[k] for kk, vv in md.items()}

def rms_metrics(pred, truth, w=None):
    d = pred - truth
    if w is None:
        rmse = math.sqrt((d * d).mean()); bias = float(d.mean())
    else:
        sw = w.sum()
        rmse = math.sqrt((d * d * w).sum() / sw); bias = float((d * w).sum() / sw)
    crmse = math.sqrt(max(rmse * rmse - bias * bias, 0.0))
    return rmse, bias, crmse

def harmonic_fit(x, period_h, dt_h=1.0):

    n = len(x); t = np.arange(n) * dt_h
    om = 2 * np.pi / period_h
    c = np.exp(-1j * om * t)
    a = (x * c).sum() / n * 2.0
    amp = float(abs(a))
    ph = float(np.angle(a))
    tmax = float((ph / om) % period_h)
    return amp, tmax

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./npy", help="npy ")
    ap.add_argument("--versions", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12, 13])
    ap.add_argument("--ckpt_tpl", default="./out/v{V}/full3d.pt")
    ap.add_argument("--results_tpl", default="./out/v{V}/results.json")
    ap.add_argument("--out_tpl", default="./out/v{V}/extra")
    ap.add_argument("--t_train", type=int, default=600, help="[h]")
    ap.add_argument("--horizon", type=int, default=0, help="[h]0  H - t_train")
    # Autoregressive rollout over R steps; the loss is accumulated over all steps.
    ap.add_argument("--max_steps", type=int, default=0, help=">0  rollout ")
    ap.add_argument("--threads", type=int, default=0, help="0 ")
    ap.add_argument("--cal_train", action=argparse.BooleanOptionalAction, default=True,
                    help=" 24h 2.6  (b) ")
    ap.add_argument("--cal_hours", type=int, default=24)
    ap.add_argument("--heights", type=float, nargs="+", default=[100.0, 200.0, 300.0],
                    help="[km]")
    # Longitude-time Fourier decomposition used for the atmospheric-tide diagnostics.
    ap.add_argument("--tide_height", type=float, default=100.0)
    ap.add_argument("--plot", type=int, default=0, help="1 ")
    args = ap.parse_args()
    nthr = args.threads if args.threads > 0 else max(1, os.cpu_count() or 1)
    torch.set_num_threads(nthr)
    print(f"[i] torch={torch.__version__}, threads={nthr}")

    gu = np.load(f"{args.dir}/gu.npy", mmap_mode="r")
    gv = np.load(f"{args.dir}/gv.npy", mmap_mode="r")
    gw = np.load(f"{args.dir}/gw.npy", mmap_mode="r")
    gt = np.load(f"{args.dir}/gt.npy", mmap_mode="r")
    gp = np.load(f"{args.dir}/gp.npy", mmap_mode="r")
    gz = np.load(f"{args.dir}/multilayer_gz.npy", mmap_mode="r")
    LEV, LAT, LON = gu.shape[1], gu.shape[2], gu.shape[3]
    H = gu.shape[0]
    horizon = args.horizon if args.horizon > 0 else H - args.t_train

    max_lead = H - args.t_train - 1
    if horizon > max_lead:
        print(f"[w] horizon {horizon} > 利用可能な最大リード {max_lead} → {max_lead} に制限")
        horizon = max_lead
    if args.max_steps > 0:
        horizon = min(horizon, args.max_steps)
    print(f"[i] data: {H}h x {LEV}層 x {LAT}x{LON}, t_train={args.t_train}, horizon={horizon}")

    h150 = np.asarray(gz[:, :, :, :].mean(axis=(0, 2, 3)), dtype=np.float64)
    zs_full = np.asarray(gz[0, 0, :, :], dtype=np.float32)

    leads = list(range(1, horizon + 1))
    lead_keep = [h for h in [11, 23, 47, 71, 95, 119, horizon - 1] if h < horizon]
    kspot = {f"{hkm:.0f}km": int(np.argmin(np.abs(h150 - hkm * 1000.0))) for hkm in args.heights}
    ktide = int(np.argmin(np.abs(h150 - args.tide_height * 1000.0)))
    print(f"[i] spot levels: {kspot}, tide level k={ktide} ({h150[ktide]/1000:.1f} km)")

    for V in args.versions:
        print(f"\n########## v{V} ##########")

        outdir = args.out_tpl.format(V=V)
        os.makedirs(outdir, exist_ok=True)
        cfg = {}
        rj = args.results_tpl.format(V=V)
        if os.path.exists(rj):
            cfg = json.load(open(rj)).get("config", {})
        ck_path = None; ck = None
        for cand in [args.ckpt_tpl.format(V=V),
                     args.ckpt_tpl.format(V=V).replace("full3d.pt", "final.pt")]:
            if os.path.exists(cand):
                ck_path = cand; break
        if ck_path is None:
            print(f"[skip] v{V}: チェックポイントが見つかりません"); continue
        ck = torch.load(ck_path, map_location="cpu")
        if not cfg and "args" in ck:
            cfg = ck["args"]
        levels = int(cfg.get("levels", 48)); stride = int(cfg.get("stride", 1))
        lev_idx = np.unique(np.round(np.linspace(0, LEV - 1, levels))).astype(int)
        lat_idx = np.arange(0, LAT, stride); lon_idx = np.arange(0, LON, stride)
        L, nlat, nlon = len(lev_idx), len(lat_idx), len(lon_idx)
        use_terrain = bool(cfg.get("use_terrain", V >= 10))
        in_ch = 7 if use_terrain else 6
        model = Full3DStepper(in_ch=in_ch, ch=tuple(cfg.get("ch", [24, 48, 32])))
        # Exponential moving average of the weights; M9 is evaluated with these weights.
        if "ema" in ck and float(cfg.get("ema_decay", 0.0) or 0.0) > 0 and ck["ema"] is not None:
            model.load_state_dict(ck["ema"]); print(f"[i] 重み: EMA ({ck_path})")
        elif "model" in ck:
            model.load_state_dict(ck["model"]); print(f"[i] 重み: final.pt model ({ck_path})")
        else:
            model.load_state_dict(ck); print(f"[i] 重み: {ck_path}")
        model.eval()

        T_tr = min(args.t_train, H)
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
        t0 = time.time()
        def sub(h, arr): return np.ascontiguousarray(arr[h][lev_idx][:, lat_idx][:, :, lon_idx])
        mU, sU = chunk_stats(lambda h: sub(h, gu), L); mV, sV = chunk_stats(lambda h: sub(h, gv), L)
        mW, sW = chunk_stats(lambda h: sub(h, gw), L); mT, sT = chunk_stats(lambda h: sub(h, gt), L)

        level_norm = ("level_weight_scale_km" in cfg) if cfg else (V >= 8)
        if not level_norm:
            def to_global(m, s):
                mg = float(m.mean()); vg = float((s.astype(np.float64) ** 2 + m.astype(np.float64) ** 2).mean() - mg * mg)
                sg = math.sqrt(max(vg, 0.0)) + 1e-8
                return np.full_like(m, mg), np.full_like(s, sg)
            mU, sU = to_global(mU, sU); mV, sV = to_global(mV, sV)
            mW, sW = to_global(mW, sW); mT, sT = to_global(mT, sT)
        print(f"[i] 正規化: {'層別' if level_norm else '全層共通（スカラー）'}")
        PSin = lambda h: np.log(np.clip(np.ascontiguousarray(gp[h][lat_idx][:, lon_idx]), 1.0, None))
        mPs_n = np.zeros(1); sPs_n = np.zeros(1); n = 0
        for s0 in range(0, T_tr, 48):
            e = min(s0 + 48, T_tr)
            a = np.stack([PSin(h) for h in range(s0, e)]).astype(np.float64)
            mPs_n += a.mean() * (e - s0); sPs_n += (a * a).mean() * (e - s0); n += e - s0
        mP = float(np.mean(mPs_n / n)); sP = float(math.sqrt(max(float(np.mean(sPs_n / n) - mP * mP), 0.0)) + 1e-8)
        print(f"[i] 統計量計算 {time.time()-t0:.1f}s")

        Z = np.asarray(gz[:, lev_idx, :, :].mean(axis=(0, 2, 3)), dtype=np.float64)
        zs2 = np.ascontiguousarray(zs_full[lat_idx][:, lon_idx])
        zs_mean, zs_std = float(zs2.mean()), float(zs2.std() + 1e-9)
        zsfc_t = torch.tensor(((zs2 - zs_mean) / zs_std).astype(np.float32))[None, None, None]
        zfeat = (Z - Z.mean()) / (Z.std() + 1e-9)
        zfeat_t = torch.tensor(zfeat.astype(np.float32))
        z_t = torch.tensor(Z.astype(np.float32))
        mU_t = torch.tensor(mU).reshape(1, 1, L, 1, 1); sU_t = torch.tensor(sU).reshape(1, 1, L, 1, 1)
        mV_t = torch.tensor(mV).reshape(1, 1, L, 1, 1); sV_t = torch.tensor(sV).reshape(1, 1, L, 1, 1)
        mW_t = torch.tensor(mW).reshape(1, 1, L, 1, 1); sW_t = torch.tensor(sW).reshape(1, 1, L, 1, 1)
        mT_t = torch.tensor(mT).reshape(1, 1, L, 1, 1); sT_t = torch.tensor(sT).reshape(1, 1, L, 1, 1)
        phi = -np.radians(180 * ((lat_idx.astype(np.float64) + 0.5 - 32) / 64))
        dtheta = float(2 * np.pi / LON) * stride
        dphi = float(phi[1] - phi[0])
        cos_phi = torch.tensor(np.cos(phi).astype(np.float32))
        sin_phi = torch.tensor(np.sin(phi).astype(np.float32))
        tan_phi = torch.tensor(np.tan(phi).astype(np.float32))
        fric_mask = torch.zeros(1, 1, L, 1, 1); fric_mask[0, 0, 0] = 1.0
        w_cos = np.cos(phi).astype(np.float64)
        T_MIN_N = ((T_MIN_K - mT_t) / sT_t); T_MAX_N = ((T_MAX_K - mT_t) / sT_t)
        P_MIN_N = (math.log(PS_MIN_HPA) - mP) / sP; P_MAX_N = (math.log(PS_MAX_HPA) - mP) / sP
        clip_inc = float(cfg.get("clip_inc", 1.0))
        # Remove the global mean of the surface-pressure increment at every step.
        mass_fix = bool(cfg.get("mass_fix", False))
        hypervisc = float(cfg.get("hypervisc", 0.05) or 0.0)

        inf_hv = bool(cfg.get("inference_hypervisc", True if "inference_hypervisc" not in cfg else False)) and hypervisc > 0
        print(f"[i] 推論時平滑化: {'あり' if inf_hv else 'なし'} (hypervisc={hypervisc}), mass_fix={mass_fix}")
        fric_coef = float(cfg.get("friction_coef", 2e-4))
        calibrate = bool(cfg.get("calibrate", False))
        cal_hours = int(cfg.get("cal_hours", args.cal_hours))

        def build_input(u, v, w, t, p):
            ps = p[:, :, None].expand(1, 1, L, nlat, nlon)
            zf = zfeat_t[None, None, :, None, None].expand(1, 1, L, nlat, nlon)
            chans = [u, v, w, t, ps, zf]
            if use_terrain:
                chans.append(zsfc_t.expand(1, 1, L, nlat, nlon))
            return torch.cat(chans, dim=1)

        def fields(u_n, v_n, w_n, t_n, p_n):
            u = u_n * sU_t + mU_t; v = v_n * sV_t + mV_t; w = w_n * sW_t + mW_t
            t = (t_n * sT_t + mT_t).clamp(min=T_MIN_K, max=T_MAX_K)
            lnps = (p_n * sP + mP).clamp(min=math.log(PS_MIN_HPA), max=math.log(PS_MAX_HPA))
            ps = torch.exp(lnps) * 100.0
            pk = hypsometric_p(ps, t, z_t)
            rho = pk / (R_D * t.clamp(min=150.0))
            r = R_EARTH + z_t[None, None, :, None, None]
            inv_r = 1.0 / r
            cos = cos_phi[None, None, None, :, None]
            inv_rc = inv_r / cos.clamp(min=COS_POLE_MIN)
            tan = tan_phi[None, None, None, :, None]
            fcor = 2.0 * OMEGA * sin_phi[None, None, None, :, None]
            return u, v, w, t, pk, rho, inv_r, inv_rc, tan, fcor

        def residual(du, dv, dw, dT, u_n, v_n, w_n, t_n, p_n):
            u, v, w, t, pk, rho, inv_r, inv_rc, tan, fcor = fields(u_n, v_n, w_n, t_n, p_n)
            du_dt, dv_dt, dT_dt = du / HOUR, dv / HOUR, dT / HOUR
            du_theta = grad_lon(u, dtheta); du_phi = grad_lat(u, dphi); du_z = grad_z(u, z_t)
            dv_theta = grad_lon(v, dtheta); dv_phi = grad_lat(v, dphi); dv_z = grad_z(v, z_t)
            dT_theta = grad_lon(t, dtheta); dT_phi = grad_lat(t, dphi); dT_z = grad_z(t, z_t)
            dp_theta = grad_lon(pk, dtheta); dp_phi = grad_lat(pk, dphi)
            fric = fric_coef * fric_mask
            fu = (du_dt + u * inv_rc * du_theta + v * inv_r * du_phi + w * du_z
                  - u * v * inv_r * tan - fcor * v + inv_rc * dp_theta / rho + fric * u)
            fv = (dv_dt + u * inv_rc * dv_theta + v * inv_r * dv_phi + w * dv_z
                  + u * u * inv_r * tan + fcor * u + inv_r * dp_phi / rho + fric * v)
            fT = dT_dt + u * inv_rc * dT_theta + v * inv_r * dT_phi + w * (dT_z + G / CP)
            return fu, fv, fT

        def load48(h):
            u = torch.tensor(np.ascontiguousarray(gu[h][lev_idx][:, lat_idx][:, :, lon_idx]))[None, None]
            v = torch.tensor(np.ascontiguousarray(gv[h][lev_idx][:, lat_idx][:, :, lon_idx]))[None, None]
            w = torch.tensor(np.ascontiguousarray(gw[h][lev_idx][:, lat_idx][:, :, lon_idx]))[None, None]
            t = torch.tensor(np.ascontiguousarray(gt[h][lev_idx][:, lat_idx][:, :, lon_idx]))[None, None]
            p = torch.tensor(np.ascontiguousarray(gp[h][lat_idx][:, lon_idx]))[None, None]
            return u, v, w, t, p

        # Normalization with statistics of the training period.
        def norm48(u, v, w, t, p):
            return ((u - mU_t) / sU_t, (v - mV_t) / sV_t, (w - mW_t) / sW_t,
                    (t - mT_t) / sT_t, (torch.log(p) - mP) / sP)

        def h48_t():
            return torch.tensor(Z.astype(np.float32))

        def smooth2d(f):
            left = torch.roll(f, 1, dims=-1); right = torch.roll(f, -1, dims=-1)
            up = torch.cat([f[..., 1:, :], f[..., -1:, :]], dim=-2)
            dn = torch.cat([f[..., :1, :], f[..., :-1, :]], dim=-2)
            return 0.25 * (left + right + up + dn)

        t0 = time.time()
        clim = {}
        for nm, arr in (("u", gu), ("v", gv), ("T", gt)):
            s = np.zeros((LEV, LAT, LON), dtype=np.float64)
            for s0 in range(0, T_tr, 48):
                e = min(s0 + 48, T_tr)
                s += np.asarray(arr[s0:e], dtype=np.float64).sum(axis=0)
            clim[nm] = (s / T_tr).astype(np.float32)
        s = np.zeros((LAT, LON), dtype=np.float64)
        for s0 in range(0, T_tr, 48):
            e = min(s0 + 48, T_tr)
            s += np.asarray(gp[s0:e], dtype=np.float64).sum(axis=0)
        clim["Ps"] = (s / T_tr).astype(np.float32)
        print(f"[i] 気候値計算 {time.time()-t0:.1f}s")
        w2 = np.repeat(w_cos[:, None], LON, axis=1); w2 = w2 / w2.sum()
        spot_k = {"sfc": 0, **kspot}
        prof_leads = sorted(set([l for l in (24, 72, horizon) if 1 <= l <= horizon]))

        rs = args.t_train
        u_n, v_n, w_n, t_n, p_n = [x.float() for x in norm48(*load48(rs))]
        res_pred = {"u": np.zeros(L), "v": np.zeros(L), "T": np.zeros(L)}; nres = 0
        spot = {sk: {v_: {} for v_ in ("u", "v", "T")} for sk in spot_k}
        spot_ps = {}
        profiles = {v_: {} for v_ in ("u", "v", "T")}
        cH_keep = int(cfg.get("cal_hours", args.cal_hours))
        surf_pred = {"u": [], "v": [], "T": []}
        tide_pred = np.zeros((horizon, nlat)); tide_truth = np.zeros((horizon, nlat))
        h48_np = Z
        t0 = time.time()
        with torch.no_grad():
            for step in range(horizon):
                x = build_input(u_n, v_n, w_n, t_n, p_n)
                d3d, d2d = model(x)
                if inf_hv:
                    d3d = d3d - hypervisc * (d3d - smooth2d(d3d))
                    d2d = d2d - hypervisc * (d2d - smooth2d(d2d))
                if clip_inc > 0:
                    d3d = torch.clamp(d3d, -clip_inc, clip_inc); d2d = torch.clamp(d2d, -clip_inc, clip_inc)
                if mass_fix:
                    d2d = d2d - d2d.mean()

                fu, fv, fT = residual(d3d[:, 0:1] * sU_t, d3d[:, 1:2] * sV_t, d3d[:, 2:3] * sW_t,
                                      d3d[:, 3:4] * sT_t, u_n, v_n, w_n, t_n, p_n)
                res_pred["u"] += (fu[0, 0] ** 2).numpy().mean(axis=(1, 2))
                res_pred["v"] += (fv[0, 0] ** 2).numpy().mean(axis=(1, 2))
                res_pred["T"] += (fT[0, 0] ** 2).numpy().mean(axis=(1, 2))
                nres += 1
                u_n = u_n + d3d[:, 0:1]; v_n = v_n + d3d[:, 1:2]; w_n = w_n + d3d[:, 2:3]
                t_n = torch.clamp(t_n + d3d[:, 3:4], min=T_MIN_N, max=T_MAX_N)
                p_n = torch.clamp(p_n + d2d, min=P_MIN_N, max=P_MAX_N)

                u48 = (u_n[0, 0].numpy() * sU[:, None, None] + mU[:, None, None])
                v48 = (v_n[0, 0].numpy() * sV[:, None, None] + mV[:, None, None])
                T48 = (t_n[0, 0].numpy() * sT[:, None, None] + mT[:, None, None])
                Ps = np.exp(p_n[0, 0].numpy() * sP + mP)
                surf_pred["u"].append(u48[0].copy()); surf_pred["v"].append(v48[0].copy())
                surf_pred["T"].append(T48[0].copy())
                tu = np.asarray(gu[rs + step + 1], dtype=np.float32)
                tv = np.asarray(gv[rs + step + 1], dtype=np.float32)
                tt = np.asarray(gt[rs + step + 1], dtype=np.float32)
                tp = np.asarray(gp[rs + step + 1], dtype=np.float32)
                lead = step + 1
                for nm, p48, t150 in (("u", u48, tu), ("v", v48, tv), ("T", T48, tt)):
                    p150 = interp_levels(p48, h48_np, h150)
                    if nm == "T":
                        tide_pred[step] = p150[ktide].mean(axis=-1)
                        tide_truth[step] = t150[ktide].mean(axis=-1)
                    if lead in prof_leads:
                        profiles[nm][lead] = level_metrics(p150, t150, w2, clim[nm])
                    ks = list(spot_k.values())
                    lm = level_metrics(p150[ks], t150[ks], w2, clim[nm][ks])
                    for j, sk in enumerate(spot_k):
                        spot[sk][nm][lead] = pick(lm, j)
                lm = level_metrics(Ps[None], tp[None], w2, clim["Ps"][None])
                spot_ps[lead] = pick(lm, 0)
        roll_sec = time.time() - t0
        print(f"[i] rollout {roll_sec:.1f}s（{roll_sec/horizon:.2f} s/step）")

        res_truth = {"u": np.zeros(L), "v": np.zeros(L), "T": np.zeros(L)}; nt = 0
        with torch.no_grad():
            for step in range(horizon - 1):
                u0, v0, w0, t0_, p0 = load48(rs + step)
                u1, v1, w1, t1, p1 = load48(rs + step + 1)
                du = u1 - u0; dv = v1 - v0; dw = w1 - w0; dT = t1 - t0_
                fu, fv, fT = residual(du, dv, dw, dT, *norm48(u0, v0, w0, t0_, p0))
                res_truth["u"] += (fu[0, 0] ** 2).numpy().mean(axis=(1, 2))
                res_truth["v"] += (fv[0, 0] ** 2).numpy().mean(axis=(1, 2))
                res_truth["T"] += (fT[0, 0] ** 2).numpy().mean(axis=(1, 2))
                nt += 1
        for d_ in (res_pred, res_truth):
            for k_ in d_: d_[k_] = np.sqrt(d_[k_] / max(nres if d_ is res_pred else nt, 1)).tolist()

        base = {b: {sk: {v_: {} for v_ in ("u", "v", "T")} for sk in spot_k} for b in ("persistence", "climatology")}
        base_ps = {"persistence": {}, "climatology": {}}
        ks = list(spot_k.values())
        init150 = {nm: np.asarray(a[rs], dtype=np.float32) for nm, a in (("u", gu), ("v", gv), ("T", gt))}
        init_ps = np.asarray(gp[rs], dtype=np.float32)
        for step in range(horizon):
            lead = step + 1
            for nm, a in (("u", gu), ("v", gv), ("T", gt)):
                t150 = np.asarray(a[rs + lead], dtype=np.float32)
                for bname, pf in (("persistence", init150[nm]), ("climatology", clim[nm])):
                    lm = level_metrics(pf[ks], t150[ks], w2, clim[nm][ks])
                    for j, sk in enumerate(spot_k):
                        base[bname][sk][nm][lead] = pick(lm, j)
            tps = np.asarray(gp[rs + lead], dtype=np.float32)
            for bname, pf in (("persistence", init_ps), ("climatology", clim["Ps"])):
                base_ps[bname][lead] = pick(level_metrics(pf[None], tps[None], w2, clim["Ps"][None]), 0)

        tide = {"height_km": float(h150[ktide] / 1000.0)}
        for nm, arr in (("truth", tide_truth), ("pred", tide_pred)):
            di_amp, di_tmax, sd_amp, sd_tmax = [], [], [], []
            for j in range(nlat):
                a1, m1 = harmonic_fit(arr[:, j], 24.0); b1, n1 = harmonic_fit(arr[:, j], 12.0)
                di_amp.append(a1); di_tmax.append(m1); sd_amp.append(b1); sd_tmax.append(n1)
            tide[nm] = {"diurnal_amp": di_amp, "diurnal_tmax": di_tmax,
                        "semidiurnal_amp": sd_amp, "semidiurnal_tmax": sd_tmax}

        cal = {}
        cH = min(cal_hours, horizon)
        if cH > 0:
            for nm in ("u", "v", "T"):
                src = (gu, gv, gt)[("u", "v", "T").index(nm)]
                tarr = np.stack([np.asarray(src[rs + h + 1][0][lat_idx][:, lon_idx], dtype=np.float32) for h in range(cH)])
                parr = np.stack(surf_pred[nm][:cH]).astype(np.float64)
                mp = float(parr.mean()); tp = float(tarr.mean())
                sp = float(parr.std() + 1e-12); st = float(tarr.std() + 1e-12)
                cal[nm] = dict(factor=st / sp, truth_mean=tp, pred_mean=mp, truth_std=st, pred_std=sp)
            print("[i] cal(検証先頭): " + ", ".join(f"{k} x{v['factor']:.3f}" for k, v in cal.items()))

        cal_train = None
        if args.cal_train and T_tr - args.cal_hours > 0:
            rs2 = T_tr - args.cal_hours
            u_n, v_n, w_n, t_n, p_n = [x.float() for x in norm48(*load48(rs2))]
            sp_series = {"u": [], "v": [], "T": []}
            with torch.no_grad():
                for step in range(args.cal_hours):
                    d3d, d2d = model(build_input(u_n, v_n, w_n, t_n, p_n))
                    if inf_hv:
                        d3d = d3d - hypervisc * (d3d - smooth2d(d3d))
                        d2d = d2d - hypervisc * (d2d - smooth2d(d2d))
                    if clip_inc > 0:
                        d3d = torch.clamp(d3d, -clip_inc, clip_inc); d2d = torch.clamp(d2d, -clip_inc, clip_inc)
                    if mass_fix:
                        d2d = d2d - d2d.mean()
                    u_n = u_n + d3d[:, 0:1]; v_n = v_n + d3d[:, 1:2]; w_n = w_n + d3d[:, 2:3]
                    t_n = torch.clamp(t_n + d3d[:, 3:4], min=T_MIN_N, max=T_MAX_N)
                    p_n = torch.clamp(p_n + d2d, min=P_MIN_N, max=P_MAX_N)
                    sp_series["u"].append(u_n[0, 0, 0].numpy() * sU[0] + mU[0])
                    sp_series["v"].append(v_n[0, 0, 0].numpy() * sV[0] + mV[0])
                    sp_series["T"].append(t_n[0, 0, 0].numpy() * sT[0] + mT[0])
            cal_train = {}
            for nm in ("u", "v", "T"):
                src = (gu, gv, gt)[("u", "v", "T").index(nm)]
                truth24 = np.stack([np.asarray(src[rs2 + s + 1][0][lat_idx][:, lon_idx], dtype=np.float32) for s in range(args.cal_hours)])
                parr = np.stack(sp_series[nm]).astype(np.float64)
                cal_train[nm] = dict(factor=float(truth24.std() / (parr.std() + 1e-12)),
                                     truth_std=float(truth24.std()), pred_std=float(parr.std()),
                                     truth_mean=float(truth24.mean()), pred_mean=float(parr.mean()))
            print("[i] cal_train: " + ", ".join(f"{k} x{v['factor']:.3f}" for k, v in cal_train.items()))

        metrics_cal = {}
        for tag, cd in (("val", cal), ("train", cal_train)):
            if not cd: continue
            metrics_cal[tag] = {nm: {} for nm in ("u", "v", "T")}
            for nm in ("u", "v", "T"):
                src = (gu, gv, gt)[("u", "v", "T").index(nm)]
                f_, tp_, mp_ = cd[nm]["factor"], cd[nm]["truth_mean"], cd[nm]["pred_mean"]
                for step in range(horizon):
                    pc_ = tp_ + (surf_pred[nm][step].astype(np.float64) - mp_) * f_
                    tr_ = np.asarray(src[rs + step + 1][0][lat_idx][:, lon_idx], dtype=np.float64)
                    lm = level_metrics(pc_[None], tr_[None], w2[lat_idx][:, lon_idx] / w2[lat_idx][:, lon_idx].sum(),
                                       clim[nm][0][lat_idx][:, lon_idx][None])
                    metrics_cal[tag][nm][step + 1] = pick(lm, 0)

        result = dict(version=V, config_used={k: cfg.get(k) for k in
                      ("levels", "stride", "ch", "mass_fix", "calibrate", "cal_hours", "clip_inc",
                       "inference_hypervisc", "ema_decay", "use_terrain", "seed", "tag") if k in cfg},
                      data=dict(H=H, LEV=LEV, t_train=args.t_train, horizon=horizon),
                      heights_km={k: float(h150[i] / 1000.0) for k, i in kspot.items()},
                      tide_height_km=float(h150[ktide] / 1000.0),
                      timing=dict(rollout_sec=round(roll_sec, 1), per_step_sec=round(roll_sec / horizon, 3)),
                      spot_levels={sk: dict(k=int(k), height_km=float(h150[k] / 1000.0)) for sk, k in spot_k.items()},
                      metrics_spot=spot, metrics_ps=spot_ps,
                      profiles=profiles, profile_heights_km=(h150 / 1000.0).round(2).tolist(),
                      baselines=base, baselines_ps=base_ps, cal=cal, metrics_cal=metrics_cal,
                      norm_mode=("level" if level_norm else "global"), inference_smoothing=inf_hv,
                      residuals=dict(pred=res_pred, truth=res_truth, n_steps_pred=nres, n_steps_truth=nt,
                                     level_heights_km=(Z / 1000.0).round(2).tolist()),
                      tide=tide, cal_train=cal_train)
        with open(os.path.join(outdir, "results_extra.json"), "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"[i] 保存: {outdir}/results_extra.json")

        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            zk = h150 / 1000.0
            for nm, unit in (("u", "m/s"), ("T", "K")):
                fig, ax = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
                for ld in prof_leads:
                    pr = profiles[nm][ld]
                    ax[0].plot(pr["rmse"], zk, label=f"t+{ld}h"); ax[1].plot(pr["bias"], zk, label=f"t+{ld}h")
                    ax[2].plot(pr["acc_gm"], zk, label=f"t+{ld}h")
                for a_, lab in zip(ax, (f"RMSE ({unit})", f"bias ({unit})", "ACC")):
                    a_.set_xlabel(lab); a_.grid(alpha=.3); a_.legend()
                ax[0].set_ylabel("height [km]")
                fig.suptitle(f"v{V} {nm}: per-level metrics (150 levels)")
                fig.tight_layout(); fig.savefig(os.path.join(outdir, f"profile_{nm}.png"), dpi=120); plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 4))
            latd = np.degrees(phi)
            ax.plot(latd, tide["truth"]["diurnal_amp"], label="truth")
            ax.plot(latd, tide["pred"]["diurnal_amp"], label="pred")
            ax.set_xlabel("latitude [deg]"); ax.set_ylabel("diurnal T amplitude [K]")
            ax.set_title(f"v{V} @ {tide['height_km']:.0f} km"); ax.legend(); ax.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "tide_diurnal.png"), dpi=120); plt.close(fig)

    print("\n[done]")

if __name__ == "__main__":
    main()
