#!/usr/bin/env python3
"""\
Training code of the multilayer PINN surrogate with solar forcing (M2-SZA).
Identical to train_v15.py except for three solar input channels (cos SZA and
the sine and cosine of the local mean solar time), enabled with --solar:

  * in_ch = (7 if use_terrain else 6) + (3 if solar else 0)
  * new arguments --solar, --time_origin and --data_start_hour
  * build_input() takes the time index of the input state ("hour") and appends
    the solar channels; every call passes it (s0 + k in training, rs + k in
    the rollout)

The solar geometry is computed by solar_geometry.py. --time_origin and
--data_start_hour must be the same in training and evaluation.
See README.md (Section 5.4) for usage.
"""


import argparse, contextlib, gc, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R_D, OMEGA, R_EARTH, HOUR = 287.0, 7.2921e-5, 6.371e6, 3600.0
G, CP = 9.80665, 1004.0
COS_POLE_MIN = 5e-2
T_MIN_K, T_MAX_K = 120.0, 1500.0
PS_MIN_HPA, PS_MAX_HPA = 300.0, 1200.0

USE_BF16 = False
CH_LAST = False

class ConvBlock3D(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv3d(cin, cout, k, padding=0)
    def forward(self, x):
        x = F.pad(x, (self.p, self.p, 0, 0, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.p, self.p, self.p, self.p), mode="replicate")
        if CH_LAST:
            x = x.to(memory_format=torch.channels_last_3d)
        return F.silu(self.conv(x))

class Full3DStepper(nn.Module):
    def __init__(self, in_ch=7, ch=(24, 48, 32)):
        super().__init__()
        layers, prev = [], in_ch
        for c in ch:
            layers.append(ConvBlock3D(prev, c)); prev = c
        self.enc = nn.Sequential(*layers)
        self.head3d = nn.Conv3d(prev, 4, 1)
        self.head2d = nn.Sequential(nn.Conv3d(prev, 16, 1), nn.SiLU())
        self.head2d_conv = nn.Conv2d(16, 1, 3, padding=0)
    def forward(self, x):
        if CH_LAST:
            x = x.to(memory_format=torch.channels_last_3d)
        f = self.enc(x)
        d3d = self.head3d(f)
        g = self.head2d(f).mean(dim=2)

        g = torch.nn.functional.pad(g, (1, 1, 0, 0), mode='circular')
        g = torch.nn.functional.pad(g, (0, 0, 1, 1), mode='replicate')
        d2d = self.head2d_conv(g)
        return d3d.contiguous(), d2d.contiguous()

def grad_lon(f, dtheta):
    # Autoregressive rollout over R steps; the loss is accumulated over all steps.
    return (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) / (2 * dtheta)

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

def smooth2d(f):

    left = torch.roll(f, 1, dims=-1)
    right = torch.roll(f, -1, dims=-1)
    up = torch.cat([f[..., 1:, :], f[..., -1:, :]], dim=-2)
    dn = torch.cat([f[..., :1, :], f[..., :-1, :]], dim=-2)
    return 0.25 * (left + right + up + dn)

def hypsometric_p(p_s, T, z):

    T_layer = 0.5 * (T[:, :, :-1] + T[:, :, 1:])
    T_layer = T_layer.clamp(min=T_MIN_K, max=T_MAX_K)
    dz = (z[1:] - z[:-1])[None, None, :, None, None]
    dlnp = -(G / R_D) * dz / T_layer

    cum = torch.cumsum(dlnp, dim=2).clamp(min=-80.0, max=0.0)
    lnp_s = torch.log(p_s + 1e-6)
    lnp = torch.cat([lnp_s[:, :, None], lnp_s[:, :, None] + cum], dim=2)
    return torch.exp(lnp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./npy")
    ap.add_argument("--train_days", type=int, default=2)
    ap.add_argument("--rollout_start", type=int, default=None,
                    help="rollout [h] train_days*24")
    ap.add_argument("--levels", type=int, default=24)
    ap.add_argument("--level_weight_scale_km", type=float, default=40.0,
                    help=" w(z)=exp(-z/scale)  scale[km]0")
    ap.add_argument("--u_weight", type=float, default=1.0, help="u ")
    ap.add_argument("--v_weight", type=float, default=1.0, help="v ")
    ap.add_argument("--w_weight", type=float, default=1.0, help="w ")
    ap.add_argument("--T_weight", type=float, default=1.3, help="T ")
    ap.add_argument("--Ps_weight", type=float, default=1.8, help="Ps ")
    ap.add_argument("--use_terrain", action=argparse.BooleanOptionalAction, default=True,
                    help=" (v10 1)")
    ap.add_argument("--bc_lower", type=str, default="kinematic",
                    choices=["kinematic", "flat", "none"],
                    help=": kinematic=w_s=u∂z_s+u∂z_s / flat=w=0 / none")
    ap.add_argument("--bc_upper", type=str, default="flat", choices=["flat", "none"],
                    help=": flat= w=0 / none")
    ap.add_argument("--w_bc", type=float, default=0.3, help=" (0=off)")
    ap.add_argument("--level_weight_floor", type=float, default=0.1,
                    help="0=")
    ap.add_argument("--w_terr", type=float, default=0.1,
                    help=" (0=off)")
    ap.add_argument("--w_wind_mass", type=float, default=0.3,
                    help=" (0=off)")
    ap.add_argument("--w_zonal", type=float, default=0.3,
                    help="u,v  (v14)")
    ap.add_argument("--w_spec_high", type=float, default=0.1,
                    help=" (0=off)")
    ap.add_argument("--top_boost", type=float, default=1.0,
                    help=" top_levels  (1=)")
    ap.add_argument("--top_levels", type=int, default=5,
                    help="top_boost ")
    ap.add_argument("--w_spec", type=float, default=0.05, help=" (0=off)")
    ap.add_argument("--w_mass", type=float, default=0.5, help="Ps (0=off)")
    # Remove the global mean of the surface-pressure increment at every step.
    ap.add_argument("--mass_fix", action=argparse.BooleanOptionalAction, default=True,
                    help=" Ps  mass fixer")
    ap.add_argument("--inference_hypervisc", action=argparse.BooleanOptionalAction, default=False,
                    help="v10 ")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                    help=" u,v,T ")
    ap.add_argument("--cal_hours", type=int, default=24, help="[h]")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ch", type=int, nargs="+", default=[24, 48, 32])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs_per_stage", type=int, default=3)
    ap.add_argument("--rollouts", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--lambda_phys", type=float, default=0.03)
    ap.add_argument("--causal_eps", type=float, default=0.03)
    ap.add_argument("--friction_coef", type=float, default=2e-4)
    ap.add_argument("--noise", type=float, default=0.05, help="(std)")
    # Clip the normalized state increment to keep the autoregressive rollout stable.
    ap.add_argument("--clip_inc", type=float, default=1.0, help="(, 0=off)")
    ap.add_argument("--hypervisc", type=float, default=0.05, help="(0~1, 0=off)")
    ap.add_argument("--bptt_steps", type=int, default=2, help="BPTT(1=detach)")
    # Exponential moving average of the weights; M9 is evaluated with these weights.
    ap.add_argument("--ema_decay", type=float, default=0.0, help="EMA(0=off)")
    # Solar forcing (M2-SZA): three input channels computed from the time of each input state.
    ap.add_argument("--solar", action=argparse.BooleanOptionalAction, default=False,
                    help="append cos SZA, sin LMT and cos LMT to the input")
    ap.add_argument("--time_origin", type=str, default="2020-01-01T00:00:00+00:00",
                    help="UTC time of data index 0")
    ap.add_argument("--data_start_hour", type=float, default=0.0,
                    help="hours added to the data index before the solar geometry is computed")
    ap.add_argument("--amp", type=str, default="fp32", choices=["fp32", "bf16"],
                    help="Xeon Max  bf16 (AMX)  conv ")
    ap.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=True,
                    help="oneDNN channels_last_3d (CPU conv )")
    ap.add_argument("--compile", action="store_true", help="torch.compile()")
    ap.add_argument("--resume", type=str, default=None, help="(v10)")
    ap.add_argument("--save_every", type=int, default=0, help="ckpt(0=)")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--outdir", type=str, default="./full3d_v15_out")
    ap.add_argument("--threads", type=int, default=0, help="OMP(0==CPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.levels, args.stride = 16, 4
        args.ch = [16, 32]
        args.epochs_per_stage, args.rollouts = 1, [1, 3]

    if args.amp == "bf16":
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                _ = torch.relu(torch.randn(4, 8))
        except Exception as e:
            print(f"[warn] bf16 autocast が使えないため fp32 にフォールバックします: {e}")
            args.amp = "fp32"

    if args.channels_last and not hasattr(torch, "channels_last_3d"):
        print("[warn] torch.channels_last_3d が存在しないため channels_last を無効化します")
        args.channels_last = False

    if args.compile and not hasattr(torch, "compile"):
        print("[warn] torch.compile が存在しないため compile を無効化します")
        args.compile = False

    global USE_BF16, CH_LAST
    USE_BF16 = (args.amp == "bf16")
    CH_LAST = args.channels_last
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    nthr = args.threads if args.threads > 0 else max(1, os.cpu_count() or 1)
    torch.set_num_threads(nthr)
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    print(f"[i] torch={torch.__version__}, threads={nthr}, amp={args.amp}, "
          f"channels_last={CH_LAST}, compile={args.compile}")

    gu = np.load(f"{args.dir}/gu.npy", mmap_mode="r")
    gv = np.load(f"{args.dir}/gv.npy", mmap_mode="r")
    gw = np.load(f"{args.dir}/gw.npy", mmap_mode="r")
    gt = np.load(f"{args.dir}/gt.npy", mmap_mode="r")
    gp = np.load(f"{args.dir}/gp.npy", mmap_mode="r")
    gz = np.load(f"{args.dir}/multilayer_gz.npy", mmap_mode="r")

    H, LEV, LAT, LON = gu.shape
    lev_idx = np.unique(np.round(np.linspace(0, LEV - 1, args.levels))).astype(int)
    lat_idx = np.arange(0, LAT, args.stride)
    lon_idx = np.arange(0, LON, args.stride)
    L, nlat, nlon = len(lev_idx), len(lat_idx), len(lon_idx)

    def sub(a):
        return np.ascontiguousarray(a[:, lev_idx][:, :, lat_idx][:, :, :, lon_idx])

    U = sub(gu); V = sub(gv); W = sub(gw); T = sub(gt)
    PS = np.ascontiguousarray(gp[:, lat_idx][:, :, lon_idx])
    Z = gz[:, lev_idx, :, :].mean(axis=(0, 2, 3))

    zs_full = np.asarray(gz[0, 0, :, :], dtype=np.float32)
    zs2 = np.ascontiguousarray(zs_full[lat_idx][:, lon_idx])
    zs_mean, zs_std = float(zs2.mean()), float(zs2.std() + 1e-9)
    zsfc_t = torch.tensor(((zs2 - zs_mean) / zs_std).astype(np.float32))
    zs_phys_t = torch.tensor(zs2)[None, None]
    print(f"[i] 地形チャネル: z_sfc = {zs2.min():.0f}〜{zs2.max():.0f} m (std={zs_std:.0f} m)")

    T_tr = args.train_days * 24

    def zs_level(a):
        m = a[:T_tr].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        s = a[:T_tr].std(axis=(0, 2, 3), keepdims=True).astype(np.float32) + 1e-8
        return ((a - m) / s).astype(np.float32), m, s

    Uz, mU, sU = zs_level(U); Vz, mV, sV = zs_level(V); Wz, mW, sW = zs_level(W); Tz, mT, sT = zs_level(T)

    PSin = np.log(np.clip(PS, 1.0, None))
    mP = float(PSin[:T_tr].mean()); sP = float(PSin[:T_tr].std()) + 1e-8
    Pz = ((PSin - mP) / sP).astype(np.float32)

    zfeat = (Z - Z.mean()) / (Z.std() + 1e-9)
    zfeat_t = torch.tensor(zfeat.astype(np.float32))
    z_t = torch.tensor(Z.astype(np.float32))

    if args.level_weight_scale_km > 0:
        wlev = np.exp(-(Z / 1000.0) / args.level_weight_scale_km).astype(np.float32)
        if args.level_weight_floor > 0:
            wlev = np.maximum(wlev, args.level_weight_floor * wlev.max())
        wlev = wlev / wlev.mean()
    if args.top_boost != 1.0 and args.top_levels > 0:
        wlev[-args.top_levels:] = wlev[-args.top_levels:] * args.top_boost
        wlev = wlev / wlev.mean()

    wlev_t = torch.tensor(wlev.astype(np.float32))[None, None, :, None, None]

    mU_t = torch.tensor(mU.astype(np.float32)).reshape(1, 1, L, 1, 1)
    sU_t = torch.tensor(sU.astype(np.float32)).reshape(1, 1, L, 1, 1)
    mV_t = torch.tensor(mV.astype(np.float32)).reshape(1, 1, L, 1, 1)
    sV_t = torch.tensor(sV.astype(np.float32)).reshape(1, 1, L, 1, 1)
    mW_t = torch.tensor(mW.astype(np.float32)).reshape(1, 1, L, 1, 1)
    sW_t = torch.tensor(sW.astype(np.float32)).reshape(1, 1, L, 1, 1)
    mT_t = torch.tensor(mT.astype(np.float32)).reshape(1, 1, L, 1, 1)
    sT_t = torch.tensor(sT.astype(np.float32)).reshape(1, 1, L, 1, 1)

    theta = 2 * np.pi * (lon_idx.astype(np.float32) + 0.5) / LON
    phi = -np.radians(180 * ((lat_idx.astype(np.float32) + 0.5 - 32) / 64))
    dtheta = float(2 * np.pi / LON) * args.stride

    dphi = float(phi[1] - phi[0])
    cos_phi = torch.tensor(np.cos(phi).astype(np.float32))
    sin_phi = torch.tensor(np.sin(phi).astype(np.float32))
    tan_phi = torch.tensor(np.tan(phi).astype(np.float32))
    fric_mask = torch.zeros(1, 1, L, 1, 1); fric_mask[0, 0, 0] = 1.0

    cos_min = torch.clamp(cos_phi, min=COS_POLE_MIN)
    dzs_th = grad_lon(zs_phys_t, dtheta) / (R_EARTH * cos_min)[None, None, :, None]
    dzs_ph = grad_lat(zs_phys_t, dphi) / R_EARTH
    mU0 = float(mU_t[0, 0, 0, 0, 0]); sU0 = float(sU_t[0, 0, 0, 0, 0])
    mV0 = float(mV_t[0, 0, 0, 0, 0]); sV0 = float(sV_t[0, 0, 0, 0, 0])
    mT0 = float(mT_t[0, 0, 0, 0, 0]); sT0 = float(sT_t[0, 0, 0, 0, 0])
    mW0 = float(mW_t[0, 0, 0, 0, 0]); sW0 = float(sW_t[0, 0, 0, 0, 0])
    mWtop = float(mW_t[0, 0, -1, 0, 0]); sWtop = float(sW_t[0, 0, -1, 0, 0])
    dth_bc = dzs_th[0, 0]; dph_bc = dzs_ph[0, 0]
    dzs_mag = torch.sqrt(dzs_th ** 2 + dzs_ph ** 2) * R_EARTH
    gnorm = (dzs_mag / (dzs_mag.max() + 1e-12)).clamp(0.0, 1.0)[0, 0]

    def bc_losses(u_n, v_n, w_n, dw_n=None, dwtop_n=None):

        zero = torch.zeros((), device=u_n.device)
        tot = zero
        if args.bc_lower != "none":
            u0 = u_n[:, :, 0] * sU0 + mU0
            v0 = v_n[:, :, 0] * sV0 + mV0
            w0 = w_n[:, :, 0] * sW0 + mW0
            if args.bc_lower == "kinematic":
                wkin = u0 * dth_bc + v0 * dph_bc
                rl = ((w0 - wkin) / sW0) ** 2
            else:
                rl = (w0 / sW0) ** 2
            tot = tot + rl.mean()
            if dw_n is not None:
                tot = tot + ((dw_n / sW0) ** 2).mean()
        if args.bc_upper == "flat":
            wtop = w_n[:, :, -1] * sWtop + mWtop
            tot = tot + ((wtop / sWtop) ** 2).mean()
            if dwtop_n is not None:
                tot = tot + ((dwtop_n / sWtop) ** 2).mean()
        return tot

    hann_t = torch.hann_window(nlat).float()

    def logspec(fx):
        h = hann_t.view([1] * (fx.dim() - 2) + [nlat, 1])
        # Normalization with statistics of the training period.
        Ff = torch.fft.rfft2(fx * h, norm="ortho")
        return torch.log(Ff.abs() ** 2 + 1e-12)

    def spec3(p_, t_):
        return ((logspec(p_) - logspec(t_)) ** 2 * wlev_t).mean()

    def spec2(p_, t_):
        return ((logspec(p_) - logspec(t_)) ** 2).mean()

    ky_ = torch.abs(torch.fft.fftfreq(nlat))[:, None]
    kx_ = torch.fft.rfftfreq(nlon)[None, :]
    kr_ = torch.sqrt(ky_ ** 2 + kx_ ** 2)
    mask_hi = (kr_ >= kr_.median()).float()
    def spec3_hi(p_, t_):
        e = ((logspec(p_) - logspec(t_)) ** 2 * wlev_t).mean(dim=(0, 1))
        return (e * mask_hi).sum() / (mask_hi.sum() * e.shape[0] + 1e-12)
    def spec2_hi(p_, t_):
        e = (logspec(p_) - logspec(t_)) ** 2
        return (e * mask_hi).sum() / (mask_hi.sum() * e.shape[0] + 1e-12)

    in_ch = (7 if args.use_terrain else 6) + (3 if args.solar else 0)
    model = Full3DStepper(in_ch=in_ch, ch=tuple(args.ch))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema_state = None
    if args.ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update_ema():
        d = args.ema_decay
        for k, v in model.state_dict().items():
            ema_state[k] = d * ema_state[k] + (1 - d) * v.detach()

    def smooth_incs(d3d, d2d, use_hv=True):
        if args.hypervisc > 0 and use_hv:
            d3d = d3d - args.hypervisc * (d3d - smooth2d(d3d))
            d2d = d2d - args.hypervisc * (d2d - smooth2d(d2d))
        if args.clip_inc > 0:
            d3d = torch.clamp(d3d, -args.clip_inc, args.clip_inc)
            d2d = torch.clamp(d2d, -args.clip_inc, args.clip_inc)
        return d3d, d2d

    def forward_model(x):
        if USE_BF16:
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                return model(x)
        return model(x)

    resume_si = resume_ep = resume_eg = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if args.ema_decay > 0 and ck.get('ema') is not None:
            ema_state = ck['ema']
        resume_si = ck.get('si', 0); resume_ep = ck.get('ep', 0); resume_eg = ck.get('ep_global', 0)
        print(f"[i] 再開: si={resume_si} ep={resume_ep} ep_global={resume_eg}")

    if args.compile:
        model = torch.compile(model)

    if args.solar:
        from solar_geometry import geometry as _sol_geom
        _slat = 90.0 - 180.0 * (np.arange(nlat) + 0.5) / nlat   # grid rows run from north to south
        _slon = 360.0 * (np.arange(nlon) + 0.5) / nlon
        _sol_cache = {}

        def solar_t(hour):
            """Solar channels [3, nlat, nlon] for the data index `hour` (cached)."""
            key = float(hour)
            g = _sol_cache.get(key)
            if g is None:
                g = torch.as_tensor(_sol_geom(args.data_start_hour + key, _slat, _slon,
                                              args.time_origin), dtype=torch.float32)
                _sol_cache[key] = g
            return g

    def build_input(u_n, v_n, w_n, t_n, p_n, hour=None):
        B = u_n.shape[0]
        ps = p_n[:, :, None].expand(B, 1, L, nlat, nlon)
        zf = zfeat_t[None, None, :, None, None].expand(B, 1, L, nlat, nlon)
        chans = [u_n, v_n, w_n, t_n, ps, zf]
        if args.use_terrain:
            chans.append(zsfc_t[None, None, None].expand(B, 1, L, nlat, nlon))
        if args.solar:
            if hour is None:
                raise ValueError("build_input needs the time index (hour) when --solar is set")
            chans.append(solar_t(hour)[None, :, None]
                         .expand(B, 3, L, nlat, nlon).to(dtype=u_n.dtype, device=u_n.device))
        return torch.cat(chans, dim=1)

    def fields(u_n, v_n, w_n, t_n, p_n):

        u = u_n * sU_t + mU_t
        v = v_n * sV_t + mV_t
        w = w_n * sW_t + mW_t
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
        fric = args.friction_coef * fric_mask
        fu = (du_dt + u * inv_rc * du_theta + v * inv_r * du_phi + w * du_z
              - u * v * inv_r * tan - fcor * v + inv_rc * dp_theta / rho + fric * u)
        fv = (dv_dt + u * inv_rc * dv_theta + v * inv_r * dv_phi + w * dv_z
              + u * u * inv_r * tan + fcor * u + inv_r * dp_phi / rho + fric * v)
        fT = dT_dt + u * inv_rc * dT_theta + v * inv_r * dT_phi + w * (dT_z + G / CP)
        return fu, fv, fT

    def load(h):
        u = torch.tensor(Uz[h][None, None]); v = torch.tensor(Vz[h][None, None])
        w = torch.tensor(Wz[h][None, None]); t = torch.tensor(Tz[h][None, None])
        p = torch.tensor(Pz[h][None, None])
        return u, v, w, t, p

    T_MIN_N = ((T_MIN_K - mT_t) / sT_t)
    T_MAX_N = ((T_MAX_K - mT_t) / sT_t)
    P_MIN_N = (math.log(PS_MIN_HPA) - mP) / sP
    P_MAX_N = (math.log(PS_MAX_HPA) - mP) / sP

    rollouts = args.rollouts
    total_epochs = args.epochs_per_stage * len(rollouts)
    ep_global, lr = resume_eg, args.lr * (0.7 ** resume_si)
    ema = [None] * 3
    ckpt_path = os.path.join(args.outdir, "ckpt.pt")

    # Residual of the primitive equations for the predicted one-step increment (physics loss).
    def phys_loss(fu, fv, fT, cw):
        fs = [fu, fv, fT]
        with torch.no_grad():
            for i, f_ in enumerate(fs):
                s = f_.std().item() + 1e-9
                ema[i] = s if ema[i] is None else 0.99 * ema[i] + 0.01 * s
        pole = (cos_phi > COS_POLE_MIN).float()[None, None, None, :, None]
        tot = sum((fs[i].abs() / ema[i] * pole * wlev_t).mean() for i in range(3))
        return tot / 3.0 * cw

    print(f"[i] {L}層×{nlat}×{nlon} / 地表{Z[0]/1000:.2f}km→圏界面{Z[-1]/1000:.0f}km, "
          f"λ_phys={args.lambda_phys}, terrain={args.use_terrain}, "
          f"bc=({args.bc_lower},{args.bc_upper},w={args.w_bc}), w_spec={args.w_spec}, "
          f"level_floor={args.level_weight_floor}, w_terr={args.w_terr}, "
          f"w_wind_mass={args.w_wind_mass}, w_zonal={args.w_zonal}, w_spec_high={args.w_spec_high}, "
          f"top_boost={args.top_boost}x{args.top_levels}lv")
    model.train()
    for si, R in enumerate(rollouts):
        if si < resume_si:
            continue
        for pg in opt.param_groups:
            pg["lr"] = lr
        lr *= 0.7
        starts = np.random.permutation(T_tr - R)
        for ep in range(args.epochs_per_stage):
            if si == resume_si and ep < resume_ep:
                continue
            eps = args.causal_eps * (1.0 - ep_global / max(total_epochs - 1, 1)) ** 2
            tot_d = tot_p = tot_b = tot_s = tot_m = tot_t = tot_wm = tot_sh = tot_zm = 0.0; n = 0
            bad_batches = 0
            for s0 in starts[np.random.permutation(len(starts))[: max(6, 20 - si * 5)]]:
                s0 = int(s0)
                u, v, w, t, p = load(s0)
                loss = 0.0
                for k in range(R):

                    if args.noise > 0:
                        u_i = u + args.noise * torch.randn_like(u)
                        v_i = v + args.noise * torch.randn_like(v)
                        w_i = w + args.noise * torch.randn_like(w)
                        t_i = t + args.noise * torch.randn_like(t)
                        p_i = p + args.noise * torch.randn_like(p)
                    else:
                        u_i, v_i, w_i, t_i, p_i = u, v, w, t, p
                    d3d, d2d = forward_model(build_input(u_i, v_i, w_i, t_i, p_i, s0 + k))
                    d3d, d2d = smooth_incs(d3d, d2d)
                    du = d3d[:, 0:1] * sU_t; dv = d3d[:, 1:2] * sV_t
                    dw = d3d[:, 2:3] * sW_t; dT = d3d[:, 3:4] * sT_t
                    u_nxt = u + d3d[:, 0:1]; v_nxt = v + d3d[:, 1:2]
                    w_nxt = w + d3d[:, 2:3]
                    t_nxt = torch.clamp(t + d3d[:, 3:4], min=T_MIN_N, max=T_MAX_N)
                    p_nxt = torch.clamp(p + d2d, min=P_MIN_N, max=P_MAX_N)
                    u_t, v_t, w_t, t_t, p_t = load(s0 + k + 1)

                    loss_u = (((u_nxt - u_t) ** 2) * wlev_t).mean()
                    loss_v = (((v_nxt - v_t) ** 2) * wlev_t).mean()
                    loss_w = (((w_nxt - w_t) ** 2) * wlev_t).mean()
                    loss_T = (((t_nxt - t_t) ** 2) * wlev_t).mean()
                    loss_Ps = F.mse_loss(p_nxt, p_t)
                    loss_d = (args.u_weight * loss_u
                              + args.v_weight * loss_v
                              + args.w_weight * loss_w
                              + args.T_weight * loss_T
                              + args.Ps_weight * loss_Ps)
                    loss = loss + loss_d
                    cw = math.exp(-eps * (T_tr - (s0 + k)))
                    if args.lambda_phys > 0:
                        fu, fv, fT = residual(du, dv, dw, dT, u, v, w, t, p)
                        lp = phys_loss(fu, fv, fT, cw)
                        loss = loss + args.lambda_phys * lp
                        tot_p += lp.item()
                    if args.w_bc > 0 and (args.bc_lower != "none" or args.bc_upper != "none"):
                        lb = bc_losses(u_nxt, v_nxt, w_nxt,
                                        dw_n=d3d[:, 2:3, 0], dwtop_n=d3d[:, 2:3, -1])
                        loss = loss + args.w_bc * lb * cw
                        tot_b += args.w_bc * lb.item()
                    if args.w_mass > 0:
                        lm = (p_nxt - p).mean() ** 2
                        loss = loss + args.w_mass * lm * cw
                        tot_m += args.w_mass * lm.item()
                    if args.w_terr > 0:
                        du0 = d3d[:, 0:1, 0] / sU0
                        dv0 = d3d[:, 1:2, 0] / sV0
                        lt = (gnorm * (du0 ** 2 + dv0 ** 2)).mean()
                        loss = loss + args.w_terr * lt * cw
                        tot_t += args.w_terr * lt.item()
                    if args.w_wind_mass > 0:
                        lwm = d3d[:, 0:1].mean() ** 2 + d3d[:, 1:2].mean() ** 2
                        loss = loss + args.w_wind_mass * lwm * cw
                        tot_wm += args.w_wind_mass * lwm.item()
                    if args.w_zonal > 0:

                        lzm = ((d3d[:, 0:1].mean(dim=-1)) ** 2).mean()\
                              + ((d3d[:, 1:2].mean(dim=-1)) ** 2).mean()
                        loss = loss + args.w_zonal * lzm * cw
                        tot_zm += args.w_zonal * lzm.item()
                    if args.w_spec_high > 0:
                        lsh = (args.u_weight * spec3_hi(u_nxt, u_t)
                               + args.v_weight * spec3_hi(v_nxt, v_t)
                               + args.T_weight * spec3_hi(t_nxt, t_t)
                               + args.Ps_weight * spec2_hi(p_nxt, p_t))
                        loss = loss + args.w_spec_high * lsh * cw
                        tot_sh += args.w_spec_high * lsh.item()
                    if args.w_spec > 0:
                        ls = (args.u_weight * spec3(u_nxt, u_t)
                              + args.v_weight * spec3(v_nxt, v_t)
                              + args.T_weight * spec3(t_nxt, t_t)
                              + args.Ps_weight * spec2(p_nxt, p_t))
                        loss = loss + args.w_spec * ls * cw
                        tot_s += args.w_spec * ls.item()
                    tot_d += loss_d.item(); n += 1
                    if not torch.isfinite(loss):
                        bad_batches += 1
                        break
                    if (k + 1) % args.bptt_steps == 0:
                        u, v, w, t, p = u_nxt.detach(), v_nxt.detach(), w_nxt.detach(), t_nxt.detach(), p_nxt.detach()
                    else:
                        u, v, w, t, p = u_nxt, v_nxt, w_nxt, t_nxt, p_nxt
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if args.ema_decay > 0:
                    update_ema()
            ep_global += 1
            if args.save_every > 0 and ep_global % args.save_every == 0:
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                            'ema': ema_state, 'si': si, 'ep': ep + 1,
                            'ep_global': ep_global, 'args': vars(args)}, ckpt_path)
                print(f"    [ckpt saved @ep_global={ep_global}]")
            print(f"  [R={R:>2d}h] ep {ep+1}/{args.epochs_per_stage} loss={tot_d/max(n,1):.4e} "
                  f"phys={tot_p/max(n,1):.3e} bc={tot_b/max(n,1):.3e} spec={tot_s/max(n,1):.3e} mass={tot_m/max(n,1):.3e} terr={tot_t/max(n,1):.3e} windm={tot_wm/max(n,1):.3e} hispec={tot_sh/max(n,1):.3e} zm={tot_zm/max(n,1):.3e} bad={bad_batches}")

    rs = args.rollout_start if args.rollout_start is not None else T_tr
    print(f"[i] 評価: {rs}h→{H}h rollout (学習 {T_tr}h; terrain={args.use_terrain} bc={args.bc_lower}/{args.bc_upper} "
          f"w_bc={args.w_bc} w_spec={args.w_spec} noise={args.noise} clip={args.clip_inc} "
          f"hvisc={args.hypervisc} bptt={args.bptt_steps} ema={args.ema_decay})")
    if args.ema_decay > 0:
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
    model.eval()
    u, v, w, t, p = load(rs)
    horizon = H - rs
    preds = []
    with torch.no_grad():
        for k in range(horizon):
            d3d, d2d = forward_model(build_input(u, v, w, t, p, rs + k))
            d3d, d2d = smooth_incs(d3d, d2d, use_hv=args.inference_hypervisc)
            if args.mass_fix:
                d2d = d2d - d2d.mean()
            u = u + d3d[:, 0:1]; v = v + d3d[:, 1:2]
            w = w + d3d[:, 2:3]
            t = torch.clamp(t + d3d[:, 3:4], min=T_MIN_N, max=T_MAX_N)
            p = torch.clamp(p + d2d, min=P_MIN_N, max=P_MAX_N)
            preds.append((u.numpy(), v.numpy(), t.numpy(), p.numpy()))
    if args.ema_decay > 0:
        model.load_state_dict(backup)

    pu = np.stack([x[0][0, 0, 0] * sU0 + mU0 for x in preds])
    pv = np.stack([x[1][0, 0, 0] * sV0 + mV0 for x in preds])
    pt = np.stack([x[2][0, 0, 0] * sT0 + mT0 for x in preds])
    pp = np.stack([np.exp(x[3][0, 0] * sP + mP) for x in preds])

    pu_raw, pv_raw, pt_raw = pu.copy(), pv.copy(), pt.copy()

    cal = None
    if args.calibrate and args.cal_hours > 0:
        cH = min(args.cal_hours, horizon)
        cal = {}
        for nm, arr, tarr in (("u", pu, U[rs:rs + cH, 0]), ("v", pv, V[rs:rs + cH, 0]),
                              ("T", pt, T[rs:rs + cH, 0])):
            mp = float(arr[:cH].mean()); tp = float(tarr[:cH].mean())
            sp = float(arr[:cH].std() + 1e-12); st = float(tarr[:cH].std() + 1e-12)
            arr[:] = tp + (arr - mp) * (st / sp)
            cal[nm] = dict(factor=st / sp, truth_mean=tp, pred_mean=mp)
        print(f"[i] 分散キャリブレーション ({cH}h): " + ", ".join(f"{k} x{v['factor']:.3f}" for k, v in cal.items()))
    print("\n===== 地表(k=0) rollout RMSE =====")
    nans, metrics = [], {}
    for h in [11, 23, 47, 71, 95, 119, horizon - 1]:
        if h >= horizon: continue
        ru = float(np.sqrt(np.mean((pu[h] - U[rs + h, 0]) ** 2)))
        rv = float(np.sqrt(np.mean((pv[h] - V[rs + h, 0]) ** 2)))
        rt = float(np.sqrt(np.mean((pt[h] - T[rs + h, 0]) ** 2)))
        rp = float(np.sqrt(np.mean((pp[h] - PS[rs + h]) ** 2)))
        print(f"  t+{h+1:>3d}h  u={ru:6.2f}  v={rv:6.2f}  T={rt:6.2f}  Ps={rp:6.2f}")
        metrics[f"t+{h+1}h"] = dict(u=ru, v=rv, T=rt, Ps=rp)
        nans.append(bool(np.isnan(ru) or np.isnan(rp)))
    print(f"[i] NaN発生: {any(nans)}")

    metrics_raw = {}
    for h in [11, 23, 47, 71, 95, 119, horizon - 1]:
        if h >= horizon: continue
        metrics_raw[f"t+{h+1}h"] = dict(
            u=float(np.sqrt(np.mean((pu_raw[h] - U[rs + h, 0]) ** 2))),
            v=float(np.sqrt(np.mean((pv_raw[h] - V[rs + h, 0]) ** 2))),
            T=float(np.sqrt(np.mean((pt_raw[h] - T[rs + h, 0]) ** 2))),
            Ps=float(np.sqrt(np.mean((pp[h] - PS[rs + h]) ** 2))))
    print("  [補正前] " + "  ".join(
        f"t+{k[2:]}: u={d['u']:.2f} T={d['T']:.2f} Ps={d['Ps']:.2f}" for k, d in metrics_raw.items()))

    def acc2d(p_, t_):
        p0 = p_ - p_.mean(); t0 = t_ - t_.mean()
        return float((p0 * t0).sum() / (np.sqrt((p0 * p0).sum() * (t0 * t0).sum()) + 1e-12))

    def spec_power(fx):
        h = np.hanning(fx.shape[-2]).astype(np.float32)[:, None]
        return np.abs(np.fft.rfft2(fx * h, norm="ortho")) ** 2

    def spec_ratio_bins(p_, t_, nb=8):
        P_, T_ = spec_power(p_), spec_power(t_)
        ky = np.fft.fftfreq(p_.shape[-2])[:, None] * p_.shape[-2]
        kx = np.fft.rfftfreq(p_.shape[-1]) * p_.shape[-1]
        kr = np.sqrt(ky ** 2 + kx ** 2)
        edges = np.logspace(0, np.log10(kr.max() + 1.0), nb + 1)
        return [float(P_[(kr >= edges[b]) & (kr < edges[b+1])].mean() /
                      (T_[(kr >= edges[b]) & (kr < edges[b+1])].mean() + 1e-12)) for b in range(nb)]

    diag = {"acc": {}, "spec_ratio": {}}
    print("\n===== 構造診断（地表: ACC / スペクトル比 低波数→高波数）=====")
    for h in [23, 47, 71, 95, 119, horizon - 1]:
        if h >= horizon: continue
        key = f"t+{h+1}h"
        accs = {"u": acc2d(pu[h], U[rs+h, 0]), "v": acc2d(pv[h], V[rs+h, 0]),
                "T": acc2d(pt[h], T[rs+h, 0]), "Ps": acc2d(pp[h], PS[rs+h])}
        ratios = {"u": spec_ratio_bins(pu[h], U[rs+h, 0]), "v": spec_ratio_bins(pv[h], V[rs+h, 0]),
                  "T": spec_ratio_bins(pt[h], T[rs+h, 0]), "Ps": spec_ratio_bins(pp[h], PS[rs+h])}
        diag["acc"][key] = accs; diag["spec_ratio"][key] = ratios
        print(f"  {key}: ACC u={accs['u']:.3f} v={accs['v']:.3f} T={accs['T']:.3f} Ps={accs['Ps']:.3f}")
    diag["acc_raw"] = {}
    for h in [23, 47, 71, 95, 119, horizon - 1]:
        if h >= horizon: continue
        diag["acc_raw"][f"t+{h+1}h"] = {"u": acc2d(pu_raw[h], U[rs+h, 0]), "v": acc2d(pv_raw[h], V[rs+h, 0]),
                                        "T": acc2d(pt_raw[h], T[rs+h, 0]), "Ps": acc2d(pp[h], PS[rs+h])}

    print(f"  spec_ratio u@t+24h = {[round(r,2) for r in diag['spec_ratio'].get('t+24h', {}).get('u', [])]}")

    torch.save(model.state_dict(), os.path.join(args.outdir, "full3d.pt"))
    torch.save({'model': model.state_dict(), 'ema': ema_state, 'args': vars(args)},
               os.path.join(args.outdir, "final.pt"))
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(dict(tag=args.tag, surface_rmse=metrics, surface_rmse_raw=metrics_raw,
                       diagnostics=diag,
                       calibration=cal, nan=bool(any(nans)),
                       elapsed_sec=round(time.time() - t0, 1),
                       config=vars(args)), f, indent=2)
    print(f"[i] 完了 ({time.time()-t0:.1f}s). results.json / final.pt を {args.outdir} に保存")

if __name__ == "__main__":
    main()
