# PINN all-atmosphere surrogate: code and trained weights

Code and trained weights for the manuscript on a physics-informed neural network
(PINN) surrogate of an all-atmosphere GCM (T42, 150 levels). The package
reproduces the training and the evaluation of the ten configurations (M0-M9)
reported in the paper.

## 1. Contents

```
code/
  train_v15.py      training, implementation v15   -> M5, M6, M7
  train_v17.py      training, implementation v17   -> M8, M9
  evaluate.py       evaluation of every metric in the paper
  eval_extra.py     model definition shared by evaluate.py
  make_table4.py    Table 4 (lambda_phys control experiment)
  plot_profiles.py  Figures 7 and 8 (vertical profiles, bias decomposition)
weights/
  M0.pt ... M9_seed2.pt   12 checkpoints (Section 4)
```

Not included: the GCM source code (available from the model developers) and the
raw GCM output, which is available from the authors on reasonable request.

## 2. Requirements

* Python 3.11, PyTorch 2.2 (CPU only, no GPU), NumPy, Matplotlib for `plot_profiles.py`
* The paper used `pytorch/2.2.0.py311_intel-2023.2` on one node of 112 CPU threads;
  `--threads 112` reproduces the published timings

## 3. Input data layout

One directory holds the GCM fields as NumPy arrays:

| file | shape | content |
|---|---|---|
| `gu.npy`, `gv.npy`, `gw.npy`, `gt.npy` | `[nt, 150, nlat, nlon]` | zonal, meridional and vertical wind, temperature |
| `gp.npy` | `[nt, nlat, nlon]` | surface pressure |
| `multilayer_gz.npy` | `[nt, 150, nlat, nlon]` | geopotential height of each level |

`nt` is the number of hourly records (744 for the 31-day run of the paper),
`nlat = 64` and `nlon = 128` for T42. Row `j = 0` of the latitude axis is the
North Pole and the meridional derivative of the physics residual assumes this
ordering. The first 600 records (25 days) are used for training and the last 144
for verification, set by `--train_days 25`.

## 4. Checkpoints

| file | configuration | size | SHA-256 (first 16 hex) |
|---|---|---|---|
| `M0.pt` | M0 (v7) | 2.84 MB | `132b357546925888` |
| `M1.pt` | M1 (v8) | 2.84 MB | `91e9bd6ae37b0cc9` |
| `M2.pt` | M2 (v9) | 1.42 MB | `c0c6a923a38d3152` |
| `M3.pt` | M3 (v10) | 1.43 MB | `23eace92e1c3881d` |
| `M4.pt` | M4 (v11) | 1.43 MB | `385ea19f0fcf874e` |
| `M5.pt` | M5 (v15 s0) | 1.43 MB | `5ce5d7e0f850f488` |
| `M6.pt` | M6 (v15 s0) | 1.43 MB | `0d798adf17d5eb3e` |
| `M7.pt` | M7 (v15 s0) | 1.43 MB | `18b255a54c3e2ab2` |
| `M8.pt` | M8 (v17 raw s0) | 1.43 MB | `cf42dd9033515257` |
| `M9_seed0.pt` | M9 (v17 s0) | 2.86 MB | `ab7da7a76157e575` |
| `M9_seed1.pt` | M9 (v17 s1) | 2.86 MB | `db1ae7737576bc2d` |
| `M9_seed2.pt` | M9 (v17 s2) | 2.86 MB | `f066003531c91eb9` |

Each file is a dictionary with the keys `config_id`, `note`, `args` (hyperparameters
of the run), `model` (trained weights) and `ema` (moving average, M0, M1 and M9
only). M9 must be evaluated with the `ema` weights; `evaluate.py` selects them
automatically whenever the entry is present. `M8.pt` is the run labelled `M7` in
the internal working directory of the project, as noted in `config_id`.

## 5. Quick start

### 5.1 Evaluate one configuration

```bash
DATA=/path/to/npy
mkdir -p runs/M9_seed0
cp weights/M9_seed0.pt runs/M9_seed0/final.pt
python code/evaluate.py --dir $DATA --run_dirs runs/M9_seed0 --out out/M9_seed0 --threads 112
```

`--run_dirs` takes one directory per configuration holding `final.pt` (or
`full3d.pt` with `--ckpt_prefer full3d`). The output `results.json` contains, for
each requested height, RMSE, bias, centered RMSE, ACC against the climatological
anomaly, pattern correlation, the persistence and climatology baselines, the
vertical profile, the mass-drift series and the Fourier decomposition of
atmospheric tides.

```bash
python code/plot_profiles.py --runs runs/M5 runs/M6 runs/M7 --out figs/
```

### 5.2 Train one configuration

The settings of every released weight are stored in the `args` entry of the
checkpoint and can be replayed as they are.

```bash
python code/train_v15.py --dir $DATA \
    --Ps_weight 1.8 \
    --T_weight 1.3 \
    --bc_lower kinematic \
    --bc_upper flat \
    --bptt_steps 2 \
    --cal_hours 24 \
    --calibrate \
    --causal_eps 0.03 \
    --ch 48 96 64 32 \
    --clip_inc 1.0 \
    --ema_decay 0.0 \
    --epochs_per_stage 6 \
    --friction_coef 0.0002 \
    --hypervisc 0.05 \
    --no-inference_hypervisc \
    --lambda_phys 0.03 \
    --level_weight_floor 0.1 \
    --level_weight_scale_km 40.0 \
    --levels 48 \
    --lr 0.001 \
    --mass_fix \
    --noise 0.05 \
    --rollout_start None \
    --rollouts 1 6 12 24 \
    --seed 0 \
    --stride 1 \
    --top_boost 2.0 \
    --top_levels 5 \
    --train_days 25 \
    --u_weight 1.0 \
    --use_terrain \
    --v_weight 1.0 \
    --w_bc 0.3 \
    --w_mass 0.5 \
    --w_spec 0.05 \
    --w_spec_high 0.1 \
    --w_terr 0.1 \
    --w_weight 1.0 \
    --w_wind_mass 0.3 \
    --w_zonal 0.3 \
    --outdir out/M7
```

```bash
python code/train_v17.py --dir $DATA \
    --Ps_weight 1.8 \
    --T_weight 1.3 \
    --bc_lower kinematic \
    --bc_upper flat \
    --bptt_steps 2 \
    --cal_hours 24 \
    --calibrate \
    --causal_eps 0.03 \
    --ch 48 96 64 32 \
    --clip_inc 1.0 \
    --ema_decay 0.999 \
    --epochs_per_stage 6 \
    --finetune_massfix \
    --friction_coef 0.0002 \
    --hypervisc 0.05 \
    --inc_boost_max 50.0 \
    --inc_norm \
    --no-inference_hypervisc \
    --lambda_phys 0.03 \
    --level_weight_floor 0.1 \
    --level_weight_scale_km 40.0 \
    --levels 48 \
    --lr 0.001 \
    --mass_fix \
    --no-mass_fix_cosphi \
    --noise 0.05 \
    --rollout_start None \
    --rollouts 1 6 12 24 \
    --seed 0 \
    --stride 1 \
    --top_boost 2.0 \
    --top_levels 5 \
    --train_days 25 \
    --u_weight 1.0 \
    --use_terrain \
    --v_weight 1.0 \
    --w_bc 0.3 \
    --w_mass 0.5 \
    --w_spec 0.05 \
    --w_spec_high 0.1 \
    --w_terr 0.1 \
    --w_weight 1.0 \
    --w_wind_mass 0.3 \
    --w_zonal 0.3 \
    --outdir out/M9_seed0
```

One run takes 32-36 min on 112 threads; `--seed` selects the seed, and seeds 0,
1 and 2 give the three M9 runs of the paper.

### 5.3 Table 4

```bash
python code/make_table4.py \
    --group M2 out/lphys/M2_l0.03_s0 out/lphys/M2_l0.03_s1 out/lphys/M2_l0.03_s2 \
               out/lphys/M2_l0_s0 out/lphys/M2_l0_s1 out/lphys/M2_l0_s2 \
    --out table4.csv
```

The table is built from the pre-calibration series `surface_rmse_raw`; the
calibrated series `surface_rmse` is read only for reference, which is why both
appear in the output and why the caption of Table 4 states that the values are
uncorrected.

## 6. Configuration map

| paper | implementation | change relative to the previous configuration |
|---|---|---|
| M0 | v7 | reference: normalization shared by all layers, `lambda_phys = 0.02` |
| M1 | v8 | layer-wise normalization, height-decaying layer weights |
| M2 | v9 | variable-wise loss weights |
| M3 | v10 | terrain input, boundary-condition and spectral losses |
| M4 | v11 | mass constraint, mass fixer at inference, no smoothing at inference |
| M5 | v15 | layer-weight floor 0.1, terrain-gradient regularization, stronger boundary loss |
| M6 | v15 | global-mean wind-increment constraint, high-wavenumber spectral loss, upper-level boost |
| M7 | v15 | zonal-mean wind-increment constraint |
| M8 | v17 | M7 losses with the double normalization of Section 2.4.4 removed |
| M9 | v17 | increment normalization, weight EMA, mass fixer in the final curriculum stage |

Implementations v7-v11 are not part of this package; only their checkpoints are
archived. They differ from v15/v17 in the number of input channels (no terrain
input in v7-v9) and in the layer-weighting scheme. `evaluate.py` infers the number
of input channels from the weights, so all twelve checkpoints can be evaluated
with the same script.

## 7. Changes made for release

* Comments and docstrings of the working scripts were removed. A short English
  header and a few English one-line comments were added where the intent of the
  code is not self-evident (mass fixer, increment clipping, EMA, physics
  residual, tide decomposition).
* Command-line help strings that were not in English were dropped; the options
  are documented in Sections 5 and 6.
* The checkpoints keep only the weights and the hyperparameters. Optimizer
  states, absolute paths, user and project names, job tags and timings were
  removed. The weights were verified to be bit-identical before and after
  rewriting.
* Console messages of the training loop are left as they were in the archived
  runs so that the logs of the paper remain comparable.

## 8. Licence and citation

To be completed on submission (licence and DOI of this archive).
