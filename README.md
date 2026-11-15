# GCM_PINNs: code, trained weights, and evaluation results of a physics-informed neural network surrogate model for a general circulation model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24050683.svg)](https://doi.org/10.5281/zenodo.24050683)

Code, trained weights, and evaluation results for the manuscript "Development and
evaluation of a physics-informed neural network surrogate model for a general
circulation model" by K. Fukazawa and Y. Miyoshi (submitted to Earth, Planets and
Space). The surrogate approximates a GCM that contains the region from the ground
surface to the exobase (T42, 150 levels). The package reproduces the training and
the evaluation of the 18 configurations of the paper (three random seeds each),
the statistical baselines, and Tables 3-7 and Figures 2-8.

Version 2.0.0 replaces version 1.x completely (Section 8).

## 1. Contents

```
code/
  train.py              training of every configuration
  evaluate.py           evaluation of every metric in the paper; fields for the figures
  eval_extra.py         model definition used by evaluate.py
  solar_geometry.py     solar zenith angle and local mean solar time (solar input channels)
  baseline_diurnal.py   baselines that retain the diurnal cycle
  pack_weights.py       packing of the checkpoints for release
jobs/
  run_config.sh         command line of each configuration: bash jobs/run_config.sh M5 0
analysis/
  lib.py, logs.py       loaders of results/
  tabs.py               Tables 3-7
  figs.py, fig2.py      Figures 3-8 and Figure 2
  make_movie.py         movie of the GCM, the surrogate, and their difference
results/
  <configuration>_seed<seed>/results.json, results_extra3.json   evaluation of the 54 runs
  M5_seed0/fields.npz, M5-SZA_seed0/fields.npz                   fields of Figures 5 and 8
  baselines/diurnal.json                                         statistical baselines
  training_log.txt                                               loss terms of every epoch
weights/
  <configuration>_seed<seed>.pt                                  54 checkpoints, SHA256.md
```

Not included: the GCM source code (available from the model developers) and the
raw GCM output, which is available from the authors on reasonable request.

## 2. Requirements

* Python 3.11, PyTorch 2.2 (CPU only, no GPU), NumPy; Matplotlib for `analysis/`;
  ffmpeg for MP4 output of `make_movie.py` (a GIF is written without it)
* The paper used `pytorch/2.2.0.py311_intel-2023.2` on one node with 112 CPU threads

## 3. Input data layout

One directory holds the GCM fields as NumPy arrays:

| file | shape | content |
|---|---|---|
| `gu.npy`, `gv.npy`, `gt.npy` | `[nt, 150, nlat, nlon]` | zonal wind, meridional wind [m/s], temperature [K] |
| `gw.npy` | `[nt, 150, nlat, nlon]` | pressure vertical velocity omega [hPa/s] |
| `gp.npy` | `[nt, nlat, nlon]` | surface pressure [hPa] |
| `multilayer_gz.npy` | `[nt, 150, nlat, nlon]` | geopotential height of each level [m] |

`nt` is the number of hourly records (744 for the 31-day run of the paper),
`nlat = 64` and `nlon = 128` for T42. Row `j = 0` of the latitude axis is the
northernmost row. The first 600 records (25 days) are used for training
(`--train_days 25`) and the last 144 for validation.

`gw.npy` is the pressure vertical velocity, not a geometric vertical velocity.
`train.py` converts it to the geometric vertical velocity, `w = -omega R T / (p g)`,
in the physics residual and in the kinematic boundary condition (`--w_var omega`,
`--omega_to_pa 100`). In the GCM output, omega at the lowest level is zero at all
grid points, so the kinematic condition is imposed at the second lowest of the 48
levels (`--bc_level 1`). At start-up, `train.py` prints the correlation between the
two sides of the kinematic condition in the training data (0.87 for the data of the
paper), which serves as a check of the unit and of the level.

## 4. Configurations

M0-M5 are cumulative. Three seeds (0, 1, 2) were run for every configuration.

| name | content |
|---|---|
| M0 | 3D CNN, normalization common to all levels, uniform level and variable weights, physics loss |
| M1 | level-wise normalization, altitude-decaying level weight |
| M2 | variable-wise loss weights (T 1.3, Ps 1.8) |
| M3 | topography input, boundary-condition loss |
| M4 | mass constraint and mass fixer, lower bound of the level weight, topographic-gradient regularization, global-mean and zonal-mean wind increment constraints |
| M5 | increment normalization, mass fixer at the final training stage (final configuration) |
| M2+S ... M5+S | M2 ... M5 with the spectral loss(es) |
| M2-P0, M2-P1, M2-P10, M5-P0 | weight of the physics loss 0, 1, 10 (M2) and 0 (M5); the reference is 0.03 |
| M2-2E, M5-2E | 12 instead of 6 epochs per curriculum stage |
| M2-SZA, M5-SZA | three solar input channels: cos(SZA), sine and cosine of the local mean solar time |

`jobs/run_config.sh` holds the exact command line of each configuration.

## 5. Quick start

### 5.1 Evaluate a released checkpoint

```bash
DATA=/path/to/npy
mkdir -p runs/M5_seed0
cp weights/M5_seed0.pt runs/M5_seed0/final.pt
cd code
python evaluate.py --dir $DATA --run_dirs ../runs/M5_seed0 \
    --t_train 600 --horizon 143 --heights 0 100 120 150 200 300 \
    --tide_heights 100 120 130 150 --timing_repeats 5 --plot 0 --threads 112
```

The output `runs/M5_seed0/extra3/results_extra3.json` contains, for each height
and lead time, the RMSE, bias, centered RMSE and anomaly correlation of the
prediction, of persistence and of the time-mean climatology, the series of the
global-mean surface pressure with and without the mass fixer, the Fourier
decomposition of the tides, and the timing. The configuration is read from the
`args` entry of the checkpoint. `--dump_fields 1` also writes `fields.npz`
(prediction, truth, initial state, and climatology at the requested lead times).

### 5.2 Train a configuration

```bash
bash jobs/run_config.sh M5 0 /path/to/npy runs/M5_seed0     # 28-38 min on 112 threads
```

### 5.3 Baselines with the diurnal cycle

```bash
cd code
python baseline_diurnal.py --dir $DATA --t_train 600 --horizon 143 --out ../results/baselines
```

### 5.4 Tables and figures of the paper

```bash
cd analysis
python tabs.py      # Tables 3-7
python figs.py      # Figures 3-8 -> ../figs
python fig2.py      # Figure 2
```

The scripts read `results/`; set the environment variable `RESULTS` to use the
output of your own runs (one directory `<configuration>_seed<seed>` per run with
`results.json`, `results_extra3.json` and, for Figures 5 and 8, `fields.npz`).

### 5.5 Movie

```bash
cd code
python evaluate.py --dir $DATA --run_dirs ../runs/M5_seed0 --t_train 600 --horizon 143 \
    --clamp_diag 0 --timing_repeats 0 --drift_series 0 --plot 0 \
    --dump_fields 1 --dump_leads $(seq 1 143) --dump_heights 0 300
cd ../analysis
python make_movie.py --fields ../runs/M5_seed0/extra3/fields.npz --label M5 --out M5.mp4
```

## 6. Notes on the results files

* `results_extra3.json`: `native[<height>][<variable>][pred|persist|clim].series`
  holds `rmse`, `bias`, `crmse`, `acc_gm` and `acc_clim` for lead times 1-143 h.
  The ACC of the paper is `acc_clim`. The values are not area weighted.
* `drift_series` holds the sum over longitude of the cos-latitude-weighted surface
  pressure; divide by the number of longitudes (128) to obtain the global mean in hPa.
* `training_log.txt`: `loss` is the data loss, `phys` the physics loss before
  multiplication by its weight, and the other terms are weighted.

## 7. Checkpoints

Each file in `weights/` is a dictionary with the keys `config_id`, `args`
(hyperparameters of the run) and `model` (trained weights at the end of training).
Sizes and SHA-256 digests are listed in `weights/SHA256.md`.

## 8. Changes from version 1.x

Version 1.x treated `gw.npy` as a geometric vertical velocity in the physics
residual and in the kinematic boundary condition, and imposed the boundary
condition at the lowest level, where omega is zero in the GCM output. All
configurations were therefore retrained with the corrected formulation, and the
set of configurations was revised. The code, weights, and results of version 1.x
are not part of this version and should not be used.

## 9. Licence and citation

* Code (`code/`, `jobs/`, `analysis/`): MIT licence (`LICENSE`).
* Trained weights (`weights/`) and results (`results/`): CC BY 4.0 (`LICENSE-weights`).

This repository is archived at Zenodo. The DOI below is the concept DOI, which
always resolves to the latest version; the DOI of each version is given on the
Zenodo record.

> Fukazawa, K. (2026) GCM_PINNs: code, trained weights, and evaluation results of a
> physics-informed neural network surrogate model for a general circulation model
> (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.24050683

If you use this code or these weights, please also cite the accompanying manuscript:

> Fukazawa, K., Miyoshi, Y. Development and evaluation of a physics-informed
> neural network surrogate model for a general circulation model. Submitted to
> Earth, Planets and Space.
