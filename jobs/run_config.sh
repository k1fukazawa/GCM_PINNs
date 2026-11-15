#!/bin/bash
# Train one configuration of the paper.
#   bash jobs/run_config.sh <name> <seed> [data_dir] [out_dir]
# <name>: M0 M1 M2 M3 M4 M5  M2+S M3+S M4+S M5+S  M2-P0 M2-P1 M2-P10 M5-P0  M2-2E M5-2E  M2-SZA M5-SZA
# One run takes 28-38 min on one node with 112 CPU threads (twice as long for the -2E runs).
set -e
NAME=$1; SEED=${2:-0}; DATA=${3:-./npy}; OUT=${4:-./runs/${NAME}_seed${SEED}}
THREADS=${THREADS:-112}
LP=0.03; EPOCHS=6; TW=1.3; PW=1.8; EXTRA=""
M0='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 0 --level_weight_floor 0 --top_boost 1.0 --w_bc 0 --w_mass 0 --w_terr 0 --w_wind_mass 0 --w_zonal 0 --w_spec 0 --w_spec_high 0 --no-use_terrain --inference_hypervisc --no-mass_fix --no-inc_norm --no-finetune_massfix --common_norm'
M2='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 40 --level_weight_floor 0 --top_boost 1.0 --w_bc 0 --w_mass 0 --w_terr 0 --w_wind_mass 0 --w_zonal 0 --w_spec 0 --w_spec_high 0 --no-use_terrain --inference_hypervisc --no-mass_fix --no-inc_norm --no-finetune_massfix'
M2S='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 40 --level_weight_floor 0 --top_boost 1.0 --w_bc 0 --w_mass 0 --w_terr 0 --w_wind_mass 0 --w_zonal 0 --w_spec 0.05 --w_spec_high 0 --no-use_terrain --inference_hypervisc --no-mass_fix --no-inc_norm --no-finetune_massfix'
M3='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 40 --level_weight_floor 0 --top_boost 1.0 --w_bc 0.1 --w_mass 0 --w_terr 0 --w_wind_mass 0 --w_zonal 0 --w_spec 0 --w_spec_high 0 --use_terrain --inference_hypervisc --no-mass_fix --no-inc_norm --no-finetune_massfix'
M3S='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 40 --level_weight_floor 0 --top_boost 1.0 --w_bc 0.1 --w_mass 0 --w_terr 0 --w_wind_mass 0 --w_zonal 0 --w_spec 0.05 --w_spec_high 0 --use_terrain --inference_hypervisc --no-mass_fix --no-inc_norm --no-finetune_massfix'
M4='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 40 --level_weight_floor 0.1 --top_boost 2.0 --w_bc 0.3 --w_mass 0.5 --w_terr 0.1 --w_wind_mass 0.3 --w_zonal 0.3 --w_spec 0 --w_spec_high 0 --use_terrain --mass_fix --no-inc_norm --no-finetune_massfix'
M4S='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 40 --level_weight_floor 0.1 --top_boost 2.0 --w_bc 0.3 --w_mass 0.5 --w_terr 0.1 --w_wind_mass 0.3 --w_zonal 0.3 --w_spec 0.05 --w_spec_high 0.1 --use_terrain --mass_fix --no-inc_norm --no-finetune_massfix'
M5='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 40 --level_weight_floor 0.1 --top_boost 2.0 --w_bc 0.3 --w_mass 0.5 --w_terr 0.1 --w_wind_mass 0.3 --w_zonal 0.3 --w_spec 0 --w_spec_high 0 --use_terrain --mass_fix --inc_norm --finetune_massfix'
M5S='--lambda_phys $LP --noise 0.05 --level_weight_scale_km 40 --level_weight_floor 0.1 --top_boost 2.0 --w_bc 0.3 --w_mass 0.5 --w_terr 0.1 --w_wind_mass 0.3 --w_zonal 0.3 --w_spec 0.05 --w_spec_high 0.1 --use_terrain --mass_fix --inc_norm --finetune_massfix'
case "$NAME" in
  M0) FLAGS=$M0; TW=1.0; PW=1.0;;
  M1) FLAGS=$M2; TW=1.0; PW=1.0;;
  M2) FLAGS=$M2;;
  M3) FLAGS=$M3;;
  M4) FLAGS=$M4;;
  M5) FLAGS=$M5;;
  M2+S) FLAGS=$M2S;;
  M3+S) FLAGS=$M3S;;
  M4+S) FLAGS=$M4S;;
  M5+S) FLAGS=$M5S;;
  M2-P0) FLAGS=$M2; LP=0;;
  M2-P1) FLAGS=$M2; LP=1.0;;
  M2-P10) FLAGS=$M2; LP=10.0;;
  M5-P0) FLAGS=$M5; LP=0;;
  M2-2E) FLAGS=$M2; EPOCHS=12;;
  M5-2E) FLAGS=$M5; EPOCHS=12;;
  M2-SZA) FLAGS=$M2; EXTRA="--solar";;
  M5-SZA) FLAGS=$M5; EXTRA="--solar";;
  *) echo "unknown configuration: $NAME" >&2; exit 1;;
esac
FLAGS=${FLAGS//\$LP/$LP}
mkdir -p "$OUT"
cd "$(dirname "$0")/../code"
python train.py --dir "$DATA" --levels 48 --stride 1 \
  --ch 48 96 64 32 --epochs_per_stage $EPOCHS --rollouts 1 6 12 24 \
  --train_days 25 --clip_inc 1.0 --hypervisc 0.05 --bptt_steps 2 --amp fp32 --channels_last --threads "$THREADS" \
  --top_levels 5 $FLAGS $EXTRA --ema_decay 0 \
  --u_weight 1.0 --v_weight 1.0 --w_weight 1.0 --T_weight $TW --Ps_weight $PW \
  --w_var omega --omega_to_pa 100 --bc_lower kinematic --bc_level 1 \
  --seed "$SEED" --save_every 0 --tag "${NAME}_seed${SEED}" --outdir "$OUT"
