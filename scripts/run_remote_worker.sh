#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: run_remote_worker.sh 0|1}"
case "$gpu" in
  0) domains='BUSI,DDTI,FALLMUD,CCA,AbdomenUS' ;;
  1) domains='MMOTU,AUL,Fetal_HC,CAMUS' ;;
  *) echo 'GPU must be 0 or 1' >&2; exit 2 ;;
esac

code=/ssd1/code/Multi-Organ_Foundation_Model
data=/ssd1/data/Multi-Organ_Foundation_Model
logs="$data/runs/unet_baseline_9v1-logs"
mkdir -p "$logs"
cd "$code"
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=2
"$code/.conda/bin/python" -u -m unet_moe.baseline_suite \
  --data-root "$data/pointed_data" \
  --output "$data/runs/unet_baseline_9v1" \
  --mode independent --domains "$domains" --resume \
  --epochs 50 --size 256 --base 16 --batch-size 2 --quota 500 \
  --workers 2 --seed 42 --device cuda \
  > "$logs/gpu${gpu}.log" 2>&1
