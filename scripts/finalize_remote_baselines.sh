#!/usr/bin/env bash
set -euo pipefail

code=/ssd1/code/Multi-Organ_Foundation_Model
data=/ssd1/data/Multi-Organ_Foundation_Model
run="$data/runs/unet_baseline_9v1"
names=(shared BUSI MMOTU DDTI AUL FALLMUD Fetal_HC CCA CAMUS AbdomenUS)

while true; do
  missing=()
  for name in "${names[@]}"; do
    [[ -f "$run/$name/test_metrics.json" ]] || missing+=("$name")
  done
  if ((${#missing[@]} == 0)); then
    break
  fi
  if ! tmux has-session -t mofm_gpu0 2>/dev/null && ! tmux has-session -t mofm_gpu1 2>/dev/null; then
    echo "Training processes ended with incomplete tests: ${missing[*]}" >&2
    exit 1
  fi
  sleep 60
done

cd "$code"
export OMP_NUM_THREADS=2
"$code/.conda/bin/python" -u -m unet_moe.baseline_suite \
  --data-root "$data/pointed_data" --output "$run" \
  --mode summarize --epochs 50 --size 256 --base 16 \
  --batch-size 2 --quota 500 --workers 2 --seed 42 \
  --device cpu --resume
