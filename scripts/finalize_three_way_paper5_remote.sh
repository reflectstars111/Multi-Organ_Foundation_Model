#!/usr/bin/env bash
set -euo pipefail

code=/ssd1/code/Multi-Organ_Foundation_Model
data=/ssd1/data/Multi-Organ_Foundation_Model
runs="$data/runs"
baseline="$runs/unet_paper5_ct7_2_v1"
moe="$runs/region_paper5_ct7_2_v1"
python="$code/.conda/bin/python"
cd "$code"

domains=(BUSI MMOTU DDTI AUL FALLMUD Fetal_HC CCA CAMUS AbdomenUS)
for attempt in $(seq 1 10080); do
  missing=()
  [[ -f "$moe/test_metrics.json" ]] || missing+=(region_top1_moe)
  [[ -f "$baseline/shared/test_metrics.json" ]] || missing+=(shared_unet)
  for domain in "${domains[@]}"; do
    [[ -f "$baseline/$domain/test_metrics.json" ]] || missing+=("$domain")
  done
  if (( ${#missing[@]} == 0 )); then
    "$python" -m unet_moe.baseline_suite \
      --data-root "$data/pointed_data" --output "$baseline" \
      --mode summarize --epochs 50 --size 256 --base 16 --batch-size 2 \
      --quota 500 --workers 2 --seed 42 --lr 0.0003 \
      --abdomen-split ct7_2 --abdomen-labels paper5 \
      --abdomen-positive-sampling 0.5 --device cpu --resume
    "$python" -m scripts.summarize_three_way_paper5 --runs-root "$runs"
    echo "$(date -Is) three-way comparison complete"
    exit 0
  fi
  if (( attempt % 60 == 1 )); then
    echo "$(date -Is) waiting for: ${missing[*]}"
  fi
  sleep 60
done
echo 'Timed out before all models completed' >&2
exit 1
