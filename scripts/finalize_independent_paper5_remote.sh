#!/usr/bin/env bash
set -euo pipefail
code=/ssd1/code/Multi-Organ_Foundation_Model
data=/ssd1/data/Multi-Organ_Foundation_Model
root="$data/runs/unet_paper5_ct7_2_v1"
cd "$code"
domains=(BUSI MMOTU DDTI AUL FALLMUD Fetal_HC CCA CAMUS AbdomenUS)
for attempt in $(seq 1 10080); do
  missing=()
  for domain in "${domains[@]}"; do
    [[ -f "$root/$domain/test_metrics.json" ]] || missing+=("$domain")
  done
  if (( ${#missing[@]} == 0 )); then
    "$code/.conda/bin/python" -m scripts.summarize_independent_paper5 --run "$root"
    echo "$(date -Is) all nine independent U-Nets complete"
    exit 0
  fi
  if (( attempt % 30 == 1 )); then
    echo "$(date -Is) waiting for: ${missing[*]}"
  fi
  sleep 60
done
echo 'Timed out before all models completed' >&2
exit 1
