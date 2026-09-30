#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: run_independent_paper5_remote.sh 0|1}"
if [[ "$gpu" != 0 && "$gpu" != 1 ]]; then
  echo 'GPU must be 0 or 1' >&2
  exit 2
fi

code=/ssd1/code/Multi-Organ_Foundation_Model
data=/ssd1/data/Multi-Organ_Foundation_Model
root="$data/runs/unet_paper5_ct7_2_v1"
logs="$data/runs/paper5_ct7_2_independent-logs"
python="$code/.conda/bin/python"
mkdir -p "$logs"
cd "$code"

streak=0
while (( streak < 3 )); do
  IFS=, read -r usage temperature < <(nvidia-smi -i "$gpu" \
    --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
  usage="${usage//[[:space:]]/}"
  temperature="${temperature//[[:space:]]/}"
  if (( usage < 85 && temperature < 82 )); then
    (( streak += 1 ))
  else
    streak=0
  fi
  echo "$(date -Is) gpu=$gpu utilization=$usage temperature=$temperature checks=$streak/3"
  if (( streak < 3 )); then sleep 30; fi
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
common=(--data-root "$data/pointed_data" --output "$root" --mode independent
        --epochs 50 --size 256 --base 16 --batch-size 2 --quota 500
        --workers 2 --seed 42 --lr 0.0003
        --abdomen-split ct7_2 --abdomen-labels paper5
        --abdomen-positive-sampling 0.5 --device cuda:0)

if [[ "$gpu" == 0 ]]; then
  domains=(BUSI MMOTU DDTI AUL FALLMUD)
else
  while [[ ! -f "$root/records.json" || ! -f "$root/protocol.json" ]]; do
    echo "$(date -Is) waiting for frozen baseline records"
    sleep 30
  done
  domains=(Fetal_HC CCA CAMUS AbdomenUS)
fi

for domain in "${domains[@]}"; do
  resume=()
  if [[ -f "$root/records.json" ]]; then resume=(--resume); fi
  echo "$(date -Is) starting independent U-Net: $domain"
  "$python" -m unet_moe.baseline_suite "${common[@]}" \
    --domains "$domain" "${resume[@]}" > "$logs/${domain}.log" 2>&1
  echo "$(date -Is) independent U-Net complete: $domain"
done

echo "$(date -Is) gpu=$gpu assigned models complete"
