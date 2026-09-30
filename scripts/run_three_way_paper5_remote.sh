#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: run_three_way_paper5_remote.sh 0|1}"
if [[ "$gpu" != 0 && "$gpu" != 1 ]]; then
  echo 'GPU must be 0 or 1' >&2
  exit 2
fi

code=/ssd1/code/Multi-Organ_Foundation_Model
data=/ssd1/data/Multi-Organ_Foundation_Model
root="$data/runs/unet_paper5_ct7_2_v1"
moe="$data/runs/region_paper5_ct7_2_v1"
logs="$data/runs/paper5_ct7_2_v1-logs"
python="$code/.conda/bin/python"
mkdir -p "$logs"
cd "$code"

wait_for_idle_gpu() {
  local streak=0 usage temperature
  while (( streak < 3 )); do
    IFS=, read -r usage temperature < <(nvidia-smi -i "$gpu" \
      --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
    usage="${usage//[[:space:]]/}"
    temperature="${temperature//[[:space:]]/}"
    if (( usage < 20 && temperature < 80 )); then
      (( streak += 1 ))
    else
      streak=0
    fi
    echo "$(date -Is) gpu=$gpu utilization=$usage temperature=$temperature idle_checks=$streak/3"
    if (( streak < 3 )); then sleep 60; fi
  done
}

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
common=(--data-root "$data/pointed_data" --epochs 50 --size 256 --base 16
        --batch-size 2 --workers 2 --seed 42 --lr 0.0003
        --abdomen-split ct7_2 --abdomen-labels paper5
        --abdomen-positive-sampling 0.5 --device cuda:0)

wait_for_idle_gpu
if [[ "$gpu" == 0 ]]; then
  resume=()
  if [[ -f "$moe/config.json" ]]; then resume=(--resume); fi
  echo "$(date -Is) starting region Top-1 MoE"
  "$python" -m unet_moe.region_train "${common[@]}" \
    --output "$moe" --top-k 1 --quota-per-domain 500 \
    "${resume[@]}" > "$logs/moe.log" 2>&1
  echo "$(date -Is) region Top-1 MoE complete"
  while [[ ! -f "$root/records.json" ]]; do sleep 60; done
  domains=(BUSI MMOTU DDTI AUL)
else
  resume=()
  if [[ -f "$root/records.json" ]]; then resume=(--resume); fi
  echo "$(date -Is) starting shared U-Net"
  "$python" -m unet_moe.baseline_suite "${common[@]}" \
    --output "$root" --mode shared --quota 500 \
    "${resume[@]}" > "$logs/shared.log" 2>&1
  echo "$(date -Is) shared U-Net complete"
  domains=(FALLMUD Fetal_HC CCA CAMUS AbdomenUS)
fi

for domain in "${domains[@]}"; do
  echo "$(date -Is) starting independent U-Net: $domain"
  "$python" -m unet_moe.baseline_suite "${common[@]}" \
    --output "$root" --mode independent --domains "$domain" \
    --quota 500 --resume > "$logs/independent_${domain}.log" 2>&1
  echo "$(date -Is) independent U-Net complete: $domain"
done

echo "$(date -Is) gpu=$gpu assigned training complete"
