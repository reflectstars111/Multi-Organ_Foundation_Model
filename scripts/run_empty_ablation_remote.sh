#!/usr/bin/env bash
# One prespecified candidate; validation only, same initialization and draws.
set -euo pipefail
code="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?usage: run_empty_ablation_remote.sh OUTPUT_ROOT [GPU=1] [--resume]}"
root="$(realpath -m "$root")"
gpu="${2:-1}"
resume=()
if [[ "${3:-}" == --resume ]]; then
  [[ -d "$root" ]] || { echo 'Missing ablation directory to resume' >&2; exit 2; }
elif [[ -n "${3:-}" || -e "$root" ]]; then
  echo 'Use a new ablation directory, or pass --resume as the third argument' >&2
  exit 2
fi
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Invalid GPU' >&2; exit 2; }
cd "$code"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec 9>"/tmp/multi_organ_fair_gpu_${gpu}.lock"
flock -n 9 || { echo 'GPU reserved by another project controller' >&2; exit 3; }
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
(( used <= 512 )) || { echo 'GPU occupied; leaving existing jobs alone' >&2; exit 3; }
mkdir -p "$root"
common=(--data-root "${UNET_MOE_DATA_ROOT:-/ssd1/data/Multi-Organ_Foundation_Model/pointed_data}"
        --models region --seeds 42 --split-seed 42 --top-k 1
        --size 512 --base 16 --batch-size 4 --epochs 12 --quota-per-domain 128
        --lr 0.0003 --workers 2 --device cuda:0
        --coarse-weight 0.3 --route-weight 0.1 --empty-fraction 0.01)
for weight in 0 0.1; do
  resume=()
  if [[ "${3:-}" == --resume && -f "$root/weight_${weight}/protocol.json" ]]; then
    resume=(--resume)
  fi
  echo "$(date -Is) validation-only empty-loss candidate weight=$weight"
  "$code/.conda/bin/python" -m unet_moe.fair_suite "${common[@]}" \
    --empty-weight "$weight" --output "$root/weight_${weight}" "${resume[@]}"
done
echo "$(date -Is) paired validation ablation complete; no test evaluated"
