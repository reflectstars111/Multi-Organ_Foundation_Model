#!/usr/bin/env bash
# Frozen three-seed comparison. Test evaluation starts only after all training.
set -euo pipefail

code="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${UNET_MOE_DATA_ROOT:-/ssd1/data/Multi-Organ_Foundation_Model/pointed_data}"
output="${1:?usage: run_fair_suite_remote.sh OUTPUT [GPU=1] [--resume]}"
output="$(realpath -m "$output")"
gpu="${2:-1}"
resume=()
if [[ "${3:-}" == --resume ]]; then
  resume=(--resume)
elif [[ -n "${3:-}" ]]; then
  echo 'Only --resume is accepted as the third argument' >&2
  exit 2
fi
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo 'GPU must be a nonnegative integer index' >&2
  exit 2
fi
cd "$code"
python="$code/.conda/bin/python"
[[ -x "$python" && -d "$data" ]] || { echo 'Missing project Python or data' >&2; exit 2; }

# A prepared parallel plan owns separate per-GPU shards and the global test
# barrier. Each worker handles its own strict resume and GPU process lock.
if [[ -f "$output/parallel_plan.json" ]]; then
  exec "$python" -m scripts.fair_parallel --output "$output" --gpu "$gpu"
fi

# Refuse to share an occupied GPU or race another instance of this controller.
exec 9>"/tmp/multi_organ_fair_gpu_${gpu}.lock"
flock -n 9 || { echo "A fair-suite controller already owns GPU $gpu" >&2; exit 3; }
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
if (( used > 512 )); then
  echo "GPU $gpu already uses ${used} MiB; leaving existing jobs alone" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
common=(--data-root "$data" --output "$output"
        --models shared,task_id,region --seeds 42,43,44 --split-seed 42
        --size 512 --base 16 --batch-size 4 --epochs 50
        --top-k 1 --experts 9
        --quota-per-domain 500 --lr 0.0003 --workers 2 --device cuda:0)

echo "$(date -Is) training all models and seeds; no test evaluation"
"$python" -m unet_moe.fair_suite "${common[@]}" "${resume[@]}"
echo "$(date -Is) all training complete; evaluating frozen checkpoints on test"
"$python" -m unet_moe.fair_suite "${common[@]}" --resume --evaluate-test
echo "$(date -Is) comparison complete"
