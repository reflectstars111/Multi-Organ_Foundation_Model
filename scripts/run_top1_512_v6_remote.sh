#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: run_top1_512_v6_remote.sh task_id|region 0|1}"
gpu="${2:?usage: run_top1_512_v6_remote.sh task_id|region 0|1}"
if [[ "$mode" != task_id && "$mode" != region ]]; then
  echo 'mode must be task_id or region' >&2
  exit 2
fi
if [[ "$gpu" != 0 && "$gpu" != 1 ]]; then
  echo 'GPU must be 0 or 1' >&2
  exit 2
fi

code=/ssd1/code/Multi-Organ_Foundation_Model
data=/ssd1/data/Multi-Organ_Foundation_Model
python="$code/.conda/bin/python"
run="$data/runs/${mode}_top1_512_v6_final_seed42"
if [[ -e "$run" ]]; then
  echo "Refusing to overwrite existing run: $run" >&2
  exit 3
fi
cd "$code"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

common=(--data-root "$data/pointed_data" --output "$run"
        --epochs 80 --batch-size 16 --size 512 --base 24
        --top-k 1 --workers 0 --lr 0.0001 --seed 42 --device cuda:0
        --abdomen-split ct_final --split-policy trainmore_v6
        --abdomen-labels paper5)
if [[ "$mode" == task_id ]]; then
  exec "$python" -m unet_moe.train "${common[@]}" \
    --experts 9 --decoder moe --steps-per-epoch 282
else
  exec "$python" -m unet_moe.region_train "${common[@]}" \
    --quota-per-domain 500 --abdomen-positive-sampling 0.5 \
    --lr-schedule constant --coarse-weight 0.3 --route-weight 0.1
fi
