#!/usr/bin/env bash
# Prespecified 512px validation ablation followed by the three-seed comparison.
set -euo pipefail
code="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs="${1:-/ssd1/data/Multi-Organ_Foundation_Model/runs}"
runs="$(realpath -m "$runs")"
gpu="${2:-1}"
resume=()
if [[ "${3:-}" == --resume ]]; then
  resume=(--resume)
elif [[ -n "${3:-}" ]]; then
  echo 'Third argument must be --resume' >&2
  exit 2
fi
logs="$runs/research_512_20260930-logs"
ablation="$runs/empty_ablation_512_20260930"
fair="$runs/fair_512_seed42_43_44_20260930"
mkdir -p "$logs"
cd "$code"
stage=starting
write_status() {
  "$code/.conda/bin/python" - "$logs/status.json" "$1" "${2:-0}" <<'PY'
import json, os, sys
from datetime import datetime
from pathlib import Path
p = Path(sys.argv[1])
now = datetime.now().astimezone().isoformat()
previous = json.loads(p.read_text()) if p.exists() else {}
record = dict(stage=sys.argv[2], exit_code=int(sys.argv[3]), controller_pid=os.getppid(),
              started_at=previous.get('started_at', now), updated_at=now, input_size=512)
temporary = p.with_suffix('.tmp')
temporary.write_text(json.dumps(record, indent=2) + '\n')
temporary.replace(p)
PY
}
trap 'failure=$?; write_status "failed:${stage}" "$failure"; exit "$failure"' ERR
stage=ablation
write_status "$stage"
ablation_resume=()
if [[ ${#resume[@]} -gt 0 && -d "$ablation" ]]; then ablation_resume=(--resume); fi
bash scripts/run_empty_ablation_remote.sh "$ablation" "$gpu" "${ablation_resume[@]}" >> "$logs/ablation.log" 2>&1
stage=ablation_summary
write_status "$stage"
if [[ ! -f "$ablation/comparison.json" ]]; then
  "$code/.conda/bin/python" -m scripts.summarize_empty_ablation \
    --runs-root "$ablation" >> "$logs/ablation.log" 2>&1
fi
stage=fair_comparison
write_status "$stage"
fair_resume=()
if [[ ${#resume[@]} -gt 0 && -f "$fair/protocol.json" ]]; then fair_resume=(--resume); fi
bash scripts/run_fair_suite_remote.sh "$fair" "$gpu" "${fair_resume[@]}" >> "$logs/fair.log" 2>&1
write_status complete
