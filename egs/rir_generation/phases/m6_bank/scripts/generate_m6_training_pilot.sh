#!/usr/bin/env bash
set -euo pipefail

backend="${1:-pyroomacoustics}"
output_root="${2:-egs/rir_generation/exp/rir_realism/m6/training_pilot}"
pilot_rooms="${PURESOUND_M6_PILOT_ROOMS:-1000}"
rir_per_room="${PURESOUND_M6_PILOT_RIR_PER_ROOM:-4}"
pilot_workers="${PURESOUND_M6_PILOT_WORKERS:-8}"
pilot_seed="${PURESOUND_M6_PILOT_SEED:-1337}"

case "${backend}" in
  pyroomacoustics|path-events-m4) ;;
  *)
    echo "backend must be pyroomacoustics or path-events-m4" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
bank_dir="${output_root}/${backend}_bank"
release_dir="${output_root}/${backend}_release"

if [[ -e "${release_dir}" ]]; then
  echo "release already exists: ${release_dir}" >&2
  echo "use a new output root; the script will not overwrite a release" >&2
  exit 2
fi

cd "${repo_root}"

PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir "${bank_dir}" \
  --n-rooms "${pilot_rooms}" \
  --rir-per-room "${rir_per_room}" \
  --sample-rate 16000 \
  --duration 1.6 \
  --scene-version v1 \
  --room-type mixed \
  --output-mode calibrated \
  --record-realized-metrics \
  --low-backend pytard \
  --high-backend "${backend}" \
  --num-workers "${pilot_workers}" \
  --seed "${pilot_seed}" \
  --emit-m6-manifest \
  --m6-bank-id "puresound-m6-pilot-${backend}-${pilot_seed}" \
  --resume

PYTHONPATH=. .venv/bin/python egs/rir_generation/phases/m6_bank/scripts/run_m6_item_qc.py \
  --bank "${bank_dir}"

PYTHONPATH=. .venv/bin/python egs/rir_generation/phases/m6_bank/scripts/build_m6_variant_release.py \
  --source-bank "${bank_dir}" \
  --output-dir "${release_dir}" \
  --release-id "puresound-m6-pilot-${backend}-${pilot_seed}"

echo "M6 pilot release ready: ${release_dir}"
echo "training recipe: synthetic_calibrated, split: train"
