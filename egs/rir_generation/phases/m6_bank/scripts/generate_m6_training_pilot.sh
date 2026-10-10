#!/usr/bin/env bash
#
# One arm of the matched Pyroomacoustics / PathEvents-M4 training pilot.
#
# This delegates to generate_m6_bank.py rather than driving the generator, QC,
# and release steps itself.  It used to duplicate them, and the copy drifted:
# it passed --low-backend pytard while M6 ships pytard-material, so both pilot
# arms would have been rendered on a low band with a global RT60 envelope
# instead of per-mode material damping.  Delegating keeps the pilot on
# whatever M6 actually defaults to.
#
# What this script adds over calling generate_m6_bank.py directly: a refusal to
# overwrite an existing release, and stable pilot bank/release identifiers.
#
# Environment knobs:
#   PURESOUND_M6_PILOT_ROOMS         rooms per arm            (default 1000)
#   PURESOUND_M6_PILOT_RIR_PER_ROOM  RIRs per room            (default 4)
#   PURESOUND_M6_PILOT_WORKERS       generator workers        (default 8)
#   PURESOUND_M6_PILOT_QC_WORKERS    QC / release workers     (default 1)
#   PURESOUND_M6_PILOT_SEED          shared scene seed        (default 1337)
#   PURESOUND_M6_PILOT_LOW_BACKEND   low band  (default: generate_m6_bank.py's)
#   PURESOUND_M6_PILOT_GPU_DEVICES   CUDA devices, e.g. "0,1" (default: none)
#
# The seed must be identical across the two arms: it is what makes the scenes
# matched, and therefore what makes the comparison an A/B rather than two
# unrelated banks.  The low backend must be identical for the same reason — it
# is the shared half of the renderer, and the crossover sits between the two.
#
# Only the low band can use the GPU; the high band is always CPU.  Set
# LOW_BACKEND=pytard-cupy-material together with GPU_DEVICES to use it.
set -euo pipefail

backend="${1:-path-events-m4}"
output_root="${2:-egs/rir_generation/exp/rir_realism/m6/training_pilot}"
pilot_rooms="${PURESOUND_M6_PILOT_ROOMS:-1000}"
rir_per_room="${PURESOUND_M6_PILOT_RIR_PER_ROOM:-4}"
pilot_workers="${PURESOUND_M6_PILOT_WORKERS:-8}"
qc_workers="${PURESOUND_M6_PILOT_QC_WORKERS:-1}"
pilot_seed="${PURESOUND_M6_PILOT_SEED:-1337}"
low_backend="${PURESOUND_M6_PILOT_LOW_BACKEND:-}"
gpu_devices="${PURESOUND_M6_PILOT_GPU_DEVICES:-}"

extra_args=()
if [[ -n "${low_backend}" ]]; then
  extra_args+=(--low-backend "${low_backend}")
fi
if [[ -n "${gpu_devices}" ]]; then
  extra_args+=(--gpu-devices "${gpu_devices}")
fi

case "${backend}" in
  pyroomacoustics|path-events-m4) ;;
  *)
    echo "backend must be pyroomacoustics or path-events-m4" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
release_dir="${output_root}/${backend}_release"

if [[ -e "${release_dir}" ]]; then
  echo "release already exists: ${release_dir}" >&2
  echo "use a new output root; the script will not overwrite a release" >&2
  exit 2
fi

cd "${repo_root}"

PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir "${output_root}" \
  --backend "${backend}" \
  --n-rooms "${pilot_rooms}" \
  --rir-per-room "${rir_per_room}" \
  --num-workers "${pilot_workers}" \
  --qc-workers "${qc_workers}" \
  --seed "${pilot_seed}" \
  --m6-bank-id "puresound-m6-pilot-${backend}-${pilot_seed}" \
  --release-id "puresound-m6-pilot-${backend}-${pilot_seed}" \
  "${extra_args[@]}" \
  --resume

echo "M6 pilot release ready: ${release_dir}"
echo "training recipe: synthetic_calibrated, split: train"
