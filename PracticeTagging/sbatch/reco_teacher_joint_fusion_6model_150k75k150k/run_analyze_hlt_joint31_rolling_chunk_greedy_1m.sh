#!/usr/bin/env bash
#SBATCH --job-name=an31roll1m
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31_roll1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31_roll1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

FUSION_JSON_DEFAULT_CHECKPOINTS="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"
FUSION_JSON_DEFAULT_DOWNLOAD_A="download_checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"
FUSION_JSON_DEFAULT_DOWNLOAD_B="download_checkpoints/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"

if [[ -z "${FUSION_JSON:-}" ]]; then
  if [[ -f "${FUSION_JSON_DEFAULT_CHECKPOINTS}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_CHECKPOINTS}"
  elif [[ -f "${FUSION_JSON_DEFAULT_DOWNLOAD_A}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_DOWNLOAD_A}"
  elif [[ -f "${FUSION_JSON_DEFAULT_DOWNLOAD_B}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_DOWNLOAD_B}"
  else
    FUSION_JSON="${FUSION_JSON_DEFAULT_CHECKPOINTS}"
  fi
fi

TARGET_TPR="${TARGET_TPR:-0.50}"
ANCHOR_MODEL="${ANCHOR_MODEL:-joint_delta}"
CANDIDATE_MODELS="${CANDIDATE_MODELS:-}"

VAL_POOL_SIZE="${VAL_POOL_SIZE:-1000000}"
VAL_POOL_OFFSET="${VAL_POOL_OFFSET:-0}"
N_CHUNKS="${N_CHUNKS:-5}"

CANDIDATE_TOPK_FIT="${CANDIDATE_TOPK_FIT:-31}"
MAX_STEPS="${MAX_STEPS:-20}"
PATIENCE="${PATIENCE:-5}"
W_STEP="${W_STEP:-0.01}"
CALIBRATION="${CALIBRATION:-raw}"
MIN_FIT_GAIN="${MIN_FIT_GAIN:-1e-6}"
MIN_CAL_GAIN="${MIN_CAL_GAIN:-1e-6}"
MIN_CUM_CAL_GAIN="${MIN_CUM_CAL_GAIN:-1e-6}"
TOPK_DIAGNOSTIC="${TOPK_DIAGNOSTIC:-10}"
SEED="${SEED:-0}"

OUT_DIR="${OUT_DIR:-}"
REPORT_JSON="${REPORT_JSON:-}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -f "${FUSION_JSON}" ]]; then
  echo "ERROR: FUSION_JSON not found: ${FUSION_JSON}" >&2
  exit 1
fi

# Clamp val-pool request to what exists in the selected fusion artifact.
read -r VAL_POOL_SIZE_ADJ VAL_POOL_OFFSET_ADJ VAL_POOL_AVAIL <<EOF
$(python3 - "${FUSION_JSON}" "${VAL_POOL_SIZE}" "${VAL_POOL_OFFSET}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

fusion_json = Path(sys.argv[1]).expanduser().resolve()
req_size = int(sys.argv[2])
req_off = int(sys.argv[3])
repo_root = Path.cwd()

fusion = json.loads(fusion_json.read_text())
run_dirs = fusion.get("run_dirs", {})
score_files = run_dirs.get("score_files", {}) if isinstance(run_dirs, dict) else {}
joint_path = score_files.get("joint_delta")
if not isinstance(joint_path, str):
    # Fallback to canonical location if score_files did not include joint_delta.
    jd = run_dirs.get("joint_delta_run_dir") if isinstance(run_dirs, dict) else None
    if isinstance(jd, str):
        joint_path = str(Path(jd) / "fusion_scores_val_test.npz")
    else:
        raise RuntimeError("Could not resolve joint_delta score npz for val-size clamp.")

npz_path = Path(joint_path)
if not npz_path.is_absolute():
    npz_path = (repo_root / npz_path).resolve()
if not npz_path.exists():
    raise FileNotFoundError(f"joint score npz not found: {npz_path}")

z = np.load(npz_path)
n_val = int(np.asarray(z["labels_val"]).shape[0])

off = max(0, int(req_off))
size = max(1, int(req_size))
if off >= n_val:
    off = 0
if size > n_val:
    size = n_val
if off + size > n_val:
    off = max(0, n_val - size)

print(size, off, n_val)
PY
)
EOF

if [[ "${VAL_POOL_SIZE_ADJ}" != "${VAL_POOL_SIZE}" || "${VAL_POOL_OFFSET_ADJ}" != "${VAL_POOL_OFFSET}" ]]; then
  echo "Adjusted val pool to fit artifact: requested size/offset=${VAL_POOL_SIZE}/${VAL_POOL_OFFSET} -> ${VAL_POOL_SIZE_ADJ}/${VAL_POOL_OFFSET_ADJ} (available val=${VAL_POOL_AVAIL})"
fi
VAL_POOL_SIZE="${VAL_POOL_SIZE_ADJ}"
VAL_POOL_OFFSET="${VAL_POOL_OFFSET_ADJ}"

CMD=(
  python analyze_hlt_joint31_rolling_chunk_greedy.py
  --fusion_json "${FUSION_JSON}"
  --target_tpr "${TARGET_TPR}"
  --anchor_model "${ANCHOR_MODEL}"
  --candidate_topk_fit "${CANDIDATE_TOPK_FIT}"
  --val_pool_size "${VAL_POOL_SIZE}"
  --val_pool_offset "${VAL_POOL_OFFSET}"
  --n_chunks "${N_CHUNKS}"
  --max_steps "${MAX_STEPS}"
  --patience "${PATIENCE}"
  --w_step "${W_STEP}"
  --calibration "${CALIBRATION}"
  --min_fit_gain "${MIN_FIT_GAIN}"
  --min_cal_gain "${MIN_CAL_GAIN}"
  --min_cum_cal_gain "${MIN_CUM_CAL_GAIN}"
  --topk_diagnostic "${TOPK_DIAGNOSTIC}"
  --seed "${SEED}"
)

if [[ -n "${CANDIDATE_MODELS}" ]]; then
  CMD+=(--candidate_models "${CANDIDATE_MODELS}")
fi
if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${REPORT_JSON}" ]]; then
  CMD+=(--report_json "${REPORT_JSON}")
fi

echo "============================================================"
echo "31-Model Rolling-Chunk Greedy Fusion (val-size clamped if needed)"
echo "Fusion json:         ${FUSION_JSON}"
echo "Target TPR:          ${TARGET_TPR}"
echo "Anchor:              ${ANCHOR_MODEL}"
echo "Val pool size/offset:${VAL_POOL_SIZE}/${VAL_POOL_OFFSET}"
echo "Chunks:              ${N_CHUNKS}"
echo "Greedy:              topk=${CANDIDATE_TOPK_FIT}, max_steps=${MAX_STEPS}, w_step=${W_STEP}"
echo "Acceptance gains:    fit>=${MIN_FIT_GAIN}, cal>=${MIN_CAL_GAIN}, cum>=${MIN_CUM_CAL_GAIN}"
echo "Calibration:         ${CALIBRATION}"
echo "Diagnostics:         rolling_step_log.csv, rolling_fit_topk_by_step.csv, rolling_chunk_stats.csv,"
echo "                     rolling_calibration_diagnostics.csv, rolling_prefilter_single_model.csv"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
