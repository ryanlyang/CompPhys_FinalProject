#!/usr/bin/env bash
#SBATCH --job-name=prelimReimpl
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=restart_studies/logs/prelim_reimpl_%j.out
#SBATCH --error=restart_studies/logs/prelim_reimpl_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Cluster sbatch launcher for:
#   restart_studies/reimplement_preliminary_studies.py
#
# Intended scope:
# - clean baseline run
# - corruption benchmark + shift-metric calibration/ranking
# - interpretability method effectiveness
# - sanity checks
# - single seed
# -----------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryan/Documents/School/CompPhys/new_final_project}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/restart_studies/reimplement_preliminary_studies.py}"

DATA_DIR="${DATA_DIR:-/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/restart_studies/results}"
RUN_NAME="${RUN_NAME:-prelim_reimpl_seed52_cluster}"
SEED="${SEED:-52}"

# Model/training defaults (single-seed "real resource" run)
FEATURE_MODE="${FEATURE_MODE:-full}"
MAX_CONSTITS="${MAX_CONSTITS:-128}"
N_TRAIN_JETS="${N_TRAIN_JETS:-150000}"
N_VAL_JETS="${N_VAL_JETS:-50000}"
N_TEST_JETS="${N_TEST_JETS:-150000}"

BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
LR="${LR:-7e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
EMBED_DIM="${EMBED_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-6}"
FF_DIM="${FF_DIM:-512}"
DROPOUT="${DROPOUT:-0.1}"

TARGET_CLASS="${TARGET_CLASS:-HToBB}"
BACKGROUND_CLASS="${BACKGROUND_CLASS:-ZJetsToNuNu}"

# Corruption suite from preliminary design
CORRUPTIONS="${CORRUPTIONS:-pt_noise:0.03,pt_noise:0.06,eta_phi_jitter:0.02,eta_phi_jitter:0.05,dropout:0.05,dropout:0.10,merge:0.10,merge:0.20,global_scale:0.03}"

# Interpretability benchmark knobs
EXPLAIN_SUBSET_SIZE="${EXPLAIN_SUBSET_SIZE:-20000}"
EXPLAIN_BATCH_SIZE="${EXPLAIN_BATCH_SIZE:-128}"
MASK_FRACS="${MASK_FRACS:-0.02,0.05,0.10,0.20}"
IG_STEPS="${IG_STEPS:-16}"
SMOOTHGRAD_SAMPLES="${SMOOTHGRAD_SAMPLES:-16}"
SMOOTHGRAD_SIGMA="${SMOOTHGRAD_SIGMA:-0.10}"
RANDOM_MASK_REPEATS="${RANDOM_MASK_REPEATS:-3}"

# Runtime
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHUFFLE_FILES="${SHUFFLE_FILES:-0}"

# Conda activation (override as needed)
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-atlas_kd}"

mkdir -p "${PROJECT_ROOT}/restart_studies/logs"

cd "${PROJECT_ROOT}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "[preflight] WARNING: conda.sh not found at ${CONDA_SH}. Using current shell Python."
fi

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
  else
    echo "[preflight] WARNING: conda env '${CONDA_ENV}' not found."
    echo "[preflight] Falling back to 'base' if available."
    if conda env list | awk '{print $1}' | grep -qx "base"; then
      conda activate base
    fi
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONHASHSEED="${SEED}"

echo "[preflight] python: $(command -v "${PYTHON_BIN}" || true)"
"${PYTHON_BIN}" - << 'PY'
import importlib.util as iu
mods = ["torch", "numpy", "sklearn", "scipy", "uproot", "awkward", "tqdm"]
missing = [m for m in mods if iu.find_spec(m) is None]
if missing:
    raise SystemExit(f"[preflight] Missing required Python modules: {missing}")
print("[preflight] module check OK")
PY

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[preflight] ERROR: script not found: ${SCRIPT_PATH}" >&2
  exit 2
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[preflight] ERROR: data dir not found: ${DATA_DIR}" >&2
  exit 2
fi

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --data_dir "${DATA_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --feature_mode "${FEATURE_MODE}"
  --max_constits "${MAX_CONSTITS}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_val_jets "${N_VAL_JETS}"
  --n_test_jets "${N_TEST_JETS}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --embed_dim "${EMBED_DIM}"
  --num_heads "${NUM_HEADS}"
  --num_layers "${NUM_LAYERS}"
  --ff_dim "${FF_DIM}"
  --dropout "${DROPOUT}"
  --target_class "${TARGET_CLASS}"
  --background_class "${BACKGROUND_CLASS}"
  --corruptions "${CORRUPTIONS}"
  --explain_subset_size "${EXPLAIN_SUBSET_SIZE}"
  --explain_batch_size "${EXPLAIN_BATCH_SIZE}"
  --mask_fracs "${MASK_FRACS}"
  --ig_steps "${IG_STEPS}"
  --smoothgrad_samples "${SMOOTHGRAD_SAMPLES}"
  --smoothgrad_sigma "${SMOOTHGRAD_SIGMA}"
  --random_mask_repeats "${RANDOM_MASK_REPEATS}"
)

if [[ "${SHUFFLE_FILES}" == "1" ]]; then
  CMD+=(--shuffle_files)
fi

echo "============================================================"
echo "Preliminary Study Reimplementation (Single Seed)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "Data: ${DATA_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[done] Results:"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/summary.json"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/clean_metrics.json"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/corruption_metrics.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/correlations.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/top_shift_metric_ranking.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/method_effectiveness_summary.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/sanity_checks.json"
