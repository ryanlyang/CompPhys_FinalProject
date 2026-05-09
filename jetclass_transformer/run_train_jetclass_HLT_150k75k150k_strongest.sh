#!/bin/bash
#SBATCH -J jcHLT150X
#SBATCH -p tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o offline_reconstructor_logs/jetclass_hlt_teacher_baseline_150k75k150k_strongest_%j.out
#SBATCH -e offline_reconstructor_logs/jetclass_hlt_teacher_baseline_150k75k150k_strongest_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOG_DIR="${WORKDIR}/offline_reconstructor_logs"
SAVE_DIR="${WORKDIR}/checkpoints/jetclass_hlt_teacher_baseline"

SCRIPT_CANDIDATES=(
  "${WORKDIR}/evaluate_jetclass_hlt_teacher_baseline.py"
  "/home/ryreu/atlas/PracticeTagging/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"
  "/home/ryreu/atlas/PracticeTagging/evaluate_jetclass_hlt_teacher_baseline.py"
  "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"
  "/home/ryan/ComputerScience/ATLAS/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"
)
SCRIPT=""
for c in "${SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SCRIPT="${c}"
    break
  fi
done

DATA_CANDIDATES=(
  "/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"
  "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0"
)
DATA_DIR=""
for d in "${DATA_CANDIDATES[@]}"; do
  if [[ -d "${d}" ]]; then
    DATA_DIR="${d}"
    break
  fi
done

cd "${WORKDIR}"

source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate atlas_kd

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

if [[ -z "${SCRIPT}" ]]; then
  echo "ERROR: evaluate_jetclass_hlt_teacher_baseline.py not found." >&2
  echo "Checked: ${SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi
if [[ -z "${DATA_DIR}" ]]; then
  echo "ERROR: jetclass_part0 data directory not found." >&2
  echo "Checked: ${DATA_CANDIDATES[*]}" >&2
  exit 2
fi

export PYTHONWARNINGS="ignore"
export TQDM_DISABLE=1

RAW_LOG="${LOG_DIR}/jetclass_hlt_teacher_baseline_raw_150k75k150k_strongest_${SLURM_JOB_ID}.log"

CMD=(
  python -u "${SCRIPT}"
  --data_dir "${DATA_DIR}"
  --save_dir "${SAVE_DIR}"
  --run_name "jetclass_part0_hlt_teacher_baseline_150k75k150k_strongest"
  --n_train_jets 150000
  --n_val_jets 75000
  --n_test_jets 150000
  --max_constits 128
  --feature_mode full
  --batch_size 512
  --epochs 30
  --patience 8
  --num_workers 2
  --device cuda
  --hlt_pt_threshold 1.8
  --eff_plateau_barrel 0.92
  --eff_plateau_endcap 0.82
  --eff_turnon_pt 1.5
  --eff_width_pt 0.50
  --smear_scale 1.45
  --reassign_scale 1.50
  --merge_prob_scale 1.35
)

echo "WORKDIR=${WORKDIR}"
echo "SCRIPT=${SCRIPT}"
echo "DATA_DIR=${DATA_DIR}"
echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${RAW_LOG}" | awk '
/Teacher ep [0-9]+:/ {
  n=$0; sub(/^.*Teacher ep /,"",n); sub(/:.*/,"",n);
  if ((n % 5) == 0) print;
  next
}
/Baseline ep [0-9]+:/ {
  n=$0; sub(/^.*Baseline ep /,"",n); sub(/:.*/,"",n);
  if ((n % 5) == 0) print;
  next
}
/HLT drop decomposition|FINAL SUMMARY|FINAL TEST METRICS|AUC|FPR|Saved outputs|Run completed|Traceback|RuntimeError|Error:|^=+/ { print }
'

echo "Full raw log: ${RAW_LOG}"
