#!/usr/bin/env bash
set -euo pipefail

# PrepBench sensitivity runs.
#
# Before running:
#   export PROJ_DIR="$(pwd)"
#   export PREPBENCH_DATA_DIR=/path/to/deepline_dataset   # if you patched the loader to use this
#   export OPENBLAS_NUM_THREADS=1
#   export MKL_NUM_THREADS=1
#   export OMP_NUM_THREADS=1
#   export NUMEXPR_NUM_THREADS=1
#
# If your loader still uses /path/to/deepline_dataset, either keep that path
# on the server or patch data_loader/load_deepline.py accordingly.

SEEDS="${SEEDS:-42 114 256 512 768}"
PROCESSES="${PROCESSES:-16}"
CONFIGS="${CONFIGS:-D-Rand D-S-F-A D-C-T-B}"

mkdir -p results/sensitivity

for BUDGET in 300 600; do
  echo "=== zero-penalty sensitivity: budget=${BUDGET}s ==="
  python experiments/run_matrix.py \
    --budget "${BUDGET}" \
    --seeds ${SEEDS} \
    --configs ${CONFIGS} \
    --out "results/sensitivity/zpen${BUDGET}" \
    --processes "${PROCESSES}" \
    --fitness-cost-penalty 0 \
    --fitness-complexity-penalty 0 \
    --run-tag "zero_penalty_${BUDGET}"

  echo "=== stronger-EA sensitivity: budget=${BUDGET}s ==="
  python experiments/run_matrix.py \
    --budget "${BUDGET}" \
    --seeds ${SEEDS} \
    --configs D-S-F-A D-C-T-B \
    --out "results/sensitivity/strongEA${BUDGET}" \
    --processes "${PROCESSES}" \
    --population-size 40 \
    --offspring-size 40 \
    --fitness-cost-penalty 0 \
    --fitness-complexity-penalty 0 \
    --run-tag "strongEA_zero_penalty_${BUDGET}"
done

python experiments/analyze_extras.py \
  --run-dirs \
    results/sensitivity/zpen300 \
    results/sensitivity/zpen600 \
    results/sensitivity/strongEA300 \
    results/sensitivity/strongEA600 \
  --out results/sensitivity/extra_audit \
  --seeds ${SEEDS}

python experiments/summarize_sensitivity_runs.py \
  --runs \
    zpen300=results/sensitivity/zpen300 \
    zpen600=results/sensitivity/zpen600 \
    strongEA300=results/sensitivity/strongEA300 \
    strongEA600=results/sensitivity/strongEA600 \
  --out results/sensitivity/summary

echo "Done. Main outputs:"
echo "  results/sensitivity/summary/sensitivity_summary_by_config.csv"
echo "  results/sensitivity/summary/sensitivity_paired_tests.csv"
echo "  results/sensitivity/extra_audit/random_fairness_audit.csv"
