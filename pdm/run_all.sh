#!/usr/bin/env bash
# Run the full pipeline in order, stopping at the first failure.
# Console output from every stage is also captured to outputs/results/.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p outputs/results outputs/figures outputs/models data/processed

STAGES=(
  "01_data_understanding"
  "02_eda"
  "03_preprocessing"
  "04_feature_engineering"
  "05_model_training"
  "06_evaluation"
  "07_shap_analysis"
  "08_business_impact"
  "09_reliability"
)

for s in "${STAGES[@]}"; do
  echo ""
  echo ">>> Running ${s}.py"
  python3 "${s}.py" | tee "outputs/results/log_${s}.txt"
done

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE"
echo "  Figures : $(ls outputs/figures/*.png | wc -l) files in outputs/figures/"
echo "  Tables  : $(ls outputs/results/*.csv | wc -l) files in outputs/results/"
echo "  Models  : $(ls outputs/models/*.joblib | wc -l) files in outputs/models/"
echo "================================================================"
