#!/bin/bash
# Unified  workflow - single entry for training, evaluation, prediction
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
SECONDS=0

# -----------------------------
# User-tunable env vars (defaults)
# -----------------------------
if [ -z "${OUTPUT_DIR+x}" ] || [ -z "${OUTPUT_DIR}" ]; then
  RUN_INDEX=1
  while [ -e "$(printf 'Project_Output_run_%03d' "$RUN_INDEX")" ]; do
    RUN_INDEX=$((RUN_INDEX + 1))
  done
  OUTPUT_DIR="$(printf 'Project_Output_run_%03d' "$RUN_INDEX")"
  OUTPUT_DIR_AUTO=1
else
  OUTPUT_DIR_AUTO=0
fi
DATA_FILE="${DATA_FILE:-data/PhosIrDB.csv}"
TEST_DATA_FILE="${TEST_DATA_FILE:-data/ours.csv}"
VIRTUAL_FILE="${VIRTUAL_FILE:-data/ir_assemble.csv}"

TRAIN_MODELS_DEFAULT="xgboost,lightgbm,random_forest,mlp,catboost,gradient_boosting,ridge,decision_tree"
TRAIN_MODELS_FULL="adaboost,catboost,decision_tree,elastic_net,gradient_boosting,knn,lasso,lightgbm,random_forest,ridge,svr,xgboost,mlp"
TRAIN_MODELS="${TRAIN_MODELS:-$TRAIN_MODELS_DEFAULT}"
TRAIN_FULL="${TRAIN_FULL:-0}"
TRAIN_TUI="${TRAIN_TUI:-1}"
TRAIN_TUI_HOLD="${TRAIN_TUI_HOLD:-1}"
TRAIN_FOLDS="${TRAIN_FOLDS:-10}"
FORCE_TRAIN="${FORCE_TRAIN:-1}"     # 1: always train; 0: reuse existing

SKIP_VIRTUAL="${SKIP_VIRTUAL:-0}"
SKIP_SHAP="${SKIP_SHAP:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

LOG_FILE="$OUTPUT_DIR/workflow.log"
PROGRESS_FILE="$OUTPUT_DIR/progress.jsonl"
TUI_PID=""

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

progress_event() {
  local event_type="$1"
  local step="$2"
  local message="${3:-}"
  python - "$PROGRESS_FILE" "$event_type" "$step" "$message" "$SECONDS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
event_type = sys.argv[2]
step = sys.argv[3]
message = sys.argv[4]
elapsed = float(sys.argv[5])
event = {"type": event_type, "step": step, "elapsed_seconds": elapsed}
if message:
    event["message"] = message
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(event, ensure_ascii=True) + "\n")
PY
}

emit_workflow_context() {
  python - "$PROGRESS_FILE" "$DATA_FILE" "$OUTPUT_DIR" "$SECONDS" <<'PY'
import csv
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1])
data_path = Path(sys.argv[2])
output_dir = sys.argv[3]
elapsed = float(sys.argv[4])

row_count = None
if data_path.exists():
    with data_path.open(newline="", encoding="utf-8") as handle:
        row_count = max(sum(1 for _ in csv.reader(handle)) - 1, 0)

event = {
    "type": "workflow_context",
    "training_data_path": str(data_path),
    "training_data_rows": row_count,
    "output_dir": output_dir,
    "screen_operations": [
        "Ctrl+C: exit final TUI",
        "TRAIN_TUI=0 bash run.sh: disable TUI",
        "TRAIN_TUI_HOLD=0 bash run.sh: exit TUI automatically",
        "screen -S phosir bash run.sh: run workflow in screen",
        "Ctrl+A then D: detach screen",
        "screen -r phosir: resume screen",
    ],
    "elapsed_seconds": elapsed,
}
progress_path.parent.mkdir(parents=True, exist_ok=True)
with progress_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(event, ensure_ascii=True) + "\n")
PY
}

run_step() {
  local step="$1"
  local label="$2"
  shift 2
  progress_event workflow_step_started "$step" "$label"
  info "$label"
  if [ "$TRAIN_TUI" = "1" ] && [ -t 1 ]; then
    if "$@" >>"$LOG_FILE" 2>&1; then
      progress_event workflow_step_completed "$step" "$label"
      return 0
    else
      progress_event workflow_step_warning "$step" "$label failed"
      warn "$label failed"
      return 1
    fi
  else
    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
      progress_event workflow_step_completed "$step" "$label"
      return 0
    else
      progress_event workflow_step_warning "$step" "$label failed"
      warn "$label failed"
      return 1
    fi
  fi
}

start_tui() {
  if [ "$TRAIN_TUI" = "1" ] && [ -t 1 ]; then
    : > "$PROGRESS_FILE"
    python "$ROOT_DIR/scripts/cli/workflow_tui.py" \
      --events "$PROGRESS_FILE" \
      --models "$TRAIN_MODELS" \
      --targets 'Max_wavelength(nm),PLQY' &
    TUI_PID=$!
    sleep 0.2
  fi
}

stop_tui() {
  if [ -n "$TUI_PID" ]; then
    kill "$TUI_PID" 2>/dev/null || true
    wait "$TUI_PID" 2>/dev/null || true
  fi
}

hold_tui_on_finish() {
  if [ -n "$TUI_PID" ] && [ "$TRAIN_TUI_HOLD" = "1" ] && [ -t 1 ]; then
    printf "TUI final screen is being kept. Press Ctrl+C to exit.\n"
    wait "$TUI_PID" 2>/dev/null || true
    TUI_PID=""
  fi
}

trap stop_tui EXIT

if [ "$OUTPUT_DIR_AUTO" -ne 1 ] && [ -e "$OUTPUT_DIR" ]; then
  existing_artifacts=()
  [ -d "$OUTPUT_DIR/all_models/automl_train" ] && existing_artifacts+=("all_models/automl_train")
  [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ] && existing_artifacts+=("virtual_predictions_all.csv")
  [ -d "$OUTPUT_DIR/shap_analysis" ] && existing_artifacts+=("shap_analysis")
  if [ "${#existing_artifacts[@]}" -gt 0 ]; then
    error "Output directory '$OUTPUT_DIR' already contains workflow artifacts: ${existing_artifacts[*]}. Use a fresh OUTPUT_DIR."
  fi
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Unified workflow"
echo "Elapsed: 0s"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

if [ "$TRAIN_FULL" = "1" ]; then
  TRAIN_MODELS="$TRAIN_MODELS_FULL"
fi

start_tui

emit_workflow_context

# -----------------------------
# Step 0: Data checks
# -----------------------------
progress_event workflow_step_started data_checks "Data checks"
if [ ! -f "$DATA_FILE" ]; then
  progress_event workflow_step_failed data_checks "Training data not found: $DATA_FILE"
  error "Training data not found: $DATA_FILE"
fi

if [ -f "$TEST_DATA_FILE" ]; then
  info "Test data found: $TEST_DATA_FILE"
else
  warn "Test data not found: $TEST_DATA_FILE (step 7 will be skipped)"
fi

if [ ! -f "$VIRTUAL_FILE" ]; then
  warn "Virtual DB not found ($VIRTUAL_FILE), generating..."
  python "$ROOT_DIR/scripts/generate_virtual_database.py" 2>&1 | tee -a "$LOG_FILE"
fi
progress_event workflow_step_completed data_checks "Data checks"

# -----------------------------
# Step 1: Virtual DB stats (optional if file exists)
# -----------------------------
if [ -f "$VIRTUAL_FILE" ]; then
  run_step virtual_db_stats "Virtual DB stats" \
    python "$ROOT_DIR/scripts/analyze_combinations.py" \
      --data "$DATA_FILE" \
      --virtual "$VIRTUAL_FILE" || true
else
  progress_event workflow_step_warning virtual_db_stats "Virtual DB not found; skipped stats"
fi

# -----------------------------
# Step 2: Train models
# -----------------------------
TRAIN_DIR="$OUTPUT_DIR/all_models/automl_train"
if [ -d "$TRAIN_DIR" ] && [ "$FORCE_TRAIN" = "0" ]; then
  info "Found existing training runs at $TRAIN_DIR (FORCE_TRAIN=0), skipping training."
  progress_event workflow_step_completed train_models "Existing training runs reused"
else
  progress_event workflow_step_started train_models "Train models"
  info "Training models: $TRAIN_MODELS"
  AUTOML_ARGS=(
    train
    data="$DATA_FILE" \
    test_data="$TEST_DATA_FILE" \
    project="$OUTPUT_DIR" \
    name=all_models \
    models="$TRAIN_MODELS" \
    training.n_folds="$TRAIN_FOLDS" \
    'training.metrics=["r2","rmse","mae"]' \
    training.save_final_model=true \
    training.save_fold_models=false \
    training.save_feature_importance=true \
    training.verbose=2 \
    feature.feature_type=combined \
    feature.morgan_bits=1024 \
    feature.morgan_radius=2 \
    feature.combination_method=mean \
    feature.use_cache=true \
    'data.smiles_columns=["L1","L2","L3"]' \
    'data.target_columns=["Max_wavelength(nm)","PLQY"]' \
    data.multi_target_strategy=intersection \
    data.nan_handling=skip \
    data.train_ratio=1.0 \
    data.val_ratio=0.0 \
    data.test_ratio=0.0 \
    comparison.enable=true \
    'comparison.formats=["markdown","html","latex","csv"]' \
    comparison.highlight_best=true \
    comparison.include_std=true \
    comparison.decimal_places.r2=4 \
    comparison.decimal_places.rmse=4 \
    comparison.decimal_places.mae=4 \
    export.enable=true \
    'export.formats=["json","csv","excel"]' \
    export.include_predictions=true \
    export.include_feature_importance=true \
    export.include_cv_details=true \
    export.generate_plots=true \
    export.generate_report=true
  )
  if [ "$TRAIN_TUI" = "1" ] && [ -t 1 ]; then
    if ! TRAIN_LOG_FILE="$LOG_FILE" TRAIN_PROGRESS_FILE="$PROGRESS_FILE" TRAIN_TUI=0 PYTHONUNBUFFERED=1 python -u "$ROOT_DIR/scripts/cli/automl.py" "${AUTOML_ARGS[@]}" >>"$LOG_FILE" 2>&1; then
      progress_event workflow_step_failed train_models "Train models failed"
      error "Train models failed"
    fi
  else
    if ! TRAIN_TUI=0 PYTHONUNBUFFERED=1 python -u "$ROOT_DIR/scripts/cli/automl.py" "${AUTOML_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
      progress_event workflow_step_failed train_models "Train models failed"
      error "Train models failed"
    fi
  fi
  progress_event workflow_step_completed train_models "Train models"
fi

# -----------------------------
# Step 3: Comparison table
# -----------------------------
run_step model_comparison "Model comparison" \
  python "$ROOT_DIR/scripts/generate_model_comparison.py" --output-dir "$OUTPUT_DIR" || true

# -----------------------------
# Step 3.5: Best model plots
# -----------------------------
run_step best_model_plots "Best model plots" \
  python "$ROOT_DIR/scripts/plot_best_model_performance.py" --project "$OUTPUT_DIR" || true

# -----------------------------
# Step 4: Virtual predictions
# -----------------------------
if [ "$SKIP_VIRTUAL" = "1" ]; then
  info "SKIP_VIRTUAL=1 set; skipping virtual DB prediction."
  progress_event workflow_step_warning virtual_predictions "SKIP_VIRTUAL=1; skipped"
else
  if [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ]; then
    info "Virtual predictions already exist, skipping."
    progress_event workflow_step_completed virtual_predictions "Virtual predictions already exist"
  else
    run_step virtual_predictions "Virtual predictions" \
      python "$ROOT_DIR/scripts/predict_all_combinations.py" \
        --project "$OUTPUT_DIR" \
        --input "$VIRTUAL_FILE" \
        --output "$OUTPUT_DIR/virtual_predictions_all.csv" || true
  fi
fi

# -----------------------------
# Step 4.5: Virtual prediction plots
# -----------------------------
if [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ]; then
  run_step virtual_plots "Virtual prediction plots" \
    python "$ROOT_DIR/scripts/plot_virtual_predictions.py" \
      --input "$OUTPUT_DIR/virtual_predictions_all.csv" \
      --output "$OUTPUT_DIR/figures" || true
else
  progress_event workflow_step_warning virtual_plots "No virtual predictions; skipped plots"
fi

# -----------------------------
# Step 5: Stratified performance
# -----------------------------
run_step stratified_analysis "Stratified analysis" \
  python "$ROOT_DIR/scripts/generate_stratified_analysis.py" \
    --project "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/figures" || true

# -----------------------------
# Step 6: Publication-oriented figures
# -----------------------------
if [ "$SKIP_FIGURES" = "1" ]; then
  info "SKIP_FIGURES=1 set; skipping figure generation."
  progress_event workflow_step_warning publication_figures "SKIP_FIGURES=1; skipped"
else
  run_step publication_figures "Publication figures" \
    python "$ROOT_DIR/scripts/generate_publication_figures.py" \
      --project "$OUTPUT_DIR" \
      --data "$DATA_FILE" \
      --output "$OUTPUT_DIR/figures" || true
fi

# -----------------------------
# Step 7: Predict test data
# -----------------------------
if [ -f "$TEST_DATA_FILE" ]; then
  progress_event workflow_step_started test_data_prediction "Test data prediction"
  mkdir -p "$OUTPUT_DIR/test_predictions"
  IFS=',' read -r -a TEST_PREDICT_MODELS <<< "$TRAIN_MODELS"
  test_prediction_failures=0
  for model in "${TEST_PREDICT_MODELS[@]}"; do
    model="$(echo "$model" | xargs)"
    [ -z "$model" ] && continue
    output_file="$OUTPUT_DIR/test_predictions/${model}_ours_predictions.csv"
    info "Predicting ours.csv with full-data model: $model"
    if [ "$TRAIN_TUI" = "1" ] && [ -t 1 ]; then
      if ! TEST_PREDICTION_PROGRESS_FILE="$PROGRESS_FILE" PYTHONUNBUFFERED=1 python -u "$ROOT_DIR/scripts/predict_test_data.py" \
          --project "$OUTPUT_DIR" \
          --input "$TEST_DATA_FILE" \
          --output "$output_file" \
          --model "$model" >>"$LOG_FILE" 2>&1; then
        warn "Test data prediction failed for $model"
        test_prediction_failures=$((test_prediction_failures + 1))
      fi
    else
      if ! TEST_PREDICTION_PROGRESS_FILE="$PROGRESS_FILE" PYTHONUNBUFFERED=1 python -u "$ROOT_DIR/scripts/predict_test_data.py" \
          --project "$OUTPUT_DIR" \
          --input "$TEST_DATA_FILE" \
          --output "$output_file" \
          --model "$model" 2>&1 | tee -a "$LOG_FILE"; then
        warn "Test data prediction failed for $model"
        test_prediction_failures=$((test_prediction_failures + 1))
      fi
    fi
  done
  if [ "$test_prediction_failures" -gt 0 ]; then
    progress_event workflow_step_warning test_data_prediction "$test_prediction_failures model prediction(s) failed"
  else
    progress_event workflow_step_completed test_data_prediction "Test data prediction"
  fi
else
  progress_event workflow_step_warning test_data_prediction "Test data not found; skipped"
fi

# -----------------------------
# Step 8: SHAP analysis
# -----------------------------
if [ "$SKIP_SHAP" = "1" ]; then
  info "SKIP_SHAP=1 set; skipping SHAP."
  progress_event workflow_step_warning shap_analysis "SKIP_SHAP=1; skipped"
else
  if [ -f "$OUTPUT_DIR/shap_analysis/shap_report.html" ]; then
    info "SHAP report exists; skipping."
    progress_event workflow_step_completed shap_analysis "SHAP report already exists"
  else
    run_step shap_analysis "SHAP analysis" \
      python "$ROOT_DIR/scripts/cli/analyze_shap.py" "$OUTPUT_DIR" \
        --models xgboost lightgbm catboost \
        --sample-size 100 || true
  fi
fi

# -----------------------------
# Step 9: Final report summary
# -----------------------------
progress_event workflow_step_started final_summary "Final summary"
python - "$OUTPUT_DIR" "$SECONDS" <<'PY' >>"$LOG_FILE" 2>&1
import json
import pandas as pd
from pathlib import Path
import sys
out_dir = Path(sys.argv[1])
elapsed = float(sys.argv[2])
report = {
    "output_dir": str(out_dir),
    "workflow_elapsed_seconds": elapsed,
    "files": [],
    "summary": {}
}

def exists(name):
    p = out_dir / name
    if p.exists():
        report["files"].append(str(p))
        return True
    return False

report["summary"]["training_dir"] = str(out_dir / "all_models" / "automl_train")
report["summary"]["comparison"] = exists("model_comparison_detailed.csv")
report["summary"]["virtual_predictions"] = exists("virtual_predictions_all.csv")
report["summary"]["final_report"] = exists("final_report.json")

if (out_dir / "virtual_predictions_all.csv").exists():
    try:
        df = pd.read_csv(out_dir / "virtual_predictions_all.csv")
        report["summary"]["virtual_rows"] = len(df)
        if "Predicted_PLQY" in df.columns:
            report["summary"]["plqy_ge_0.9"] = int((df["Predicted_PLQY"] >= 0.9).sum())
            report["summary"]["plqy_ge_0.8"] = int((df["Predicted_PLQY"] >= 0.8).sum())
    except Exception as e:
        report["summary"]["virtual_error"] = str(e)

with open(out_dir / "workflow_summary.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=True)

print("Workflow summary saved:", out_dir / "workflow_summary.json")
PY
progress_event workflow_step_completed final_summary "Final summary"
progress_event workflow_finished final_summary "Workflow completed"

DURATION=$SECONDS
printf "\n==========================================\n"
printf "Workflow completed in %02dh:%02dm:%02ds\n" $((DURATION/3600)) $(((DURATION%3600)/60)) $((DURATION%60))
printf "Output directory: %s\n" "$OUTPUT_DIR"
printf "Log file: %s\n" "$LOG_FILE"
printf "==========================================\n\n"

hold_tui_on_finish
