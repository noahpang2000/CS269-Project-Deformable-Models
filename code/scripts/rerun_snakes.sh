#!/usr/bin/env bash
# Re-run the deep-snake experiments with the current (paper-faithful)
# implementations, recording ADDITIVELY into a timestamped run directory so the
# committed results/*.csv and RESULTS.md are never overwritten.
#
# Scope (verified): only the snake methods changed since the recorded-results
# commit a34201c. unet/dals/classical code is byte-identical, so they are NOT
# re-run here (their numbers stand). We re-run:
#   - deep_snake_simple : the OLD detector-free snake (was "deep_snake", IoU 0.032),
#                         reproduced on the current code/splits.
#   - deep_snake_paper  : the NEW snake with a separately-trained CenterNet detector.
#
# Pipeline for deep_snake_paper (3 stages, per the paper's decoupled design):
#   1. build COCO fire-box dataset from thermal masks  (create_coco)
#   2. train the CenterNet box detector                (mmdet train.py)
#   3. train the snake on GT-mask boxes, then eval via the trained detector
#
# Usage (from repo root, inside the flame-snake conda env):
#   conda activate flame-snake
#   bash scripts/rerun_snakes.sh [DATASET] [EPOCHS] [DET_EPOCHS]
#     DATASET    : flame3 (default) or flame1
#     EPOCHS     : snake epochs (default 50, matches RESULTS.md)
#     DET_EPOCHS : CenterNet detector epochs (default 50)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${1:-flame3}"
EPOCHS="${2:-50}"
DET_EPOCHS="${3:-50}"

# Per-dataset detector config (centernet_flame.py = flame3, centernet_flame1.py = flame1).
if [[ "$DATASET" == "flame3" ]]; then
  DET_CONFIG="flame/deep/centernet_flame.py"
elif [[ "$DATASET" == "flame1" ]]; then
  DET_CONFIG="flame/deep/centernet_flame1.py"
else
  echo "ERROR: unknown dataset '$DATASET' (expected flame3 or flame1)" >&2; exit 1
fi

# Stable, sortable run id. Passed in by the caller-friendly default below.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="results/rerun_${DATASET}_${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
DET_WORKDIR="${RUN_DIR}/centernet"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$DET_WORKDIR"

echo "=== Rerun ${DATASET} ${RUN_ID} | snake epochs=${EPOCHS} detector epochs=${DET_EPOCHS} ==="
echo "Recording into ${RUN_DIR} (originals in results/ untouched)"

run() { echo "+ $*" | tee -a "${LOG_DIR}/commands.log"; "$@"; }

# run_deep.py prefixes FLAME-1 outputs with 'flame1_' (flame3 stays unprefixed).
PREFIX=""; [[ "$DATASET" != "flame3" ]] && PREFIX="${DATASET}_"

# Snapshot the exact code state for provenance.
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null || true
git status --short > "${RUN_DIR}/git_dirty.txt" 2>/dev/null || true

# Helper: run_deep writes results/<prefix><method>_per_frame.csv; move into RUN_DIR.
collect_csv() {
  local method="$1"
  if [[ -f "results/${PREFIX}${method}_per_frame.csv" ]]; then
    mv "results/${PREFIX}${method}_per_frame.csv" "${RUN_DIR}/${PREFIX}${method}_per_frame.csv"
  fi
}

# ---------------------------------------------------------------------------
# Method 1: deep_snake_simple  (old detector-free approach, current code)
# ---------------------------------------------------------------------------
echo ">>> [1/2] deep_snake_simple : train ${EPOCHS}ep + eval"
run python run_deep.py --method deep_snake_simple --dataset "$DATASET" --mode train --epochs "$EPOCHS" \
    2>&1 | tee "${LOG_DIR}/deep_snake_simple_train.log"
run python run_deep.py --method deep_snake_simple --dataset "$DATASET" --mode eval \
    2>&1 | tee "${LOG_DIR}/deep_snake_simple_eval.log"
collect_csv deep_snake_simple

# ---------------------------------------------------------------------------
# Method 2: deep_snake_paper  (CenterNet detector + snake)
# ---------------------------------------------------------------------------
echo ">>> [2/2] deep_snake_paper : COCO + detector(${DET_EPOCHS}ep) + snake(${EPOCHS}ep) + eval"

# Stage 1: COCO fire-box dataset (idempotent; symlinks images).
run python -m flame.deep.create_coco --dataset "$DATASET" 2>&1 | tee "${LOG_DIR}/create_coco.log"

# Stage 2: train CenterNet detector (per-dataset config, LR auto-scaled for batch 4).
run python flame/mmdetection/tools/train.py "$DET_CONFIG" \
    --work-dir "$DET_WORKDIR" \
    --cfg-options train_cfg.max_epochs="$DET_EPOCHS" \
    2>&1 | tee "${LOG_DIR}/centernet_train.log"

# Locate the final detector checkpoint mmdet wrote.
DET_CKPT="$(ls -t "${DET_WORKDIR}"/epoch_*.pth 2>/dev/null | head -1 || true)"
if [[ -z "$DET_CKPT" ]]; then
  echo "ERROR: no CenterNet checkpoint produced in ${DET_WORKDIR}" >&2
  exit 1
fi
echo "detector checkpoint: ${DET_CKPT}" | tee -a "${LOG_DIR}/commands.log"

# Stage 3: train the snake (boxes from GT mask), then system-eval with the detector.
run python run_deep.py --method deep_snake_paper --dataset "$DATASET" --mode train --epochs "$EPOCHS" \
    2>&1 | tee "${LOG_DIR}/deep_snake_paper_train.log"
run python run_deep.py --method deep_snake_paper --dataset "$DATASET" --mode eval \
    --mmdet-config "$DET_CONFIG" \
    --mmdet-checkpoint "$DET_CKPT" \
    2>&1 | tee "${LOG_DIR}/deep_snake_paper_eval.log"
collect_csv deep_snake_paper

# ---------------------------------------------------------------------------
# Summary: mean IoU/Dice per method, written beside the per-frame CSVs.
# ---------------------------------------------------------------------------
python - "$RUN_DIR" "$PREFIX" "$DATASET" <<'PY' 2>&1 | tee "${RUN_DIR}/summary.md"
import csv, sys, datetime
from pathlib import Path

run_dir, prefix, dataset = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
methods = ["deep_snake_simple", "deep_snake_paper"]
print(f"# Rerun summary — {run_dir.name}  (dataset: {dataset})\n")
print(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
print("| Method | n | IoU | Dice |")
print("|---|---:|---:|---:|")
for m in methods:
    p = run_dir / f"{prefix}{m}_per_frame.csv"
    if not p.exists():
        print(f"| {m} | — | (no CSV) | — |"); continue
    rows = list(csv.DictReader(p.open()))
    n = len(rows)
    iou = sum(float(r["iou"]) for r in rows) / n if n else 0.0
    dice = sum(float(r["dice"]) for r in rows) / n if n else 0.0
    print(f"| {m} | {n} | {iou:.4f} | {dice:.4f} |")
if dataset == "flame3":
    print("\nPrior recorded baseline (RESULTS.md): old `deep_snake` IoU 0.032, Dice 0.032.")
PY

echo "=== Done. Results in ${RUN_DIR}/ ==="
echo "Per-frame CSVs, logs, detector checkpoint, and summary.md are all under that dir."
