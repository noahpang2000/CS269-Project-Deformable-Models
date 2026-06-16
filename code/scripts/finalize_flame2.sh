#!/usr/bin/env bash
# Compute the metric tables, the pooled-vs-specific transfer analysis, and the
# figures for the FLAME-2 / combined reruns. mmdet-free (Deep Snake paper is
# scored at oracle here), so it runs cleanly.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/flame-snake/bin/python
export OMP_NUM_THREADS=4

echo "########## FLAME-2 metrics (full classical) ##########"
$PY scripts/compute_flame2_metrics.py --dataset flame2

echo "########## combined metrics (color floor only; contours are dataset-agnostic) ##########"
$PY scripts/compute_flame2_metrics.py --dataset combined --no-contour-classical

echo "########## combined transfer (pooled vs dataset-specific) ##########"
$PY scripts/combined_transfer.py

echo "########## figures ##########"
$PY scripts/make_flame2_figs.py

echo "FINALIZE DONE"
