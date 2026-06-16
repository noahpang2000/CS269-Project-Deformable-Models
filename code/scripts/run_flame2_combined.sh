#!/usr/bin/env bash
# Train the headline methods on FLAME-2 and the pooled combined (FLAME 1+2+3)
# sets under the flame-snake env (torch 2.1 + mmdet 3.3). Evaluation + the metric
# tables (IoU/Dice/BF@2px incl. classical and Deep Snake paper) are produced
# afterwards by scripts/compute_flame2_metrics.py, so this driver only trains
# (+ a quick deep eval) and trains the CenterNet detector.
#
# Speed: DataLoader uses 6 workers (parallel JPG decode -- the old num_workers=0
# left the GPU ~16% utilised), train/val thinned by stride 2 (adjacent video
# frames are near-duplicates; the TEST split stays full), 30 epochs, CenterNet 10.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/flame-snake/bin/python
mkdir -p logs results work_dirs
EPOCHS=30
CN_EPOCHS=10
STRIDE=2
NW=6
SEG="unet dals deep_snake_simple"

train_eval () {  # <dataset> <method>
  echo "===== [$1] train $2 ====="
  $PY run_deep.py --method "$2" --dataset "$1" --mode train --epochs "$EPOCHS" \
      --train-stride "$STRIDE" --num-workers "$NW"
  echo "===== [$1] eval  $2 ====="
  $PY run_deep.py --method "$2" --dataset "$1" --mode eval
}

############### FLAME-2 ###############
echo "########## DATASET: flame2 ##########"
for M in $SEG; do train_eval flame2 "$M"; done

echo "===== [flame2] Deep Snake (paper): CenterNet -> snake head -> eval ====="
$PY -m flame.deep.create_coco --dataset flame2
$PY flame/mmdetection/tools/train.py flame/deep/centernet_flame2.py \
    --work-dir work_dirs/centernet_flame2 \
    --cfg-options train_cfg.max_epochs=$CN_EPOCHS
CN_CKPT=$(ls -t work_dirs/centernet_flame2/epoch_*.pth | head -1)
echo "CenterNet checkpoint: $CN_CKPT"
$PY run_deep.py --method deep_snake_paper --dataset flame2 --mode train --epochs "$EPOCHS" \
    --train-stride "$STRIDE" --num-workers "$NW"
echo "$CN_CKPT" > results/flame2_centernet_ckpt.txt
$PY run_deep.py --method deep_snake_paper --dataset flame2 --mode eval \
    --mmdet-config flame/deep/centernet_flame2.py --mmdet-checkpoint "$CN_CKPT" \
    --conf-threshold 0.05

############### COMBINED (FLAME 1+2+3) ###############
echo "########## DATASET: combined ##########"
for M in $SEG; do train_eval combined "$M"; done

echo "ALL DONE"
