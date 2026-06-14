#!/usr/bin/env bash
# Resume the pipeline after the FLAME-2 segmentation models (unet/dals/
# deep_snake_simple) already finished: train the CenterNet detector (with a retry
# guard for the flaky mmcv init segfault), the Deep Snake (paper) snake head, then
# the combined (FLAME 1+2+3) segmentation models.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/flame-snake/bin/python
export OMP_NUM_THREADS=4          # reduce the mmcv/OpenCV thread-race segfault risk
EPOCHS=30; CN_EPOCHS=10; STRIDE=2; NW=6

############### FLAME-2 Deep Snake (paper) ###############
echo "===== [flame2] CenterNet (retry up to 3x on segfault) ====="
$PY -m flame.deep.create_coco --dataset flame2
rm -rf work_dirs/centernet_flame2
ok=0
for attempt in 1 2 3; do
  echo "--- CenterNet attempt $attempt ---"
  $PY flame/mmdetection/tools/train.py flame/deep/centernet_flame2.py \
      --work-dir work_dirs/centernet_flame2 \
      --cfg-options train_cfg.max_epochs=$CN_EPOCHS && { ok=1; break; }
  echo "attempt $attempt failed (exit $?); retrying"; sleep 5
done
[ "$ok" = 1 ] || { echo "CenterNet FAILED after 3 attempts"; exit 1; }

CN_CKPT=$(ls -t work_dirs/centernet_flame2/epoch_*.pth | head -1)
echo "CenterNet checkpoint: $CN_CKPT"
echo "$CN_CKPT" > results/flame2_centernet_ckpt.txt

echo "===== [flame2] Deep Snake (paper) snake head ====="
$PY run_deep.py --method deep_snake_paper --dataset flame2 --mode train --epochs "$EPOCHS" \
    --train-stride "$STRIDE" --num-workers "$NW"
$PY run_deep.py --method deep_snake_paper --dataset flame2 --mode eval \
    --mmdet-config flame/deep/centernet_flame2.py --mmdet-checkpoint "$CN_CKPT" \
    --conf-threshold 0.05

############### COMBINED (FLAME 1+2+3) ###############
echo "########## DATASET: combined ##########"
for M in unet dals deep_snake_simple; do
  echo "===== [combined] train $M ====="
  $PY run_deep.py --method "$M" --dataset combined --mode train --epochs "$EPOCHS" \
      --train-stride "$STRIDE" --num-workers "$NW"
  echo "===== [combined] eval $M ====="
  $PY run_deep.py --method "$M" --dataset combined --mode eval
done

echo "ALL DONE"
