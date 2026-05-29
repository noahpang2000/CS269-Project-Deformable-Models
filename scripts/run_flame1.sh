#!/bin/bash
# FLAME-1 experiment suite: resumes after color/kass/gac classical (already done).
# Run inside a detached screen so it survives the parent shell.
set -e
cd /home/brody/Cs269-Project-Deformable-Models
export PYTHONUNBUFFERED=1
echo "=== START $(date) ==="
echo
echo ">>> Deep baselines (light aug, 50ep each) -- color/kass/gac already done"
echo "    [removing any partial flame1_unet.pt from earlier killed run]"
rm -f models/flame1_unet.pt
python3 -u run_deep.py --method unet       --mode train --dataset flame1 --epochs 50
python3 -u run_deep.py --method unet       --mode eval  --dataset flame1
python3 -u run_deep.py --method dals       --mode train --dataset flame1 --epochs 50
python3 -u run_deep.py --method dals       --mode eval  --dataset flame1
python3 -u run_deep.py --method deep_snake --mode train --dataset flame1 --epochs 50
python3 -u run_deep.py --method deep_snake --mode eval  --dataset flame1
echo
# Augmented (medium aug, 120ep) runs removed: baselines already at 0.77 IoU on
# FLAME-1, augmentation expected to add little when not data-starved, and the
# ~6h-per-run cost is not worth it. Add back if needed:
#   python3 -u run_deep.py --method unet --mode train --dataset flame1 --epochs 120 --augment medium --tag _aug
#   python3 -u run_deep.py --method unet --mode eval  --dataset flame1 --tag _aug
#   python3 -u run_deep.py --method dals --mode train --dataset flame1 --epochs 120 --augment medium --tag _aug
#   python3 -u run_deep.py --method dals --mode eval  --dataset flame1 --tag _aug
echo "=== END $(date) ==="
