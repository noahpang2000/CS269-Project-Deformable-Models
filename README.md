# CS269 Project: Deformable Models for Wildfire Segmentation

Segmenting wildfire from drone imagery, comparing classical active-contour
(deformable) models against deep methods, across the **FLAME-1, FLAME-2, and
FLAME-3** datasets and a pooled **combined** set.

**Methods.** Classical Kass Snakes and Geodesic Active Contours (GAC); deep
methods U-Net, Deep Active Lesion Segmentation (DALS), and Deep Snake in two
variants (a detector-free "simple" variant and the paper-faithful CenterNet
`+` snake "paper" variant); plus an R$-$G color-threshold floor.

**The task differs by dataset:**

| Dataset | RGB | Ground truth | Difficulty |
|---|---|---|---|
| **FLAME-1** | close drone views, flame directly visible | hand-labeled flame masks | fire is visible; refine the boundary |
| **FLAME-2** | aerial, heavily smoke-obscured | derived from the **colorized-IR** video (no raw Celsius) | fire barely visible in RGB |
| **FLAME-3** | aerial, smoke-obscured | thermal TIFF thresholded at $\geq$150 °C | fire barely visible in RGB |
| **combined** | union of all three | each dataset's own GT | mixes visible-flame and thermal-hot labels |

## Data layout

Datasets live under `data/` (gitignored). The pipeline reads:

```
data/FLAME1/{images,Masks}/                       # RGB jpg + binary PNG mask
data/FLAME2/{images,Masks,ir}/                    # sampled from the video pairs
data/FLAME2/#1-7) All Video Pairs.zip             # source IR+RGB videos
data/FLAME3/Fire/RGB/Corrected FOV/  + Thermal/Celsius TIFF/
```

**FLAME-2** ships as seven paired RGB`+`colorized-IR videos with no per-pixel
temperature. `scripts/extract_flame2.py` samples paired frames at ~1 fps and
derives the GT from the IR palette (dense hot-speckle regions), resizing the
mask into the RGB frame under a *resize-and-pair* policy (`flame.data.flame2_fire_mask`):

```bash
python scripts/extract_flame2.py            # writes data/FLAME2/{images,Masks,ir}
```

## Setup

Two environments are used on this machine:

- **base** (`python3`, torch) — runs everything except the CenterNet detector.
- **`flame-snake`** conda env (torch 2.1 `+` mmdet 3.3 `+` mmcv 2.1) — required for
  the Deep Snake (paper) CenterNet pipeline. Run those steps with
  `~/anaconda3/envs/flame-snake/bin/python`.

```bash
pip install -r requirements.txt        # numpy, opencv, scikit-image, tifffile, torch
```

## Running

**Classical** (Kass / GAC / color floor), per-frame, scored against the GT:

```bash
python run_snakes.py --method all --dataset flame3 --init oracle
python run_snakes.py --dataset flame2 --split test --init oracle --max-side 1024
```

**Deep methods** — `--dataset {flame1,flame2,flame3,combined}`, `--mode {train,eval}`:

```bash
python run_deep.py --method unet --dataset flame2 --mode train --epochs 30 \
                   --train-stride 2 --num-workers 6
python run_deep.py --method unet --dataset flame2 --mode eval
```

**Deep Snake (paper)** needs the CenterNet detector (run in the `flame-snake` env):

```bash
python -m flame.deep.create_coco --dataset flame2
python flame/mmdetection/tools/train.py flame/deep/centernet_flame2.py \
       --work-dir work_dirs/centernet_flame2
python run_deep.py --method deep_snake_paper --dataset flame2 --mode train
python run_deep.py --method deep_snake_paper --dataset flame2 --mode eval \
       --mmdet-config flame/deep/centernet_flame2.py \
       --mmdet-checkpoint work_dirs/centernet_flame2/epoch_10.pth --conf-threshold 0.05
```

End-to-end drivers: `scripts/run_flame2_combined.sh` (train) and
`scripts/finalize_flame2.sh` (metrics + figures).

## Metrics & figures

```bash
python scripts/compute_flame2_metrics.py --dataset flame2     # IoU / Dice / BF@2px
python scripts/combined_transfer.py                           # pooled vs dataset-specific
python scripts/make_flame2_figs.py                            # galleries + bar charts
```

Checkpoints → `models/`, per-frame scores → `results/`, both gitignored.

## Code layout

```
flame/
├── data.py            # frame loading + GT generation (all datasets, incl. FLAME-2 IR)
├── splits.py          # 70/15/15 splits (per-dataset for the combined set)
├── kass.py, gac.py    # classical contours
├── baselines.py       # color-threshold floor
├── metrics.py         # IoU, Dice
├── contour_utils.py   # fire energy + polygon/level-set geometry
└── deep/
    ├── unet.py, dals.py
    ├── deep_snake_simplified.py   # detector-free Deep Snake
    ├── deep_snake.py              # CenterNet-based Deep Snake (paper)
    ├── create_coco.py, centernet_*.py   # detector training data + configs
    ├── dataset.py, losses.py
run_snakes.py          # CLI: classical methods
run_deep.py            # CLI: train/eval the deep methods
scripts/               # extraction, metrics, transfer, figures, drivers
```

> The LaTeX report (`report.tex`, `report/`) is kept out of version control.
