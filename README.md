# CS269-Project-Deformable-Models

Wildfire segmentation on the **FLAME-3** dataset (Sycan Marsh CV subset),
comparing classical deformable models against deep methods.

**The task.** Ground truth is where the fire actually is — the ≥150 °C region of
the paired thermal frame. In the RGB the fire is usually **not visible as red
flame**: it is obscured by smoke or otherwise hard to see, and measurably the
in-fire pixels are *no redder than the background* (see the color floor below).
So this is not "segment the orange flames" — it is "recover the location of a
largely hidden hot region from RGB," with thermal used only to define the label.
That is what makes a trivial color rule fail and a learned method potentially
worthwhile.

The classical methods are run on the RGB frames and scored against the
thermal-thresholded GT:

| Method | Family | Implementation |
|---|---|---|
| **Kass Snakes** (Kass et al. 1988) | parametric, single closed contour | `skimage.segmentation.active_contour` |
| **Geodesic Active Contours** (Caselles et al. 1997) | level set, boundary-driven, multi-component | `skimage.segmentation.morphological_geodesic_active_contour` |

## Data layout

The FLAME-3 subset is copied locally under `data/` (gitignored). Only the two
directories the pipeline reads are kept:

```
data/FLAME3/Fire/
├── RGB/Corrected FOV/   00001.JPG ... 00622.JPG   (aligned RGB input)
└── Thermal/Celsius TIFF/ 00001.TIFF ... 00622.TIFF (radiometric, degrees C)
```

The ground-truth mask is **not** stored — it is regenerated each run by
thresholding the Celsius TIFF (`>= 150 °C` by default, plus 3×3 morphological
cleanup and a 20-px minimum-blob filter). This is the literal
"temperature-thresholded infrared frame" used as the target.

## Setup

Requires Python with the scientific stack (`numpy`, `opencv-python`,
`scikit-image`, `tifffile`):

```bash
pip install -r requirements.txt
```

On this machine the working interpreter is the miniconda base env
(`~/miniconda3/python.exe`, Python 3.13) — it already has all four packages.

## Running

```bash
# Both methods on the first 30 frames (quick check):
python run_snakes.py --method both --limit 30

# Kass only, every frame, custom thermal threshold:
python run_snakes.py --method kass --threshold-c 200

# Specific frames:
python run_snakes.py --method gac --frames 00536 00540 00589
```

Key flags (`python run_snakes.py --help` for all):

- `--method {kass,gac,both}`
- `--energy-mode {rg,thermal}` — image feature driving the contour
  (`rg` = red-minus-green opponency, the default fire cue)
- `--threshold-c` — thermal threshold in °C for the GT mask (default 150)
- `--limit N` / `--frames ...` — subset selection

Frames whose thermal GT is empty are skipped automatically (no contour can be
initialised from an empty mask).

## Output

Per-run console summary (mean / median / std IoU and Dice, median latency) plus
a per-frame CSV at `results/<method>_per_frame.csv` with columns:
`frame, iou, dice, gt_px, pred_px, latency_s`.

Both methods are **oracle-initialised**: the contour starts from the GT mask
scaled outward 15% (Kass) or the GT dilated to ~1.32× area (GAC), matching the
project proposal. They measure how well each method *refines* a known-location
prior, not detection from scratch.

## Deep baselines (PyTorch)

Three learned methods take **RGB only** at inference and are trained against the
same thermal-thresholded masks. They need `torch` + `torchvision`
(`pip install -r requirements.txt`); CPU works but is slow, no GPU required.

| Method | Output | Notes |
|---|---|---|
| **U-Net** | per-pixel mask | compact from-scratch U-Net; control for "is a contour head needed?" |
| **DALS** | level set | U-Net trunk + differentiable Chan-Vese evolution (Hatamizadeh 2019, simplified) |
| **Deep Snake** | polygon(s) | circular-conv contour deformation (Peng 2020). **Not** the official repo: the CenterNet detector + deformable-conv ops are replaced by a coarse-seg head whose connected components seed the initial contours. |

```bash
python run_deep.py --method unet       --mode train --epochs 50
python run_deep.py --method unet       --mode eval
python run_deep.py --method dals       --mode train
python run_deep.py --method deep_snake --mode train --batch-size 4
# quick check on a handful of frames:
python run_deep.py --method unet --mode train --epochs 2 --limit 16
```

Checkpoints (best val IoU) → `models/<method>.pt` (gitignored); test scores →
`results/<method>_per_frame.csv`. Train/val/test is a contiguous 70/15/15 split
(`flame/splits.py`) to avoid temporal leakage between near-duplicate frames.

> These deep methods are written but **not yet run/verified** here — there is no
> PyTorch or GPU in this environment, so they were delivered as code only and
> pass a syntax check but not an execution test. Train them where torch is
> installed.

## Code layout

```
flame/
├── data.py           # frame loading + thermal-threshold GT generation
├── contour_utils.py  # fire-energy features + polygon/level-set geometry
├── kass.py           # Kass Snakes (KassConfig, run_kass)
├── gac.py            # Geodesic Active Contours (GACConfig, run_gac)
├── metrics.py        # IoU + Dice
├── splits.py         # contiguous 70/15/15 train/val/test split
└── deep/
    ├── dataset.py    # torch datasets (FlameDataset, SnakeDataset)
    ├── losses.py     # BCE+Dice, Dice, cyclic contour loss
    ├── unet.py       # U-Net
    ├── dals.py       # DALS (level-set head)
    └── deep_snake.py # Deep Snake-style contour deformation
run_snakes.py         # CLI: classical Kass/GAC, score, write CSV
run_deep.py           # CLI: train/eval the deep baselines
```
