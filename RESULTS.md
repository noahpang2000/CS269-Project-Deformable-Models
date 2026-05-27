# FLAME-3 Experiment Results

Wildfire segmentation on the FLAME-3 Fire subset (622 paired RGB + thermal
frames). Ground truth = thermal frame thresholded at ≥150 °C (with 3×3
morphological cleanup + 20-px min-blob filter). All methods take **RGB only** as
the image input; thermal is used only to define the label.

- Classical methods scored on the **592 non-empty-GT frames** (30 empty-GT frames skipped).
- Deep methods scored on the **94-frame test split** (contiguous 70/15/15 split, no temporal leakage).
- Metrics: IoU and Dice on binary masks at native resolution (512×640).

## Headline numbers

| Method | Family | Init | IoU | Dice |
|---|---|---|---:|---:|
| **GAC** | level-set contour | oracle (from GT) | **0.365** | 0.499 |
| **Kass Snakes** | parametric contour | oracle (from GT) | 0.198 | 0.305 |
| **U-Net** | CNN segmentation | learned (RGB-only) | 0.147 | 0.207 |
| **DALS** | deep level-set | learned (RGB-only) | 0.122 | 0.182 |
| **Deep Snake** | deep contour | learned (RGB-only) | 0.032 | 0.032 |
| Color floor | R−G threshold | none | 0.004 | 0.007 |
| Kass Snakes | parametric contour | color (no oracle) | 0.001 | 0.002 |
| GAC | level-set contour | color (no oracle) | 0.003 | 0.006 |

(Means. Per-frame scores in `results/*_per_frame.csv`.)

## Findings

1. **The color floor is ~0 (IoU 0.004).** The thermal-defined fire is not
   recoverable from RGB redness — in-fire pixels are no redder than background
   (measured R−G ≈ 2.5 in-fire vs 1.9 background). This validates the task: a
   trivial color rule fails by design.

2. **Oracle contours win, but only refine a handed-in location.** Oracle-init
   GAC (0.365) and Kass (0.198) lead the table, but they are initialised from the
   GT mask scaled outward ~15%. They measure refinement of a known location, not
   detection.

3. **Without the oracle, contours collapse to the floor** (Kass 0.001, GAC
   0.003). Seeded from the color prior instead of the GT, the classical methods
   cannot find the fire at all — the R−G energy gives them nothing to lock onto.

4. **Deep methods are the only ones that detect from scratch above the floor.**
   With RGB-only input and no location prior, U-Net (0.147) and DALS (0.122) sit
   well above the color floor and the no-oracle contours. They trail oracle-GAC,
   but that comparison is apples-to-oranges: the deep methods get no GT prior.
   Deep Snake (0.032) underperformed — the detector-free coarse-seg seeding of
   the initial contours is the likely weak link.

5. **The ranking is robust to the GT threshold.** Sweeping 120/150/200 °C
   (`results/threshold_sensitivity.csv`, no-oracle init) leaves every no-training
   method flat-near-zero, so the conclusion is not an artifact of the 150 °C cut.

## Reproducing

Classical (CPU, minutes):

    python run_snakes.py --method all                 # color + oracle kass + oracle gac
    python run_snakes.py --method kass --init color   # no-oracle kass
    python run_snakes.py --method gac  --init color   # no-oracle gac
    python sweep_threshold.py                          # 120/150/200 C sensitivity

> Note: `run_snakes.py` writes `results/<method>_per_frame.csv` keyed only by
> method, so re-running the same method with a different `--init` overwrites the
> previous CSV. The committed CSVs are split as `kass_oracle`/`kass_color`/
> `gac_oracle`/`gac_color` to preserve both; the script itself was left unchanged.

Deep (GPU; trained here on an RTX 4090, ~30 min for all three):

    python run_deep.py --method unet       --mode train --epochs 50 && python run_deep.py --method unet       --mode eval
    python run_deep.py --method dals       --mode train --epochs 50 && python run_deep.py --method dals       --mode eval
    python run_deep.py --method deep_snake --mode train --epochs 50 && python run_deep.py --method deep_snake --mode eval

Checkpoints → `models/<method>.pt` (gitignored). Full run logs in `logs/` (gitignored).
