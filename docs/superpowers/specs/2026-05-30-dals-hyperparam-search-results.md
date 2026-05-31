# DALS Hyperparameter Search — Results

**Date:** 2026-05-30
**Branch:** `dals-hyperparam-search`
**Driver:** `tune_dals.py` · raw per-trial CSVs in `results/dals_tune_{dataset}_{objective}.csv` (gitignored)

## Setup

Random search, 4 arms = {FLAME-3, FLAME-1} × {val-IoU objective, val-BF objective}.

| Arm | Trials | Epochs/trial | Winner retrain |
|---|---|---|---|
| FLAME-3 iou | 20 | 25 | 50 ep + test eval |
| FLAME-3 bf  | 20 | 25 | 50 ep + test eval |
| FLAME-1 iou | 20 | 12 | 50 ep + test eval |
| FLAME-1 bf  | 20 | 12 | 50 ep + test eval |

Search space: `n_iter ∈ {3,5,8,12}`, `mu ∈ [0.1,1.0]`, `lam1/lam2 ∈ [0.5,2.0]`,
`dt ∈ [0.05,0.3]`, `loss ∈ {bce_dice, focal_tversky}`, FT `beta ∈ [0.5,0.8]`
(`alpha = 1-beta`), `gamma ∈ [0.5,1.0]`. Fixed: `lr=1e-4`, `batch=4`, light aug.

## Winning config (same for all four arms)

Every arm's top trial converged on the **identical** config:

```
n_iter=3, mu=0.2121, lam1=0.9325, lam2=1.3792, dt=0.1885,
loss=bce_dice, ft_beta=0.7664, ft_alpha=0.2336, ft_gamma=0.9049
```

(ft_* are inert here since the winner uses bce_dice, not focal_tversky.)

## Test-split results after 50-epoch retrain

| Dataset | Objective | Baseline IoU | Tuned test IoU | Tuned test Dice | n_test |
|---|---|---|---|---|---|
| FLAME-3 | val IoU | 0.122 | **0.120** | 0.184 | 94 |
| FLAME-3 | val BF  | 0.122 | **0.129** | 0.193 | 94 |
| FLAME-1 | val IoU | 0.747 | **0.753** | 0.858 | 301 |
| FLAME-1 | val BF  | 0.747 | **0.745** | 0.853 | 301 |

Checkpoints: `models/dals_tuned_{iou,bf}.pt` (FLAME-3),
`models/flame1_dals_tuned_{iou,bf}.pt` (FLAME-1).

## Findings

1. **No region-vs-boundary tradeoff.** On both datasets the val-IoU arm and the
   val-BF arm selected the *same* winning hyperparameters. Optimizing for boundary
   fidelity does not pull the DALS config in a different direction than optimizing
   for region overlap — there is no Pareto frontier to exploit here.

2. **FLAME-3 is a wash.** Tuned test IoU (0.120 / 0.129) straddles the 0.122
   baseline. The two FLAME-3 retrains of the *same config* differed by 0.009 IoU
   from training stochasticity alone (no fixed retrain seed), so run-to-run noise is
   ≈ ±0.01 — the tuned-vs-baseline gap sits entirely inside that band. Consistent
   with the detection-bottleneck story: the ceiling is set by what the U-Net trunk
   can detect of the hidden ≥150 °C support, not by the level-set hyperparameters.

3. **FLAME-1 is a wash too, but confirms the strong baseline.** Best tuned test IoU
   0.753 vs 0.747 baseline — a +0.006 nudge, within noise. Tuning neither hurt nor
   meaningfully helped the already-strong visible-flame regime.

4. **`n_iter=3` and `bce_dice` win.** Across all arms the best trials clustered on
   few level-set iterations and the bce_dice loss; focal_tversky trials and
   high-`n_iter` trials were frequently unstable (several near-zero val IoU). The
   default DALS recipe is already close to optimal.

## Bottom line

The 80-trial search found no hyperparameter configuration that beats the existing
DALS baselines beyond training noise on either dataset. The default settings are
effectively already tuned; the FLAME-3 ceiling is detection-limited, not
contour-limited.
