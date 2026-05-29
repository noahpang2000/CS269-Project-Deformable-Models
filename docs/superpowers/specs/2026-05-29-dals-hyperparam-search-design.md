# DALS Hyperparameter Search — Design

**Date:** 2026-05-29
**Status:** Approved (brainstorming)

## Goal

Find DALS configurations that improve on the current baselines
(FLAME-1 IoU 0.747, FLAME-3 IoU 0.122), running **two search objectives
separately** — val IoU and val boundary F-score — to see whether optimizing for
region overlap vs. boundary fidelity selects different hyperparameters.

## Search space

**DALS structural** (currently hardcoded in `flame/deep/dals.py`; must be exposed
as CLI flags first):

| Param | Range / set | Sampling |
|---|---|---|
| `n_iter` | {3, 5, 8, 12} | categorical |
| init `mu` | [0.1, 1.0] | uniform |
| init `lam1` | [0.5, 2.0] | uniform |
| init `lam2` | [0.5, 2.0] | uniform |
| init `dt` | [0.05, 0.3] | uniform |

**Loss** (loss family already on CLI; Focal-Tversky sub-params must be exposed):

| Param | Range / set | Notes |
|---|---|---|
| `loss` | {bce_dice, focal_tversky} | categorical |
| `ft_beta` | [0.5, 0.8] | only when loss=focal_tversky; `ft_alpha = 1 - ft_beta` |
| `ft_gamma` | [0.5, 1.0] | only when loss=focal_tversky |

**Fixed at defaults** (not part of this search): `lr=1e-4`, `batch_size=4`,
`augment=light`, `size=NET_SIZE`. Keeps the search focused on the DALS-specific
story rather than generic training knobs.

## Strategy

- **Random search, 20 trials** per arm; each trial trained at **reduced epochs
  (25)** for speed.
- **4 arms:** {FLAME-1, FLAME-3} × {val IoU objective, val BF objective}.
- Every trial records **both** val IoU and val BF@2px regardless of arm, so the
  ranked tables support the region-vs-boundary comparison.
- **Seeded RNG** for reproducible config sampling.
- Sequential on a single GPU; no Optuna, no pruning, no parallel scheduling.

## Objective metrics

- **Val IoU** — matches the repo's existing checkpoint-selection protocol.
- **Val boundary F-score @ 2px** — averaged over val frames, reusing
  `flame/boundary_metrics.boundary_fscore`.

## Code changes

1. **`flame/deep/dals.py`** — `DALS.__init__` accepts `n_iter, mu, lam1, lam2, dt`
   (defaults = current values: 5, 0.2, 1.0, 1.0, 0.1). Backward-compatible.
2. **`run_deep.py`** — new flags `--n-iter --mu --lam1 --lam2 --dt --ft-alpha
   --ft-beta --ft-gamma`, threaded into `build_model`/`DALS(...)` and
   `focal_tversky(...)`. Defaults preserve current behavior.
3. **`tune_dals.py`** (new) — the search driver: sample configs, reuse
   `run_deep`'s train/eval functions, compute val IoU + val BF per trial, write a
   ranked CSV per arm, print winners.
4. **Val BF eval helper** — reuse `boundary_fscore(pred, gt, tol_px=2.0)` over the
   val split.

## Deliverables

- `results/dals_tune_{dataset}_{objective}.csv` — 4 ranked-trial files.
- **Retrain the winner of each arm** at full 50 epochs, eval on the **test split**
  (IoU + Dice + the arm's selected metric), save
  `models/{prefix}dals_tuned_{objective}.pt`.
- Summary: baseline vs. best-IoU-config vs. best-BF-config, per dataset.

## Cost / verification

- **Smoke test first:** 2 trials × 2 epochs to confirm wiring and measure
  per-trial wall-clock. Report **projected total runtime** before launching the
  full 4-arm search so the user can green-light or trim (fewer trials / smaller
  val subset). FLAME-3 trials are fast; FLAME-1 at 1024px is heavier.

## Out of scope (YAGNI)

- Optuna / Bayesian search / trial pruning.
- Parallel trial scheduling.
- Running the full boundary suite (`make_boundary_table.py`) on winners — chosen
  deliverable is test-set IoU/Dice eval, not the full boundary table.
- Tuning lr / batch / augment.
