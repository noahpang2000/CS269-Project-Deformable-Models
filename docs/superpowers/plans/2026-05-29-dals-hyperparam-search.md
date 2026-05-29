# DALS Hyperparameter Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a random-search hyperparameter optimizer for DALS that tunes the Chan-Vese structural params and loss knobs against two objectives (val IoU, val boundary F-score) on both FLAME datasets, then retrains and test-evaluates the winners.

**Architecture:** Expose DALS's currently-hardcoded structural params and the Focal-Tversky sub-params as `run_deep.py` CLI flags (backward-compatible defaults). A new `tune_dals.py` driver samples configs with a seeded RNG, reuses `run_deep`'s `train`/`evaluate_split` for each trial at reduced epochs, scores each trial on both val IoU and val BF@2px, writes a ranked CSV per arm, and retrains the winner of each arm at full epochs with a test-set eval.

**Tech Stack:** Python, PyTorch, the existing `flame/` package (`flame.deep.dals`, `flame.deep.losses`, `flame.boundary_metrics`, `flame.splits`), `run_deep.py`.

**Note on testing:** This repo has no pytest suite; existing code is validated by small smoke runs against the real data. Tasks below use short inline Python verification commands in that style rather than a pytest harness.

---

### Task 1: Parameterize the DALS module

**Files:**
- Modify: `flame/deep/dals.py:42-50` (`DALS.__init__`)

- [ ] **Step 1: Make `DALS.__init__` accept structural params with current defaults**

Replace the `__init__` (lines 42-50) with:

```python
    def __init__(self, n_iter: int = 5, base: int = 32,
                 mu: float = 0.2, lam1: float = 1.0,
                 lam2: float = 1.0, dt: float = 0.1):
        super().__init__()
        self.trunk = UNet(in_ch=3, out_ch=1, base=base)
        self.n_iter = n_iter
        self.mu = nn.Parameter(torch.tensor(float(mu)))
        self.lam1 = nn.Parameter(torch.tensor(float(lam1)))
        self.lam2 = nn.Parameter(torch.tensor(float(lam2)))
        self.dt = nn.Parameter(torch.tensor(float(dt)))
```

- [ ] **Step 2: Verify defaults unchanged and params wire through**

Run:
```bash
python -c "
from flame.deep.dals import DALS
m = DALS()
assert m.n_iter == 5
assert abs(float(m.mu) - 0.2) < 1e-6 and abs(float(m.dt) - 0.1) < 1e-6
m2 = DALS(n_iter=8, mu=0.5, lam1=1.5, lam2=0.7, dt=0.2)
assert m2.n_iter == 8 and abs(float(m2.mu) - 0.5) < 1e-6
print('OK: DALS params wire through, defaults preserved')
"
```
Expected: `OK: DALS params wire through, defaults preserved`

- [ ] **Step 3: Commit**

```bash
git add flame/deep/dals.py
git commit -m "Parameterize DALS structural hyperparameters

n_iter, mu, lam1, lam2, dt now constructor args (defaults unchanged).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Make focal_tversky params overridable from a config

**Files:**
- Modify: `flame/deep/losses.py:19-38`

**Context:** `focal_tversky` already takes `alpha/beta/gamma` kwargs (line 19-21).
`bce_focal_tversky` (line 36-38) calls it with defaults and is what `run_deep.compute_loss`
uses. We need a way to pass non-default Tversky params through. Add a kwargs-forwarding
variant rather than changing existing signatures.

- [ ] **Step 1: Add a parameterized wrapper to `losses.py`**

Append to `flame/deep/losses.py`:

```python
def bce_focal_tversky_p(logits: torch.Tensor, target: torch.Tensor,
                        alpha: float = 0.3, beta: float = 0.7,
                        gamma: float = 0.75) -> torch.Tensor:
    """bce_focal_tversky with explicit Tversky params (for hyperparam search)."""
    return F.binary_cross_entropy_with_logits(logits, target) + \
        focal_tversky(logits, target, alpha=alpha, beta=beta, gamma=gamma)
```

- [ ] **Step 2: Verify it matches the default wrapper when params are default**

Run:
```bash
python -c "
import torch
from flame.deep.losses import bce_focal_tversky, bce_focal_tversky_p
torch.manual_seed(0)
lg = torch.randn(2,1,16,16); tg = (torch.rand(2,1,16,16) > 0.7).float()
a = bce_focal_tversky(lg, tg); b = bce_focal_tversky_p(lg, tg)
assert torch.allclose(a, b), (a, b)
c = bce_focal_tversky_p(lg, tg, alpha=0.2, beta=0.8, gamma=0.5)
assert not torch.allclose(a, c)
print('OK: parameterized FT loss matches default and varies with params')
"
```
Expected: `OK: parameterized FT loss matches default and varies with params`

- [ ] **Step 3: Commit**

```bash
git add flame/deep/losses.py
git commit -m "Add bce_focal_tversky_p with explicit Tversky params

Lets the hyperparam search vary alpha/beta/gamma without touching
the default loss path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Thread the new params through run_deep.py

**Files:**
- Modify: `run_deep.py` — `build_model` (line 57-60), `compute_loss` (line 146-164),
  `train`/`run_eval` model construction (lines 194, 227), `parse_args` (line 241-284)

**Context:** `build_model(method, in_ch)` constructs the model; for `dals` it currently
calls `DALS()` with no args. `compute_loss` selects `bce_focal_tversky` vs `bce_dice`.
We add a `cfg` dict carrying the search params and plumb it through. Keep all defaults
equal to current behavior.

- [ ] **Step 1: Add CLI flags in `parse_args`**

After the `--loss` argument block (line 253-254), add:

```python
    ap.add_argument("--n-iter", dest="n_iter", type=int, default=5,
                    help="DALS Chan-Vese unroll steps (dals only)")
    ap.add_argument("--mu", type=float, default=0.2, help="DALS init curvature weight")
    ap.add_argument("--lam1", type=float, default=1.0, help="DALS init inside-region weight")
    ap.add_argument("--lam2", type=float, default=1.0, help="DALS init outside-region weight")
    ap.add_argument("--dt", type=float, default=0.1, help="DALS init level-set time step")
    ap.add_argument("--ft-alpha", dest="ft_alpha", type=float, default=0.3,
                    help="Focal-Tversky alpha (FP weight), used when --loss focal_tversky")
    ap.add_argument("--ft-beta", dest="ft_beta", type=float, default=0.7,
                    help="Focal-Tversky beta (FN weight)")
    ap.add_argument("--ft-gamma", dest="ft_gamma", type=float, default=0.75,
                    help="Focal-Tversky focusing exponent")
```

- [ ] **Step 2: Make `build_model` accept DALS structural params**

Replace `build_model` (lines 57-60) with:

```python
def build_model(method: str, in_ch: int = 3,
                dals_cfg: dict | None = None) -> torch.nn.Module:
    if method == "unet":
        return UNet(in_ch=in_ch)
    if method == "dals":
        cfg = dals_cfg or {}
        return DALS(n_iter=cfg.get("n_iter", 5), mu=cfg.get("mu", 0.2),
                    lam1=cfg.get("lam1", 1.0), lam2=cfg.get("lam2", 1.0),
                    dt=cfg.get("dt", 0.1))
    return {"deep_snake": DeepSnake, "thermal": ThermalRegUNet}[method]()
```

- [ ] **Step 3: Make `compute_loss` use the parameterized FT loss for dals/unet**

In `compute_loss` (line 146-164), change the import line and the `seg` selection.
Replace line 147:

```python
    seg = bce_focal_tversky if loss == "focal_tversky" else bce_dice
```

with a closure that carries the FT params (passed via a new optional `ft` dict arg):

```python
def compute_loss(method: str, model, batch, device, loss: str = "bce_dice",
                 ft: dict | None = None) -> torch.Tensor:
    ft = ft or {}
    if loss == "focal_tversky":
        def seg(lg, tg):
            return bce_focal_tversky_p(lg, tg, alpha=ft.get("alpha", 0.3),
                                       beta=ft.get("beta", 0.7), gamma=ft.get("gamma", 0.75))
    else:
        seg = bce_dice
```

Also update the `focal_tversky(phi, mask)` call inside the dals branch (line 157) to pass params:

```python
        phi_loss = (focal_tversky(phi, mask, alpha=ft.get("alpha", 0.3),
                                  beta=ft.get("beta", 0.7), gamma=ft.get("gamma", 0.75))
                    if loss == "focal_tversky" else dice_loss(phi, mask))
```

Update the import at line 31-34 to include `bce_focal_tversky_p`:

```python
from flame.deep.losses import (
    bce_dice, bce_focal_tversky, bce_focal_tversky_p, cyclic_contour_loss,
    dice_loss, focal_tversky, weighted_thermal_loss,
)
```

- [ ] **Step 4: Pass cfg dicts from `train` into model + loss**

In `train` (line 194), change model construction:

```python
    dals_cfg = {"n_iter": args.n_iter, "mu": args.mu, "lam1": args.lam1,
                "lam2": args.lam2, "dt": args.dt}
    model = build_model(method, in_ch=in_ch, dals_cfg=dals_cfg).to(device)
```

In the training loop (line 206), change the loss call:

```python
            ft = {"alpha": args.ft_alpha, "beta": args.ft_beta, "gamma": args.ft_gamma}
            loss = compute_loss(method, model, batch, device, args.loss, ft=ft)
```

In `run_eval` (line 227), change model construction to use the same cfg:

```python
    dals_cfg = {"n_iter": args.n_iter, "mu": args.mu, "lam1": args.lam1,
                "lam2": args.lam2, "dt": args.dt}
    model = build_model(method, in_ch=SPEC_CHANNELS[spec], dals_cfg=dals_cfg).to(device)
```

- [ ] **Step 5: Verify a 1-epoch tuned DALS train run works end to end**

Run (FLAME-3, tiny, non-default params):
```bash
python run_deep.py --method dals --mode train --dataset flame3 \
  --limit 16 --epochs 1 --n-iter 8 --mu 0.5 --lam1 1.5 --lam2 0.7 --dt 0.2 \
  --loss focal_tversky --ft-beta 0.8 --ft-gamma 0.6 --tag _smoke 2>&1 | tail -5
```
Expected: prints `dals: ... train frames ...`, one `epoch 1 ... val IoU=...` line, and
`best val IoU=... -> models/dals_smoke.pt`. No exceptions.

- [ ] **Step 6: Verify default path is unchanged (regression check)**

Run:
```bash
python -c "
from run_deep import build_model
import torch
m = build_model('dals')
assert m.n_iter == 5 and abs(float(m.mu)-0.2) < 1e-6
print('OK: build_model dals defaults unchanged')
"
rm -f models/dals_smoke.pt results/dals_smoke_per_frame.csv
```
Expected: `OK: build_model dals defaults unchanged`

- [ ] **Step 7: Commit**

```bash
git add run_deep.py
git commit -m "Thread DALS + Focal-Tversky hyperparams through run_deep CLI

New flags --n-iter --mu --lam1 --lam2 --dt --ft-alpha --ft-beta --ft-gamma,
all defaulting to current values. build_model/compute_loss/train/run_eval
plumb them via cfg dicts.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Val boundary F-score helper

**Files:**
- Create: `flame/deep/val_metrics.py`

**Context:** Trials need a val-set mean boundary F-score @2px alongside val IoU.
`evaluate_split` in run_deep already produces predictions per frame; rather than
re-run inference, the helper takes a trained model and computes mean BF directly,
reusing `predict_native` and `boundary_fscore`.

- [ ] **Step 1: Create the helper**

Create `flame/deep/val_metrics.py`:

```python
"""Validation-set boundary F-score for the hyperparameter search.

Mirrors run_deep.evaluate_split's prediction path (predict_native) but scores
the symmetric boundary F-score @2px instead of IoU/Dice. Used by tune_dals.py
as the boundary objective.
"""
from __future__ import annotations

import numpy as np
import torch

from flame.boundary_metrics import boundary_fscore
from flame.data import load_frame
from flame.metrics import iou
from run_deep import predict_native


@torch.no_grad()
def val_iou_and_bf(method: str, model, frame_ids: list[str], threshold_c: float,
                   size: int, device, dataset: str = "flame3",
                   eval_max_side: int | None = None,
                   tol_px: float = 2.0) -> tuple[float, float]:
    """Mean val IoU and mean val boundary F-score@tol_px over frame_ids."""
    model.eval()
    ious, bfs = [], []
    for fid in frame_ids:
        frame = load_frame(fid, threshold_c=threshold_c, dataset=dataset,
                           max_side=eval_max_side)
        pred = predict_native(method, model, frame, size, device)
        ious.append(iou(pred, frame.gt_mask))
        bfs.append(boundary_fscore(pred, frame.gt_mask, tol_px=tol_px))
    return float(np.mean(ious)), float(np.mean(bfs))
```

- [ ] **Step 2: Verify it returns two floats in [0,1] on a quick run**

Run:
```bash
python -c "
import torch
from run_deep import build_model
from flame.splits import make_splits
from flame.deep.val_metrics import val_iou_and_bf
from flame.deep.dataset import NET_SIZE
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = build_model('dals').to(dev)
m.load_state_dict(torch.load('models/dals.pt', map_location=dev, weights_only=True))
val = make_splits(dataset='flame3')['val'][:8]
i, b = val_iou_and_bf('dals', m, val, 150.0, NET_SIZE, dev, dataset='flame3')
assert 0.0 <= i <= 1.0 and 0.0 <= b <= 1.0, (i, b)
print(f'OK: val IoU={i:.3f} val BF={b:.3f}')
"
```
Expected: `OK: val IoU=0.xxx val BF=0.xxx` (both finite, in range).

- [ ] **Step 3: Commit**

```bash
git add flame/deep/val_metrics.py
git commit -m "Add val IoU + boundary F-score helper for tuning

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The search driver (tune_dals.py) — config sampling

**Files:**
- Create: `tune_dals.py` (sampling + arg parsing only in this task)

- [ ] **Step 1: Create `tune_dals.py` with seeded config sampling**

Create `tune_dals.py`:

```python
"""Random-search hyperparameter optimization for DALS.

Two objectives (val IoU, val boundary F-score @2px) searched separately on each
dataset. Each trial trains DALS at reduced epochs with a sampled config; the best
config per (dataset, objective) is reported and (optionally) retrained at full
epochs with a test-set eval.

    python tune_dals.py --dataset flame3 --objective iou --trials 20 --epochs 25
    python tune_dals.py --dataset flame1 --objective bf  --trials 20 --epochs 25 --retrain

Outputs: results/dals_tune_<dataset>_<objective>.csv (ranked trials).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

N_ITER_CHOICES = [3, 5, 8, 12]
LOSS_CHOICES = ["bce_dice", "focal_tversky"]


def sample_config(rng: np.random.Generator) -> dict:
    """One random DALS config. FT params sampled but only used if loss=focal_tversky."""
    beta = float(rng.uniform(0.5, 0.8))
    return {
        "n_iter": int(rng.choice(N_ITER_CHOICES)),
        "mu": round(float(rng.uniform(0.1, 1.0)), 4),
        "lam1": round(float(rng.uniform(0.5, 2.0)), 4),
        "lam2": round(float(rng.uniform(0.5, 2.0)), 4),
        "dt": round(float(rng.uniform(0.05, 0.3)), 4),
        "loss": str(rng.choice(LOSS_CHOICES)),
        "ft_beta": round(beta, 4),
        "ft_alpha": round(1.0 - beta, 4),
        "ft_gamma": round(float(rng.uniform(0.5, 1.0)), 4),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["flame3", "flame1"], required=True)
    ap.add_argument("--objective", choices=["iou", "bf"], required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap train frames per trial (smoke testing)")
    ap.add_argument("--val-limit", dest="val_limit", type=int, default=None,
                    help="Cap val frames scored per trial (speed)")
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain the winning config at --full-epochs and eval test")
    ap.add_argument("--full-epochs", dest="full_epochs", type=int, default=50)
    return ap.parse_args()
```

- [ ] **Step 2: Verify sampling is seeded and ranges are respected**

Run:
```bash
python -c "
import numpy as np
from tune_dals import sample_config
r1 = np.random.default_rng(0); r2 = np.random.default_rng(0)
c1 = sample_config(r1); c2 = sample_config(r2)
assert c1 == c2, 'same seed must give same config'
for _ in range(200):
    c = sample_config(r1)
    assert c['n_iter'] in (3,5,8,12)
    assert 0.1 <= c['mu'] <= 1.0 and 0.05 <= c['dt'] <= 0.3
    assert 0.5 <= c['lam1'] <= 2.0 and 0.5 <= c['lam2'] <= 2.0
    assert 0.5 <= c['ft_beta'] <= 0.8 and abs(c['ft_alpha'] + c['ft_beta'] - 1.0) < 1e-6
    assert c['loss'] in ('bce_dice','focal_tversky')
print('OK: sampling seeded and within ranges')
"
```
Expected: `OK: sampling seeded and within ranges`

- [ ] **Step 3: Commit**

```bash
git add tune_dals.py
git commit -m "tune_dals: seeded config sampling + CLI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The search driver — train/score one trial

**Files:**
- Modify: `tune_dals.py`

**Context:** We reuse `run_deep`'s `train`-style logic but need per-trial control of
epochs and config without writing a checkpoint per trial. Build a small trial runner
that constructs the dataset/model/optimizer inline (mirroring `run_deep.train`) so we
can keep the model in memory and score it with `val_iou_and_bf`. This duplicates a few
lines of `run_deep.train` deliberately — the trial runner needs the in-memory model,
which `train` does not return.

- [ ] **Step 1: Add the trial runner to `tune_dals.py`**

Append:

```python
import torch
from torch.utils.data import DataLoader

from flame.deep.dataset import NET_SIZE, FlameDataset
from flame.deep.val_metrics import val_iou_and_bf
from flame.data import DEFAULT_THRESHOLD_C
from flame.splits import make_splits
from run_deep import build_model, compute_loss

THRESHOLD_C = DEFAULT_THRESHOLD_C


def _eval_max_side(dataset: str) -> int | None:
    return 1024 if dataset == "flame1" else None


def _load_max_side(dataset: str) -> int | None:
    return 1024 if dataset == "flame1" else None


def run_trial(cfg: dict, dataset: str, epochs: int, device,
              limit: int | None, val_limit: int | None) -> dict:
    """Train DALS with cfg for `epochs`, return cfg + final val IoU/BF."""
    splits = make_splits(dataset=dataset)
    train_ids, val_ids = splits["train"], splits["val"]
    if limit:
        train_ids = train_ids[:limit]
    if val_limit:
        val_ids = val_ids[:val_limit]

    ds = FlameDataset(train_ids, threshold_c=THRESHOLD_C, size=NET_SIZE,
                      augment=True, in_channels="rgb", aug_mode="light",
                      dataset=dataset, load_max_side=_load_max_side(dataset))
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    dals_cfg = {k: cfg[k] for k in ("n_iter", "mu", "lam1", "lam2", "dt")}
    model = build_model("dals", dals_cfg=dals_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ft = {"alpha": cfg["ft_alpha"], "beta": cfg["ft_beta"], "gamma": cfg["ft_gamma"]}

    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            loss = compute_loss("dals", model, batch, device, cfg["loss"], ft=ft)
            loss.backward()
            opt.step()

    val_iou, val_bf = val_iou_and_bf("dals", model, val_ids, THRESHOLD_C, NET_SIZE,
                                     device, dataset=dataset,
                                     eval_max_side=_eval_max_side(dataset))
    return {**cfg, "val_iou": round(val_iou, 4), "val_bf": round(val_bf, 4)}
```

- [ ] **Step 2: Verify one trial runs and returns scored config**

Run:
```bash
python -c "
import numpy as np, torch
from tune_dals import sample_config, run_trial
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = sample_config(np.random.default_rng(1))
r = run_trial(cfg, 'flame3', epochs=1, device=dev, limit=16, val_limit=8)
assert 'val_iou' in r and 'val_bf' in r
assert 0.0 <= r['val_iou'] <= 1.0 and 0.0 <= r['val_bf'] <= 1.0
print(f'OK: trial scored val_iou={r[\"val_iou\"]} val_bf={r[\"val_bf\"]}')
"
```
Expected: `OK: trial scored val_iou=... val_bf=...`

- [ ] **Step 3: Commit**

```bash
git add tune_dals.py
git commit -m "tune_dals: single-trial train+score runner

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: The search driver — search loop, ranking, CSV, optional retrain

**Files:**
- Modify: `tune_dals.py`

- [ ] **Step 1: Add the search loop + main()**

Append:

```python
def retrain_and_test(cfg: dict, dataset: str, full_epochs: int, objective: str,
                     device) -> dict:
    """Retrain winning cfg at full epochs, eval on test split. Saves a checkpoint."""
    from run_deep import evaluate_split, _prefix
    splits = make_splits(dataset=dataset)
    train_ids, test_ids = splits["train"], splits["test"]
    ds = FlameDataset(train_ids, threshold_c=THRESHOLD_C, size=NET_SIZE,
                      augment=True, in_channels="rgb", aug_mode="light",
                      dataset=dataset, load_max_side=_load_max_side(dataset))
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    dals_cfg = {k: cfg[k] for k in ("n_iter", "mu", "lam1", "lam2", "dt")}
    model = build_model("dals", dals_cfg=dals_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ft = {"alpha": cfg["ft_alpha"], "beta": cfg["ft_beta"], "gamma": cfg["ft_gamma"]}
    for _ in range(full_epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            loss = compute_loss("dals", model, batch, device, cfg["loss"], ft=ft)
            loss.backward()
            opt.step()
    tag = f"_tuned_{objective}"
    ckpt = ROOT / "models" / f"{_prefix(dataset)}dals{tag}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    summary = evaluate_split("dals", model, test_ids, THRESHOLD_C, NET_SIZE, device,
                             dataset=dataset, eval_max_side=_eval_max_side(dataset))
    return {"checkpoint": str(ckpt), "test_iou": round(summary["iou"], 4),
            "test_dice": round(summary["dice"], 4), "n_test": summary["n"]}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    key = "val_iou" if args.objective == "iou" else "val_bf"

    trials = []
    for t in range(1, args.trials + 1):
        cfg = sample_config(rng)
        res = run_trial(cfg, args.dataset, args.epochs, device,
                        args.limit, args.val_limit)
        trials.append(res)
        print(f"[{args.dataset}/{args.objective}] trial {t:2d}/{args.trials}  "
              f"n_iter={res['n_iter']} loss={res['loss']:13s} "
              f"val_iou={res['val_iou']:.4f} val_bf={res['val_bf']:.4f}")

    trials.sort(key=lambda r: r[key], reverse=True)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"dals_tune_{args.dataset}_{args.objective}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        w.writeheader(); w.writerows(trials)
    print(f"\nwrote {out}  (ranked by {key})")
    best = trials[0]
    print(f"BEST {args.dataset}/{args.objective}: {key}={best[key]}  cfg={best}")

    if args.retrain:
        print(f"\nRetraining winner at {args.full_epochs} epochs...")
        ft_res = retrain_and_test(best, args.dataset, args.full_epochs,
                                  args.objective, device)
        print(f"RETRAINED: {ft_res}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify a tiny end-to-end search (2 trials, 1 epoch) writes a ranked CSV**

Run:
```bash
python tune_dals.py --dataset flame3 --objective iou --trials 2 --epochs 1 \
  --limit 16 --val-limit 8 --seed 0 2>&1 | tail -6
test -f results/dals_tune_flame3_iou.csv && \
  echo "CSV rows: $(($(wc -l < results/dals_tune_flame3_iou.csv) - 1))"
```
Expected: two `trial` lines, a `wrote ... ranked by val_iou` line, a `BEST ...` line,
and `CSV rows: 2`.

- [ ] **Step 3: Verify CSV is sorted by objective**

Run:
```bash
python -c "
import csv
rows = list(csv.DictReader(open('results/dals_tune_flame3_iou.csv')))
vals = [float(r['val_iou']) for r in rows]
assert vals == sorted(vals, reverse=True), vals
print('OK: ranked descending by val_iou')
"
rm -f results/dals_tune_flame3_iou.csv
```
Expected: `OK: ranked descending by val_iou`

- [ ] **Step 4: Commit**

```bash
git add tune_dals.py
git commit -m "tune_dals: search loop, ranking, CSV, optional retrain+test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Cost smoke test + projected-runtime report (verification gate)

**Files:** none (operational step)

**Context:** Before committing the GPU to 4 arms × 20 trials × 25 epochs, measure
real per-trial wall-clock on each dataset and report the projection. This is the
gate agreed with the user.

- [ ] **Step 1: Time 2 trials × 2 epochs on each dataset**

Run:
```bash
echo "=== FLAME-3 timing ==="; time python tune_dals.py --dataset flame3 \
  --objective iou --trials 2 --epochs 2 --val-limit 16 --seed 0 2>&1 | tail -3
echo "=== FLAME-1 timing ==="; time python tune_dals.py --dataset flame1 \
  --objective iou --trials 2 --epochs 2 --val-limit 16 --seed 0 2>&1 | tail -3
rm -f results/dals_tune_flame3_iou.csv results/dals_tune_flame1_iou.csv
```
Expected: both complete without error; capture the `real` times.

- [ ] **Step 2: Compute and report the projection**

From the measured per-(2-trial,2-epoch) times, extrapolate to 20 trials × 25 epochs
× 4 arms (FLAME-1 ×2 objectives, FLAME-3 ×2). Report the projected total to the user
and STOP for green-light before Task 9. Note: the two objectives per dataset can share
nothing (different RNG streams) so they are 2 independent searches each.

---

### Task 9: Run the full 4-arm search + report (after green-light)

**Files:** writes `results/dals_tune_*.csv`, `models/*dals_tuned_*.pt`

- [ ] **Step 1: Run all four arms with retrain**

Run (sequential; adjust trials/epochs only if Task 8 projection required trimming):
```bash
for ds in flame3 flame1; do for obj in iou bf; do
  python tune_dals.py --dataset $ds --objective $obj --trials 20 --epochs 25 \
    --retrain --full-epochs 50 --seed 0 2>&1 | tee logs/dals_tune_${ds}_${obj}.log
done; done
```
Expected: 4 ranked CSVs in `results/`, 4 checkpoints `models/{,flame1_}dals_tuned_{iou,bf}.pt`,
4 `RETRAINED:` lines with test IoU/Dice.

- [ ] **Step 2: Build the comparison summary**

Run:
```bash
python -c "
import csv, glob
print('arm | best val_iou | best val_bf | retrained test_iou (from logs)')
for f in sorted(glob.glob('results/dals_tune_*.csv')):
    rows = list(csv.DictReader(open(f)))
    top = rows[0]
    print(f'{f.split(\"dals_tune_\")[1][:-4]:14s} | {top[\"val_iou\"]} | {top[\"val_bf\"]} | n_iter={top[\"n_iter\"]} loss={top[\"loss\"]}')
"
```
Expected: one line per arm with the winning config.

- [ ] **Step 3: Report to user**

Summarize: per dataset, baseline DALS (FLAME-3 IoU 0.122 / FLAME-1 0.747) vs.
best-IoU-config test IoU vs. best-BF-config, and whether the IoU-arm and BF-arm
picked different configs. Note any config that beat the baseline.

- [ ] **Step 4: Commit results**

```bash
git add results/dals_tune_*.csv docs/ tune_dals.py
git commit -m "DALS hyperparameter search results (4 arms, both datasets)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** search space (Tasks 1-3, 5), two objectives (Task 4 + `key` in
  Task 7), 4 arms (Task 9), random/seeded/reduced-epoch search (Tasks 5-7), winner
  retrain + test eval (Task 7 `retrain_and_test`, Task 9), ranked CSV per arm (Task 7),
  smoke test + projected-runtime gate (Task 8). All spec sections mapped.
- **Backward compatibility:** every new flag/arg defaults to current values; Task 3
  steps 6 and Task 1 step 2 explicitly regression-check defaults.
- **Type consistency:** `dals_cfg` keys (`n_iter,mu,lam1,lam2,dt`) and `ft` keys
  (`alpha,beta,gamma`) are used identically in `build_model`, `compute_loss`,
  `run_trial`, `retrain_and_test`. Objective `key` is `val_iou`/`val_bf`, matching the
  dict keys returned by `run_trial`.
- **Checkpoints not overwritten:** winners saved as `dals_tuned_{iou,bf}.pt` /
  `flame1_dals_tuned_{iou,bf}.pt` via `_prefix`, never the baseline `dals.pt`.
