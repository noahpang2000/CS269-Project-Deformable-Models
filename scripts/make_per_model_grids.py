"""Per-model output grids for the paper.

FLAME-1: 2x2 = 2 frames (rows) x [RGB+GT | RGB+prediction].
FLAME-3: 2x3 = rows (RGB+GT, thermal, RGB+prediction) x 2 frame-columns.
One PNG per model type, emailed individually. Outputs report/figs/per_model/.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figs" / "per_model"
OUT.mkdir(parents=True, exist_ok=True)

from flame.data import load_frame, smoke_fraction
from flame.splits import make_splits
from flame.metrics import iou as iou_fn, dice as dice_fn
from flame.deep.unet import UNet
from flame.deep.dals import DALS
from flame.deep.deep_snake_simplified import DeepSnake
from flame.deep.deep_snake import DeepSnakePipeline
from flame.kass import run_kass, KassConfig
from flame.gac import run_gac, GACConfig
import run_deep as R

SIZE, device = 512, "cuda"
RED, BLUE = (0, 0, 255), (255, 90, 0)
GAP = 5

# Diverse FLAME-1 frames (2 distinct scenes).
F1 = ["image_1702", "image_1947"]


def overlay(bgr, mask, color, alpha=0.45):
    out = bgr.copy(); m = (mask > 0).astype(np.uint8)
    t = np.zeros_like(out); t[m > 0] = color
    out = cv2.addWeighted(out, 1.0, t, alpha, 0)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def grid(rows, gap=GAP):
    """rows: list of lists of HxWx3 images -> tiled with white separators."""
    hsep = lambda w: np.full((gap, w, 3), 255, np.uint8)
    out_rows = []
    for r in rows:
        parts = []
        for i, im in enumerate(r):
            if i: parts.append(np.full((im.shape[0], gap, 3), 255, np.uint8))
            parts.append(im)
        out_rows.append(np.hstack(parts))
    final = []
    for i, r in enumerate(out_rows):
        if i: final.append(hsep(r.shape[1]))
        final.append(r)
    return np.vstack(final)


def fit(mask, sz):
    return cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST) if mask.shape[0] != sz else mask


# ---- predictors keyed by model name; return native-res 0/255 mask ----
def make_predictors(dataset):
    pref = "flame1_" if dataset == "flame1" else ""
    preds = {}

    @torch.no_grad()
    def unet_fn():
        m = UNet().to(device).eval(); m.load_state_dict(torch.load(ROOT/f"models/{pref}unet.pt", map_location=device, weights_only=True))
        def f(fr, sz):
            x = R.image_tensor(fr.rgb, SIZE, device)
            p = (torch.sigmoid(m(x))[0,0].detach().cpu().numpy() > 0.5).astype(np.uint8)*255
            return fit(p, sz)
        return f
    preds["U-Net"] = unet_fn

    @torch.no_grad()
    def snake_simple_fn():
        m = DeepSnake().to(device).eval(); m.load_state_dict(torch.load(ROOT/f"models/{pref}deep_snake_simple.pt", map_location=device, weights_only=True))
        def f(fr, sz):
            with torch.no_grad():
                x = R.image_tensor(fr.rgb, SIZE, device)
                return fit(R._snake_predict(m, x, SIZE, device)*255, sz)
        return f
    preds["Deep Snake (simple)"] = snake_simple_fn

    def kass_fn():
        def f(fr, sz):
            return fit(run_kass(fr, KassConfig(), init_mask=None), sz)  # oracle
        return f
    preds["Kass (oracle)"] = kass_fn

    def gac_fn():
        def f(fr, sz):
            return fit(run_gac(fr, GACConfig(), init_mask=None), sz)    # oracle
        return f
    preds["GAC (oracle)"] = gac_fn

    if dataset == "flame1":
        @torch.no_grad()
        def dals_fn():
            m = DALS().to(device).eval(); m.load_state_dict(torch.load(ROOT/"models/flame1_dals.pt", map_location=device, weights_only=True))
            def f(fr, sz):
                x = R.image_tensor(fr.rgb, SIZE, device); phi,_ = m(x)
                p = (torch.sigmoid(phi)[0,0].detach().cpu().numpy() > 0.5).astype(np.uint8)*255
                return fit(p, sz)
            return f
        preds["DALS"] = dals_fn

        def paper_fn():
            pipe = DeepSnakePipeline(str(ROOT/"flame/deep/centernet_flame1.py"),
                                     str(ROOT/"results/rerun_flame1_20260601_193349/centernet/epoch_10.pth"),
                                     snake_feat_dim=64, device=device).to(device).eval()
            tw = torch.load(ROOT/"models/flame1_deep_snake_paper.pt", map_location=device, weights_only=True)
            pipe.snake.load_state_dict({k[6:]: v for k,v in tw.items() if k.startswith("snake.")})
            def f(fr, sz):
                with torch.no_grad():
                    return fit(R._snake_predict_paper(pipe, fr, SIZE, device, conf_threshold=0.05)*255, sz)
            return f
        preds["Deep Snake (paper)"] = paper_fn
    return preds


def thermal_panel(fr, sz):
    th = fr.thermal_c
    thn = np.clip((th - th.min())/(th.ptp()+1e-6), 0, 1)
    return cv2.applyColorMap((cv2.resize(thn,(sz,sz))*255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def flame1_grids():
    sz = 360
    preds = make_predictors("flame1")
    for name, builder in preds.items():
        fn = builder()
        rows = []
        for fid in F1:
            # classical methods need a downscaled frame for tractability
            classical = name in ("Kass (oracle)", "GAC (oracle)")
            fr = load_frame(fid, dataset="flame1", max_side=(sz if classical else None))
            bgr = cv2.cvtColor(cv2.resize(fr.rgb, (sz, sz)), cv2.COLOR_RGB2BGR)
            gt = cv2.resize(fr.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
            pred = fn(fr, sz)
            rows.append([overlay(bgr, gt, BLUE), overlay(bgr, pred, RED)])
        out = OUT / f"flame1_{name.replace(' ','_').replace('(','').replace(')','')}.png"
        cv2.imwrite(str(out), grid(rows)); print(out.name)


def flame3_grids():
    sz = 340
    # FLAME-3: 2 non-degenerate frames with real fire
    f3 = []
    for fid in make_splits(dataset="flame3")["test"]:
        fr = load_frame(fid, dataset="flame3")
        if smoke_fraction(fr.rgb) < 0.5 and (fr.gt_mask>0).sum() >= 200:
            f3.append(fid)
        if len(f3) >= 2: break
    preds = make_predictors("flame3")
    for name, builder in preds.items():
        fn = builder()
        # rows = (GT, thermal, prediction); columns = the 2 frames
        gt_row, th_row, pr_row = [], [], []
        for fid in f3:
            classical = name in ("Kass (oracle)", "GAC (oracle)")
            fr = load_frame(fid, dataset="flame3", max_side=(sz if classical else None))
            bgr = cv2.cvtColor(cv2.resize(fr.rgb, (sz, sz)), cv2.COLOR_RGB2BGR)
            gt = cv2.resize(fr.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
            pred = fn(fr, sz)
            gt_row.append(overlay(bgr, gt, BLUE))
            th_row.append(thermal_panel(fr, sz))
            pr_row.append(overlay(bgr, pred, RED))
        out = OUT / f"flame3_{name.replace(' ','_').replace('(','').replace(')','')}.png"
        cv2.imwrite(str(out), grid([gt_row, th_row, pr_row])); print(out.name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="both")
    a = ap.parse_args()
    if a.ds in ("both", "flame1"): flame1_grids()
    if a.ds in ("both", "flame3"): flame3_grids()
