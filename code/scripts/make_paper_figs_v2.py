"""Rich narrative figure set for the wildfire paper. Each figure makes ONE point;
methods are shown in small groups, not all at once. Outputs report/figs/paper/.
"""
import sys, math
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

from code.flame.data import load_frame, smoke_fraction
from code.flame.splits import make_splits

SMOKE_TAU = 0.5  # frames with smoke_fraction >= this are degenerate; excluded


def clean_ids(ds, k=4, min_gt_px=200):
    """First k test ids that are (a) NOT degenerate smoke-outs and (b) have a
    real fire GT mask of at least min_gt_px pixels -- so FLAME-3 visuals always
    show an actual fire to segment, never an empty landscape."""
    out = []
    for fid in make_splits(dataset=ds)["test"]:
        fr = load_frame(fid, dataset=ds)
        if smoke_fraction(fr.rgb) >= SMOKE_TAU:
            continue
        if (fr.gt_mask > 0).sum() < min_gt_px:
            continue
        out.append(fid)
        if len(out) >= k:
            break
    return out
from code.flame.contour_utils import fire_energy
from code.flame.metrics import iou as iou_fn, dice as dice_fn
from code.flame.deep.unet import UNet
from code.flame.deep.dals import DALS
from code.flame.deep.deep_snake_simplified import DeepSnake
from code.flame.deep.deep_snake import DeepSnakePipeline
import code.run_deep as R

SIZE, device = 512, "cuda"
RED, BLUE, GRN = (0, 0, 255), (255, 90, 0), (0, 200, 0)

# Visually DIVERSE FLAME-1 frames (the test split is contiguous video, so
# adjacent ids look near-identical). These are spread across the split with
# different fire locations/sizes: big-left, centered, right-shifted, far-right.
F1_DIVERSE = ["image_1702", "image_1838", "image_1947", "image_2002"]


GAP = 4  # thin white separator between panels (replaces black title bars)


def overlay(bgr, mask, color, alpha=0.45):
    out = bgr.copy(); m = (mask > 0).astype(np.uint8)
    t = np.zeros_like(out); t[m > 0] = color
    out = cv2.addWeighted(out, 1.0, t, alpha, 0)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def col(cells_with_titles, sz):
    """Stack images vertically into one column with thin white separators.
    (Titles are ignored -- labeling is done in the LaTeX caption, no black bars.)"""
    imgs = []
    for _title, img in cells_with_titles:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imgs.append(img)
    sep = np.full((GAP, sz, 3), 255, np.uint8)
    parts = []
    for i, im in enumerate(imgs):
        if i: parts.append(sep)
        parts.append(im)
    return np.vstack(parts)


def hstack_sep(cols):
    """Horizontally stack columns with thin white separators."""
    h = cols[0].shape[0]
    sep = np.full((h, GAP, 3), 255, np.uint8)
    parts = []
    for i, c in enumerate(cols):
        if i: parts.append(sep)
        parts.append(c)
    return np.hstack(parts)


def add_header(img, text, h=34):
    """Prepend a single dark header bar with `text` to the top of a montage."""
    w = img.shape[1]
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([bar, img])


def get(fid, ds, sz=SIZE):
    fr = load_frame(fid, dataset=ds)
    rgb = cv2.resize(fr.rgb, (sz, sz))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gt = cv2.resize(fr.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
    return fr, rgb, bgr, gt


def load_unet(p):
    m = UNet().to(device).eval(); m.load_state_dict(torch.load(ROOT/p, map_location=device, weights_only=True)); return m

def load_dals(p):
    m = DALS().to(device).eval(); m.load_state_dict(torch.load(ROOT/p, map_location=device, weights_only=True)); return m

def load_snake(p):
    m = DeepSnake().to(device).eval(); m.load_state_dict(torch.load(ROOT/p, map_location=device, weights_only=True)); return m

def _fit(mask, sz):
    """Resize a SIZE-res prediction to the display size sz (mask is HxW)."""
    if mask.shape[0] != sz:
        mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
    return mask

@torch.no_grad()
def p_unet(m, rgb):
    x = R.image_tensor(rgb, SIZE, device)
    p = (torch.sigmoid(m(x))[0,0].cpu().numpy() > 0.5).astype(np.uint8)*255
    return _fit(p, rgb.shape[0])

@torch.no_grad()
def p_dals(m, rgb):
    x = R.image_tensor(rgb, SIZE, device); phi,_ = m(x)
    p = (torch.sigmoid(phi)[0,0].cpu().numpy() > 0.5).astype(np.uint8)*255
    return _fit(p, rgb.shape[0])

@torch.no_grad()
def p_snake(m, rgb):
    x = R.image_tensor(rgb, SIZE, device)
    p = R._snake_predict(m, x, SIZE, device)*255
    return _fit(p, rgb.shape[0])

def score(p, gt):
    return iou_fn(p, gt), dice_fn(p, gt)


# ================= FIG A: FLAME-1 dataset gallery (RGB + GT) =================
def fig_flame1_gallery():
    sz = 300; ids = F1_DIVERSE
    cols = []
    for fid in ids:
        fr, rgb, bgr, gt = get(fid, "flame1", sz)
        cols.append(col([("FLAME-1 RGB", bgr), ("flame GT (red)", overlay(bgr, gt, RED))], sz))
    cv2.imwrite(str(OUT/"f1_gallery.png"), hstack_sep(cols)); print("f1_gallery.png")


# ============ FIG B: FLAME-3 dataset gallery (RGB + thermal + GT) ============
def fig_flame3_gallery():
    sz = 300; ids = clean_ids("flame3", 4)
    cols = []
    for fid in ids:
        fr = load_frame(fid, dataset="flame3")
        rgb = cv2.resize(fr.rgb, (sz, sz)); bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gt = cv2.resize(fr.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
        th = fr.thermal_c
        thn = np.clip((th - th.min())/(th.ptp()+1e-6), 0, 1)
        thn = cv2.applyColorMap((cv2.resize(thn,(sz,sz))*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cols.append(col([("FLAME-3 RGB", bgr), ("thermal", thn),
                         ("fire GT (red)", overlay(bgr, gt, RED))], sz))
    cv2.imwrite(str(OUT/"f3_gallery.png"), hstack_sep(cols)); print("f3_gallery.png")


# ===== FIG C: color-energy diagnostic (R-G) on FLAME-3 only =====
# (FLAME-1 R-G row dropped per request; the point is that on FLAME-3 the colour
#  energy is uninformative where the GT fire is.)
def fig_color_energy():
    sz = 300
    cols = []
    for fid in clean_ids("flame3", 4):
        fr, rgb, bgr, gt = get(fid, "flame3", sz)
        e = fire_energy(rgb, "rg")
        eh = cv2.applyColorMap((e*255).astype(np.uint8), cv2.COLORMAP_JET)
        cols.append(col([("FLAME-3 RGB", bgr), ("R-G energy", eh),
                         ("GT", overlay(bgr, gt, RED))], sz))
    cv2.imwrite(str(OUT/"color_energy.png"), hstack_sep(cols)); print("color_energy.png")


# ===== FIG D: FLAME-3 failure gallery (U-Net, DALS, DeepSnake all fail) =====
def fig_flame3_failures():
    # NB: the FLAME-3 dals.pt predates the DALS rewrite (incompatible shape), so we
    # show U-Net + Deep Snake here -- both fail, which is the point. We require
    # frames where BOTH models actually PRODUCE a prediction (non-empty), so no
    # row is blank -- the failure is that the predictions miss, not that there's
    # nothing drawn.
    sz = 280
    un = load_unet("models/unet.pt"); sn = load_snake("models/deep_snake_simple.pt")
    ids = []
    for fid in clean_ids("flame3", 40):           # scan a larger pool
        fr, rgb, bgr, gt = get(fid, "flame3", sz)
        if p_unet(un, rgb).sum() > 0 and p_snake(sn, rgb).sum() > 0:
            ids.append(fid)
        if len(ids) >= 3:
            break
    cols = []
    for fid in ids:
        fr, rgb, bgr, gt = get(fid, "flame3", sz)
        pu, ps = p_unet(un, rgb), p_snake(sn, rgb)
        cols.append(col([("FLAME-3 RGB", bgr), ("GT (blue)", overlay(bgr, gt, BLUE)),
                         (f"U-Net IoU{score(pu,gt)[0]:.2f}", overlay(bgr, pu, RED)),
                         (f"DeepSnake IoU{score(ps,gt)[0]:.2f}", overlay(bgr, ps, RED))], sz))
    cv2.imwrite(str(OUT/"f3_failures.png"), hstack_sep(cols)); print("f3_failures.png", ids)


# ===== FIG E: FLAME-1 winner gallery (U-Net only, several frames) =====
def fig_flame1_unet():
    sz = 300; ids = F1_DIVERSE
    un = load_unet("models/flame1_unet.pt")
    cols = []
    for fid in ids:
        fr, rgb, bgr, gt = get(fid, "flame1", sz)
        pu = p_unet(un, rgb)
        cols.append(col([("RGB", bgr), ("GT (blue)", overlay(bgr, gt, BLUE)),
                         (f"U-Net (red) IoU{score(pu,gt)[0]:.2f}", overlay(bgr, pu, RED))], sz))
    cv2.imwrite(str(OUT/"f1_unet.png"), hstack_sep(cols)); print("f1_unet.png")


# ===== FIG F: region vs contour -- U-Net vs DALS vs Deep Snake (one frame, big) =====
def fig_region_vs_contour():
    sz = 360
    fid = make_splits(dataset="flame1")["test"][2]
    fr, rgb, bgr, gt = get(fid, "flame1", sz)
    un = load_unet("models/flame1_unet.pt"); da = load_dals("models/flame1_dals.pt"); sn = load_snake("models/flame1_deep_snake_simple.pt")
    pu, pd, ps = p_unet(un, rgb), p_dals(da, rgb), p_snake(sn, rgb)
    cells = [("GT (blue)", overlay(bgr, gt, BLUE)),
             (f"U-Net D{score(pu,gt)[1]:.2f}", overlay(bgr, pu, RED)),
             (f"DALS D{score(pd,gt)[1]:.2f}", overlay(bgr, pd, RED)),
             (f"DeepSnake D{score(ps,gt)[1]:.2f}", overlay(bgr, ps, RED))]
    row = hstack_sep([col([c], sz) for c in cells])
    cv2.imwrite(str(OUT/"region_vs_contour.png"), row); print("region_vs_contour.png")


# ===== FIG G: detector-gated Deep Snake (paper) -- shows misses =====
def fig_paper_snake():
    sz = 300; ids = [make_splits(dataset="flame1")["test"][i] for i in (2,20,50)]
    pipe = DeepSnakePipeline(str(ROOT/"flame/deep/centernet_flame1.py"),
                             str(ROOT/"results/rerun_flame1_20260601_193349/centernet/epoch_10.pth"),
                             snake_feat_dim=64, device=device).to(device).eval()
    tw = torch.load(ROOT/"models/flame1_deep_snake_paper.pt", map_location=device, weights_only=True)
    pipe.snake.load_state_dict({k[6:]: v for k,v in tw.items() if k.startswith("snake.")})
    cols = []
    for fid in ids:
        fr, rgb, bgr, gt = get(fid, "flame1", sz)
        pp = R._snake_predict_paper(pipe, fr, sz, device, conf_threshold=0.05)*255
        cols.append(col([("RGB", bgr), ("GT (blue)", overlay(bgr, gt, BLUE)),
                         (f"DeepSnake-paper IoU{score(pp,gt)[0]:.2f}", overlay(bgr, pp, RED))], sz))
    cv2.imwrite(str(OUT/"paper_snake.png"), hstack_sep(cols)); print("paper_snake.png")


# ===== FIG H: initialization study visual (GAC color vs oracle, FLAME-3) =====
def fig_init_study():
    from code.flame.gac import run_gac, GACConfig
    from code.flame.baselines import color_threshold_mask
    sz = 320
    fid = clean_ids("flame3", 1)[0]
    fr = load_frame(fid, dataset="flame3")
    bgr = cv2.cvtColor(cv2.resize(fr.rgb,(sz,sz)), cv2.COLOR_RGB2BGR)
    gt = cv2.resize(fr.gt_mask,(sz,sz),interpolation=cv2.INTER_NEAREST)
    # downscale frame for speed
    frs = load_frame(fid, dataset="flame3", max_side=sz) if False else fr
    col_init = color_threshold_mask(fr.rgb, 0.5)
    gac_color = run_gac(fr, GACConfig(), init_mask=col_init)
    gac_oracle = run_gac(fr, GACConfig(), init_mask=None)  # oracle from GT
    gc = cv2.resize(gac_color,(sz,sz),interpolation=cv2.INTER_NEAREST)
    go = cv2.resize(gac_oracle,(sz,sz),interpolation=cv2.INTER_NEAREST)
    cells = [("FLAME-3 RGB", bgr), ("GT (blue)", overlay(bgr, gt, BLUE)),
             (f"GAC color-init IoU{score(gc,gt)[0]:.2f}", overlay(bgr, gc, RED)),
             (f"GAC oracle-init IoU{score(go,gt)[0]:.2f}", overlay(bgr, go, RED))]
    cv2.imwrite(str(OUT/"init_study_vis.png"), hstack_sep([col([c],sz) for c in cells])); print("init_study_vis.png")


# ===== FIG I: what oracle initialization looks like (FLAME-1) =====
def fig_oracle_init():
    from code.flame.contour_utils import (init_snake_from_mask, polygon_to_mask,
                                     outer_contour, largest_component_mask)
    from code.flame.gac import run_gac, GACConfig
    sz = 340
    fid = make_splits(dataset="flame1")["test"][2]
    fr = load_frame(fid, dataset="flame1", max_side=sz)   # downscale for the classical run
    bgr = cv2.cvtColor(cv2.resize(fr.rgb, (sz, sz)), cv2.COLOR_RGB2BGR)
    gt = cv2.resize(fr.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
    # The oracle seed: GT contour scaled out 1.15x (what Kass starts from).
    seed_xy = init_snake_from_mask(cv2.resize(fr.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST),
                                   dilate_factor=1.15, n_points=200)
    seed_mask = polygon_to_mask(seed_xy, (sz, sz)) if len(seed_xy) >= 3 else np.zeros((sz, sz), np.uint8)
    # Evolved GAC result from that oracle seed.
    gac = run_gac(fr, GACConfig(), init_mask=None)
    gac = cv2.resize(gac, (sz, sz), interpolation=cv2.INTER_NEAREST)
    cells = [("RGB", bgr),
             ("GT (green)", overlay(bgr, gt, GRN)),
             ("oracle init: GT dilated 15% (yellow)", overlay(bgr, seed_mask, (0, 200, 255))),
             (f"GAC evolved from oracle (red) D{score(gac, gt)[1]:.2f}", overlay(bgr, gac, RED))]
    cv2.imwrite(str(OUT/"oracle_init.png"), hstack_sep([col([c], sz) for c in cells]))
    print("oracle_init.png")


# ===== FIG J: broad FLAME-1 output gallery across ALL method types =====
def fig_all_methods():
    from code.flame.kass import run_kass, KassConfig
    from code.flame.gac import run_gac, GACConfig
    from code.flame.deep.deep_snake import DeepSnakePipeline
    sz = 240
    ids = [F1_DIVERSE[0], F1_DIVERSE[1], F1_DIVERSE[3]]   # diverse: big-left, centered, far-right
    un = load_unet("models/flame1_unet.pt")
    da = load_dals("models/flame1_dals.pt")
    sn = load_snake("models/flame1_deep_snake_simple.pt")
    pipe = DeepSnakePipeline(str(ROOT/"flame/deep/centernet_flame1.py"),
                             str(ROOT/"results/rerun_flame1_20260601_193349/centernet/epoch_10.pth"),
                             snake_feat_dim=64, device=device).to(device).eval()
    tw = torch.load(ROOT/"models/flame1_deep_snake_paper.pt", map_location=device, weights_only=True)
    pipe.snake.load_state_dict({k[6:]: v for k, v in tw.items() if k.startswith("snake.")})

    cols = []
    for fid in ids:
        fr, rgb, bgr, gt = get(fid, "flame1", sz)
        frs = load_frame(fid, dataset="flame1", max_side=sz)   # downscaled for classical
        bgrs = cv2.cvtColor(cv2.resize(frs.rgb, (sz, sz)), cv2.COLOR_RGB2BGR)
        gts = cv2.resize(frs.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
        kass = cv2.resize(run_kass(frs, KassConfig(), init_mask=None), (sz, sz), interpolation=cv2.INTER_NEAREST)
        gac = cv2.resize(run_gac(frs, GACConfig(), init_mask=None), (sz, sz), interpolation=cv2.INTER_NEAREST)
        pu, pd, ps = p_unet(un, rgb), p_dals(da, rgb), p_snake(sn, rgb)
        pp = R._snake_predict_paper(pipe, fr, sz, device, conf_threshold=0.05) * 255
        cells = [("RGB", bgr), ("GT", overlay(bgr, gt, BLUE)),
                 (f"Kass(orc) D{score(kass,gts)[1]:.2f}", overlay(bgrs, kass, RED)),
                 (f"GAC(orc) D{score(gac,gts)[1]:.2f}", overlay(bgrs, gac, RED)),
                 (f"U-Net D{score(pu,gt)[1]:.2f}", overlay(bgr, pu, RED)),
                 (f"DALS D{score(pd,gt)[1]:.2f}", overlay(bgr, pd, RED)),
                 (f"DSnake-s D{score(ps,gt)[1]:.2f}", overlay(bgr, ps, RED)),
                 (f"DSnake-p D{score(pp,gt)[1]:.2f}", overlay(bgr, pp, RED))]
        cols.append(col(cells, sz))
    cv2.imwrite(str(OUT/"all_methods.png"), hstack_sep(cols)); print("all_methods.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
    only = ap.parse_args().only
    figs = {"A": fig_flame1_gallery, "B": fig_flame3_gallery, "D": fig_flame3_failures, "E": fig_flame1_unet, "F": fig_region_vs_contour,
            "G": fig_paper_snake, "H": fig_init_study, "I": fig_oracle_init, "J": fig_all_methods}
    for k, fn in figs.items():
        if only and k != only: continue
        try: fn()
        except Exception as e:
            import traceback; print(f"FIG {k} FAILED:", e); traceback.print_exc()
