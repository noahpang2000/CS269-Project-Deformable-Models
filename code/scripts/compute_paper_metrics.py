"""Fill metric gaps for the paper: IoU / Dice / Boundary-F1@2px for the methods
not in results/boundary_summary.csv (the two new Deep Snake variants, JEPA U-Net),
on FLAME-1. Writes results/paper_metrics.csv (merged with the existing summary).
"""
import sys, csv
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from code.flame.data import load_frame
from code.flame.splits import make_splits
from code.flame.metrics import iou as iou_fn, dice as dice_fn
from code.flame.deep.unet import UNet
from code.flame.deep.deep_snake_simplified import DeepSnake
from code.flame.deep.deep_snake import DeepSnakePipeline
import code.run_deep as R

SIZE, device = 512, "cuda"


def boundary_f1(pred, gt, tol=2):
    """F1 of boundary pixels within `tol` px (Boundary-F metric)."""
    p = (pred > 0).astype(np.uint8); g = (gt > 0).astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    def boundary(m):
        er = cv2.erode(m, np.ones((3, 3), np.uint8))
        return m - er
    pb, gb = boundary(p), boundary(g)
    k = 2 * tol + 1
    gb_d = cv2.dilate(gb, np.ones((k, k), np.uint8))
    pb_d = cv2.dilate(pb, np.ones((k, k), np.uint8))
    prec = (pb * gb_d).sum() / (pb.sum() + 1e-9)
    rec = (gb * pb_d).sum() / (gb.sum() + 1e-9)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


@torch.no_grad()
def score(predict_fn, ids):
    ious, dices, bfs = [], [], []
    for fid in ids:
        fr = load_frame(fid, dataset="flame1")
        pred = predict_fn(fr)  # native-res 0/255
        ious.append(iou_fn(pred, fr.gt_mask))
        dices.append(dice_fn(pred, fr.gt_mask))
        bfs.append(boundary_f1(pred, fr.gt_mask))
    return np.mean(ious), np.mean(dices), np.mean(bfs)


def main():
    test = make_splits(dataset="flame1")["test"]

    def unet_predict(model):
        def f(fr):
            x = R.image_tensor(fr.rgb, SIZE, device)
            m = (torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            h, w = fr.gt_mask.shape
            return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return f

    rows = []

    # JEPA-pretrained U-Net
    m = UNet().to(device).eval(); m.load_state_dict(torch.load(ROOT/"models/flame1_unet_jepa.pt", map_location=device, weights_only=True))
    rows.append(("flame1", "unet_jepa", *score(unet_predict(m), test)))

    # Deep Snake (simple)
    s = DeepSnake().to(device).eval(); s.load_state_dict(torch.load(ROOT/"models/flame1_deep_snake_simple.pt", map_location=device, weights_only=True))
    def snake_simple(fr):
        x = R.image_tensor(fr.rgb, SIZE, device)
        m = R._snake_predict(s, x, SIZE, device) * 255
        h, w = fr.gt_mask.shape
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    rows.append(("flame1", "deep_snake_simple", *score(lambda fr: snake_simple(fr), test)))

    # Deep Snake (paper) via detector pipeline @ conf 0.10
    pipe = DeepSnakePipeline(str(ROOT/"flame/deep/centernet_flame1.py"),
                             str(ROOT/"results/rerun_flame1_20260601_193349/centernet/epoch_10.pth"),
                             snake_feat_dim=64, device=device).to(device).eval()
    tw = torch.load(ROOT/"models/flame1_deep_snake_paper.pt", map_location=device, weights_only=True)
    pipe.snake.load_state_dict({k[len("snake."):]: v for k, v in tw.items() if k.startswith("snake.")})
    def snake_paper(fr):
        m = R._snake_predict_paper(pipe, fr, SIZE, device, conf_threshold=0.10) * 255
        h, w = fr.gt_mask.shape
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    rows.append(("flame1", "deep_snake_paper", *score(lambda fr: snake_paper(fr), test)))

    out = ROOT / "results" / "paper_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "method", "iou", "dice", "bf_2px"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])
            print(f"{r[1]:20s} IoU={r[2]:.3f} Dice={r[3]:.3f} BF@2={r[4]:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
