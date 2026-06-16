"""Recompute FLAME-3 metrics with degenerate smoke-covered frames excluded
(smoke_fraction >= 0.5), and deep_snake_paper at the tuned conf 0.05.
Writes results/filtered_metrics.csv. FLAME-1 has no degenerate frames so its
numbers are unchanged (we recompute the snake variants there too for the table).
"""
import sys, csv
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flame.data import load_frame, smoke_fraction
from flame.splits import make_splits
from flame.metrics import iou as iou_fn, dice as dice_fn
from flame.deep.unet import UNet
from flame.deep.deep_snake_simplified import DeepSnake
from flame.deep.deep_snake import DeepSnakePipeline
import run_deep as R
from scripts.compute_paper_metrics import boundary_f1  # reuse

SIZE, device, SMOKE_TAU = 512, "cuda", 0.5


def keep_ids(ds):
    """Test ids with degenerate (smoke_fraction >= TAU) frames removed."""
    ids = make_splits(dataset=ds)["test"]
    kept = [f for f in ids if smoke_fraction(load_frame(f, dataset=ds).rgb) < SMOKE_TAU]
    return ids, kept


@torch.no_grad()
def score(predict, ids, ds):
    I = D = B = 0.0
    for fid in ids:
        fr = load_frame(fid, dataset=ds)
        p = predict(fr)
        I += iou_fn(p, fr.gt_mask); D += dice_fn(p, fr.gt_mask); B += boundary_f1(p, fr.gt_mask)
    n = len(ids)
    return I/n, D/n, B/n


def main():
    rows = []
    # ---- FLAME-3: filtered ----
    all3, kept3 = keep_ids("flame3")
    print(f"FLAME-3: {len(all3)} -> {len(kept3)} after dropping smoke>= {SMOKE_TAU} "
          f"({len(all3)-len(kept3)} removed)")
    un3 = UNet().to(device).eval(); un3.load_state_dict(torch.load(ROOT/"models/unet.pt", map_location=device, weights_only=True))
    sn3 = DeepSnake().to(device).eval(); sn3.load_state_dict(torch.load(ROOT/"models/deep_snake_simple.pt", map_location=device, weights_only=True))
    def u3(fr):
        x=R.image_tensor(fr.rgb,SIZE,device); m=(torch.sigmoid(un3(x))[0,0].cpu().numpy()>0.5).astype(np.uint8)*255
        h,w=fr.gt_mask.shape; return cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
    def s3(fr):
        x=R.image_tensor(fr.rgb,SIZE,device); m=R._snake_predict(sn3,x,SIZE,device)*255
        h,w=fr.gt_mask.shape; return cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
    for name, fn in [("unet", u3), ("deep_snake_simple", s3)]:
        rows.append(("flame3_filtered", name, *score(fn, kept3, "flame3")))

    # ---- FLAME-1: snake variants (no smoke frames; paper at tuned conf 0.05) ----
    all1, kept1 = keep_ids("flame1")
    print(f"FLAME-1: {len(all1)} -> {len(kept1)} (no degenerate frames expected)")
    sn1 = DeepSnake().to(device).eval(); sn1.load_state_dict(torch.load(ROOT/"models/flame1_deep_snake_simple.pt", map_location=device, weights_only=True))
    def s1(fr):
        x=R.image_tensor(fr.rgb,SIZE,device); m=R._snake_predict(sn1,x,SIZE,device)*255
        h,w=fr.gt_mask.shape; return cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
    rows.append(("flame1", "deep_snake_simple", *score(s1, kept1, "flame1")))

    pipe = DeepSnakePipeline(str(ROOT/"flame/deep/centernet_flame1.py"),
                             str(ROOT/"results/rerun_flame1_20260601_193349/centernet/epoch_10.pth"),
                             snake_feat_dim=64, device=device).to(device).eval()
    tw = torch.load(ROOT/"models/flame1_deep_snake_paper.pt", map_location=device, weights_only=True)
    pipe.snake.load_state_dict({k[6:]: v for k,v in tw.items() if k.startswith("snake.")})
    def p1(fr):
        m=R._snake_predict_paper(pipe, fr, SIZE, device, conf_threshold=0.05)*255
        h,w=fr.gt_mask.shape; return cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
    rows.append(("flame1", "deep_snake_paper@0.05", *score(p1, kept1, "flame1")))

    out = ROOT/"results"/"filtered_metrics.csv"
    with open(out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["dataset","method","iou","dice","bf_2px"])
        for r in rows:
            w.writerow([r[0],r[1],f"{r[2]:.4f}",f"{r[3]:.4f}",f"{r[4]:.4f}"])
            print(f"  {r[0]:18s} {r[1]:22s} IoU {r[2]:.3f} Dice {r[3]:.3f} BF {r[4]:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
