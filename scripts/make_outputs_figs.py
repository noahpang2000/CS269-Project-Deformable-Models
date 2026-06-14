"""Qualitative outputs-vs-GT figures: each deep method's prediction on the same
FLAME-1 test frames, for the report. Outputs report/figs/outputs/."""
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figs" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

from flame.data import load_frame
from flame.splits import make_splits
from flame.deep.unet import UNet
from flame.deep.dals import DALS
from flame.deep.deep_snake_simplified import DeepSnake
from flame.deep.deep_snake import DeepSnakePipeline
from flame.contour_utils import polygon_to_mask
import run_deep as R

SIZE, MINCC, device = 512, 30, "cuda"
DET_CFG = str(ROOT / "flame/deep/centernet_flame1.py")
DET_CKPT = str(ROOT / "results/rerun_flame1_20260601_193349/centernet/epoch_10.pth")


def overlay(rgb_bgr, mask, color):
    out = rgb_bgr.copy()
    m = (mask > 0).astype(np.uint8)
    tint = np.zeros_like(out); tint[m > 0] = color
    out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def titlebar(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


@torch.no_grad()
def pred_unet(m, rgb):
    x = R.image_tensor(rgb, SIZE, device)
    return (torch.sigmoid(m(x))[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255


@torch.no_grad()
def pred_dals(m, rgb):
    x = R.image_tensor(rgb, SIZE, device)
    phi, _ = m(x)
    return (torch.sigmoid(phi)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255


@torch.no_grad()
def pred_snake_simple(m, rgb):
    x = R.image_tensor(rgb, SIZE, device)
    return R._snake_predict(m, x, SIZE, device) * 255


@torch.no_grad()
def pred_snake_paper(pipe, frame):
    # Full detector->snake pipeline at conf 0.10 (the threshold from the sweep).
    return R._snake_predict_paper(pipe, frame, SIZE, device, conf_threshold=0.10) * 255


def main():
    from flame.metrics import iou as iou_fn
    test = make_splits(dataset="flame1")["test"]
    # representative frames: a couple big fires, a couple sparse
    picks = [test[2], test[20], test[50], test[120]]

    unet = UNet().to(device).eval(); unet.load_state_dict(torch.load(ROOT/"models/flame1_unet.pt", map_location=device, weights_only=True))
    dals = DALS().to(device).eval(); dals.load_state_dict(torch.load(ROOT/"models/flame1_dals.pt", map_location=device, weights_only=True))
    snake = DeepSnake().to(device).eval(); snake.load_state_dict(torch.load(ROOT/"models/flame1_deep_snake_simple.pt", map_location=device, weights_only=True))
    # JEPA-pretrained U-Net (same UNet architecture, SSL-pretrained encoder).
    jepa = UNet().to(device).eval(); jepa.load_state_dict(torch.load(ROOT/"models/flame1_unet_jepa.pt", map_location=device, weights_only=True))
    # Deep Snake (paper): detector -> octagon -> snake pipeline.
    pipe = DeepSnakePipeline(DET_CFG, DET_CKPT, snake_feat_dim=64, device=device).to(device).eval()
    tw = torch.load(ROOT/"models/flame1_deep_snake_paper.pt", map_location=device, weights_only=True)
    pipe.snake.load_state_dict({k[len("snake."):]: v for k, v in tw.items() if k.startswith("snake.")})

    rows = []
    for fid in picks:
        fr = load_frame(fid, dataset="flame1")
        rgb = cv2.resize(fr.rgb, (SIZE, SIZE))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gt = cv2.resize(fr.gt_mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        pu = pred_unet(unet, rgb)
        pj = pred_unet(jepa, rgb)            # JEPA-pretrained U-Net
        pd = pred_dals(dals, rgb)
        ps = pred_snake_simple(snake, rgb)
        pp = pred_snake_paper(pipe, fr)      # Deep Snake (paper)

        def iou_at(p):
            return iou_fn(cv2.resize(p, (fr.gt_mask.shape[1], fr.gt_mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST), fr.gt_mask)
        cells = [
            titlebar(bgr, f"{fid}: RGB"),
            titlebar(overlay(bgr, gt, (255, 80, 0)), "GT"),
            titlebar(overlay(bgr, pu, (0, 0, 255)), f"U-Net  IoU {iou_at(pu):.2f}"),
            titlebar(overlay(bgr, pj, (0, 0, 255)), f"U-Net+JEPA  IoU {iou_at(pj):.2f}"),
            titlebar(overlay(bgr, pd, (0, 0, 255)), f"DALS  IoU {iou_at(pd):.2f}"),
            titlebar(overlay(bgr, ps, (0, 0, 255)), f"DeepSnake(s)  IoU {iou_at(ps):.2f}"),
            titlebar(overlay(bgr, pp, (0, 0, 255)), f"DeepSnake(p)  IoU {iou_at(pp):.2f}"),
        ]
        rows.append(np.hstack(cells))
    montage = np.vstack(rows)
    scale = min(1.0, 3400 / montage.shape[1])   # 7 columns -> allow wider output
    if scale < 1.0:
        montage = cv2.resize(montage, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "outputs_vs_gt.png"), montage)
    print("wrote", OUT / "outputs_vs_gt.png", montage.shape)


if __name__ == "__main__":
    main()
