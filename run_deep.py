"""Train / evaluate the deep baselines (U-Net, DALS, Deep Snake) on FLAME-3.

RGB-only input; ground truth is the thermal-thresholded mask. Scored with IoU
and Dice at native resolution, matching run_snakes.py.

    python run_deep.py --method unet       --mode train --epochs 50
    python run_deep.py --method dals       --mode eval
    python run_deep.py --method deep_snake --mode train --epochs 50

Checkpoints -> models/<method>.pt ; per-frame scores -> results/<method>_per_frame.csv.
Requires torch + torchvision (see requirements.txt); no GPU is required but CPU
training is slow.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from flame.data import DEFAULT_THRESHOLD_C, load_frame
from flame.contour_utils import polygon_to_mask
from flame.metrics import dice, iou
from flame.splits import make_splits
from flame.deep.dataset import NET_SIZE, FlameDataset, SnakeDataset, ThermalDataset
from flame.deep.losses import (
    bce_dice, bce_focal_tversky, cyclic_contour_loss, dice_loss, focal_tversky,
    weighted_thermal_loss,
)
from flame.deep.unet import UNet
from flame.deep.dals import DALS
from flame.deep.deep_snake import DeepSnake
from flame.deep.thermal_reg import ThermalRegUNet, THR_NORM
from flame.deep.channels import SPEC_CHANNELS

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
N_POINTS = 128
MIN_CC_PX = 30


def build_model(method: str, in_ch: int = 3) -> torch.nn.Module:
    if method == "unet":
        return UNet(in_ch=in_ch)
    return {"dals": DALS, "deep_snake": DeepSnake, "thermal": ThermalRegUNet}[method]()


def image_tensor(rgb: np.ndarray, size: int, device, in_channels: str = "rgb") -> torch.Tensor:
    from flame.deep.channels import build_input
    r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    feat = build_input(r, in_channels)  # [H,W,C] float32 in [0,1]
    return torch.from_numpy(feat).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def _circle(cx: float, cy: float, r: float, n: int) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1).astype(np.float32)


def _snake_predict(model, x, size, device) -> np.ndarray:
    """RGB-only Deep Snake inference: coarse seg -> per-CC circle init -> deform -> union."""
    coarse, _ = model(x, None)
    prob = (torch.sigmoid(coarse)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(prob, connectivity=8)
    mask = np.zeros((size, size), dtype=np.uint8)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] < MIN_CC_PX:
            continue
        cx, cy = centroids[lbl]
        r = max(np.sqrt(stats[lbl, cv2.CC_STAT_AREA] / np.pi), 2.0)
        init = torch.from_numpy(_circle(cx, cy, r, N_POINTS)).unsqueeze(0).to(device)
        _, contours = model(x, init)
        poly = contours[-1][0].detach().cpu().numpy()
        mask = np.maximum(mask, polygon_to_mask(poly, (size, size)))
    return mask


@torch.no_grad()
def predict_native(method: str, model, frame, size: int, device,
                   in_channels: str = "rgb") -> np.ndarray:
    """Predicted 0/255 mask at the frame's native resolution."""
    x = image_tensor(frame.rgb, size, device, in_channels)
    if method == "unet":
        logits = model(x)
        net_mask = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    elif method == "dals":
        phi, _ = model(x)
        net_mask = (torch.sigmoid(phi)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    elif method == "thermal":
        t_hat = model(x)  # normalized temperature field in [0,1]
        net_mask = (t_hat[0, 0].cpu().numpy() >= THR_NORM).astype(np.uint8) * 255
    else:
        net_mask = _snake_predict(model, x, size, device) * 255
    h, w = frame.gt_mask.shape
    return cv2.resize(net_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def evaluate_split(method: str, model, frame_ids: list[str], threshold_c: float,
                   size: int, device, write_csv: bool = False, tag: str = "",
                   in_channels: str = "rgb") -> dict:
    model.eval()
    rows = []
    for fid in frame_ids:
        frame = load_frame(fid, threshold_c=threshold_c)
        t0 = time.perf_counter()
        pred = predict_native(method, model, frame, size, device, in_channels)
        latency = time.perf_counter() - t0
        rows.append({
            "frame": fid,
            "iou": iou(pred, frame.gt_mask),
            "dice": dice(pred, frame.gt_mask),
            "gt_px": int((frame.gt_mask > 0).sum()),
            "pred_px": int((pred > 0).sum()),
            "latency_s": latency,
        })
    ious = np.array([r["iou"] for r in rows])
    dices = np.array([r["dice"] for r in rows])
    if write_csv:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"{method}{tag}_per_frame.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  per-frame scores -> {out}")
    return {"iou": float(ious.mean()), "dice": float(dices.mean()), "n": len(rows)}


def compute_loss(method: str, model, batch, device, loss: str = "bce_dice") -> torch.Tensor:
    seg = bce_focal_tversky if loss == "focal_tversky" else bce_dice
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    if method == "thermal":
        t_hat = model(image)
        return weighted_thermal_loss(t_hat, batch["thermal"].to(device), THR_NORM)
    if method == "unet":
        return seg(model(image), mask)
    if method == "dals":
        phi, prob_logits = model(image)
        phi_loss = focal_tversky(phi, mask) if loss == "focal_tversky" else dice_loss(phi, mask)
        return phi_loss + seg(prob_logits, mask)
    coarse, contours = model(image, batch["init_contour"].to(device))
    gt = batch["gt_contour"].to(device)
    loss = bce_dice(coarse, mask)
    for c in contours:
        loss = loss + cyclic_contour_loss(c, gt)
    return loss


def train(method: str, args) -> None:
    device = torch.device(args.device)
    splits = make_splits(exclude_occluded=args.exclude_occluded)
    train_ids, val_ids = splits["train"], splits["val"]
    if args.limit:
        train_ids, val_ids = train_ids[: args.limit], val_ids[: max(1, args.limit // 4)]

    # Engineered input channels apply to the per-pixel U-Net only.
    spec = args.in_channels if method == "unet" else "rgb"
    in_ch = SPEC_CHANNELS[spec]
    ds_cls = {"deep_snake": SnakeDataset, "thermal": ThermalDataset}.get(method, FlameDataset)
    if ds_cls is FlameDataset:
        ds_kwargs = {"in_channels": spec, "aug_mode": args.augment}
    elif ds_cls is SnakeDataset:
        ds_kwargs = {"aug_mode": args.augment}   # SnakeDataset honours aug_mode, not in_channels
    else:
        ds_kwargs = {}                            # ThermalDataset: light aug only
    train_ds = ds_cls(train_ids, threshold_c=args.threshold_c, size=args.size,
                      augment=True, **ds_kwargs)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"{method}: {len(train_ds)} train frames, {len(val_ids)} val frames, device={device}")

    model = build_model(method, in_ch=in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_iou = -1.0
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = MODELS_DIR / f"{method}{args.tag}.pt"
    print(f"  loss={args.loss}  in_channels={spec}({in_ch})  checkpoint={ckpt.name}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = compute_loss(method, model, batch, device, args.loss)
            loss.backward()
            opt.step()
            running += loss.item()
        val = evaluate_split(method, model, val_ids, args.threshold_c, args.size, device,
                             in_channels=spec)
        print(f"  epoch {epoch:3d}  loss={running / max(len(loader), 1):.4f}  "
              f"val IoU={val['iou']:.4f}  Dice={val['dice']:.4f}")
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(model.state_dict(), ckpt)
    print(f"  best val IoU={best_iou:.4f}  ->  {ckpt}")


def run_eval(method: str, args) -> None:
    device = torch.device(args.device)
    ckpt = MODELS_DIR / f"{method}{args.tag}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}; train first.")
    spec = args.in_channels if method == "unet" else "rgb"
    model = build_model(method, in_ch=SPEC_CHANNELS[spec]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    test_ids = make_splits(exclude_occluded=args.exclude_occluded)["test"]
    if args.limit:
        test_ids = test_ids[: args.limit]
    print(f"=== {method.upper()}{args.tag}  eval on {len(test_ids)} test frames "
          f"(in_channels={spec}) ===")
    summary = evaluate_split(method, model, test_ids, args.threshold_c, args.size,
                             device, write_csv=True, tag=args.tag, in_channels=spec)
    print(f"  test IoU mean={summary['iou']:.4f}  Dice mean={summary['dice']:.4f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["unet", "dals", "deep_snake", "thermal"], required=True)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=NET_SIZE)
    ap.add_argument("--threshold-c", type=float, default=DEFAULT_THRESHOLD_C)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="Subset frames for a quick run")
    ap.add_argument("--loss", choices=["bce_dice", "focal_tversky"], default="bce_dice",
                    help="Segmentation loss for unet/dals (deep_snake unaffected)")
    ap.add_argument("--tag", default="",
                    help="Suffix for checkpoint/CSV names so variants don't overwrite "
                         "the baseline, e.g. --tag _ft")
    ap.add_argument("--in-channels", dest="in_channels", default="rgb",
                    choices=list(SPEC_CHANNELS), help="Input channels for U-Net "
                    "(rgb | rgb_rg | rgb_hsv_rg); other methods use rgb")
    ap.add_argument("--augment", choices=["light", "medium", "strong"], default="light",
                    help="light = flip only; medium = small rot/scale/shift + mild "
                         "brightness/contrast (no hue); strong = aggressive geo+photometric")
    ap.add_argument("--exclude-occluded", dest="exclude_occluded", action="store_true",
                    help="Drop full-frame-smoke (>=0.95) frames from train/val/test "
                         "as degenerate, unsolvable inputs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args.method, args)
    else:
        run_eval(args.method, args)


if __name__ == "__main__":
    main()
