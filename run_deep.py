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
import torchvision.models as models

from flame.data import DEFAULT_DATASET, DEFAULT_THRESHOLD_C, load_frame
from flame.contour_utils import polygon_to_mask
from flame.metrics import dice, iou
from flame.splits import make_splits
from flame.deep.dataset import NET_SIZE, FlameDataset, SnakeDataset, PaperSnakeDataset
from flame.deep.losses import bce_dice, cyclic_contour_loss, dice_loss, snake_contour_loss
from flame.deep.unet import UNet
from flame.deep.dals import DALS
from flame.deep.deep_snake_simplified import DeepSnake
from flame.deep.deep_snake import DeepSnakePaper, DeepSnakePipeline

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
N_POINTS = 128
MIN_CC_PX = 30


def _prefix(dataset: str) -> str:
    """Checkpoint/CSV filename prefix so FLAME-1 runs don't clobber FLAME-3.

    FLAME-3 ('') keeps the existing unprefixed names (back-compat with the
    recorded results); FLAME-1 gets 'flame1_'.
    """
    return "" if dataset == "flame3" else f"{dataset}_"

class DeepSnakeTrainWrapper(torch.nn.Module):
    """
    A lightweight wrapper strictly for training the Snake head.
    Provides deep features without the overhead of an object detector.
    """
    def __init__(self, snake_feat_dim=64):
        super().__init__()
        # 1. Backbone (ResNet18) truncated at layer1 -> STRIDE 4, 64 channels.
        # The full backbone (-2) is stride 32: a 512px image -> 16x16 map, so an
        # ~80px fire box covers ~2 feature cells and all 128 contour vertices
        # sample (nearly) the same feature -> the snake can't tell vertices apart
        # and learns ~0 offsets. Stride 4 gives a 128x128 map so vertices around
        # a blob get distinct features (this is what the Deep Snake paper uses).
        resnet = models.resnet18(pretrained=True)
        self.backbone = torch.nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)

        # 2. Channel reducer (layer1 outputs 64 channels at stride 4).
        self.reduce = torch.nn.Conv2d(64, snake_feat_dim, kernel_size=1)

        # 3. The decoupled Snake module
        self.snake = DeepSnakePaper(feat_dim=snake_feat_dim)

    def extract_features(self, x):
        """Used in compute_loss to get the feature map"""
        feat = self.backbone(x)
        return self.reduce(feat)
        
    def forward(self, image, boxes, image_size):
        """Optional: allows direct calling if needed"""
        feat = self.extract_features(image)
        return self.snake(feat, boxes, image_size)

def build_model(method: str) -> torch.nn.Module:
    return {"unet": UNet, "dals": DALS, "deep_snake_simple": DeepSnake, "deep_snake_paper": DeepSnakeTrainWrapper}[method]()


def image_tensor(rgb: np.ndarray, size: int, device) -> torch.Tensor:
    r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(r).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


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

def _snake_predict_paper(model, frame, size, device, conf_threshold: float = 0.3) -> np.ndarray:
    """RGB-only Deep Snake inference (Handles both Oracle and System evaluation)."""
    x = image_tensor(frame.rgb, size, device)

    # Check which model we are currently running
    if isinstance(model, DeepSnakePipeline):
        # PHASE 2: SYSTEM EVALUATION
        # The pipeline handles MMDet box predictions internally
        contours, _ = model(x, conf_threshold=conf_threshold)
    else:
        # PHASE 1: ORACLE EVALUATION (per-instance) -- one GT box per fire
        # connected component, matching how the per-instance snake was trained.
        h, w = frame.gt_mask.shape
        gt_rs = cv2.resize(frame.gt_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        num, _, stats, _ = cv2.connectedComponentsWithStats(
            (gt_rs > 0).astype(np.uint8), connectivity=8)
        comp_boxes = [[x, y, x + bw, y + bh]
                      for lbl in range(1, num)
                      for (x, y, bw, bh, area) in [stats[lbl]]
                      if area >= MIN_CC_PX]
        if not comp_boxes:
            return np.zeros((size, size), dtype=np.uint8)
        boxes = torch.tensor(comp_boxes, device=device).float()   # [K, 4]
        contours = model(x, boxes, image_size=(size, size))
    
    # Draw the final contours onto the mask
    mask = np.zeros((size, size), dtype=np.uint8)
    if len(contours) == 0:
        return mask
        
    final_contours = contours[-1].detach().cpu().numpy()
    for poly in final_contours:
        if cv2.contourArea(poly.astype(np.float32)) >= MIN_CC_PX:
            mask = np.maximum(mask, polygon_to_mask(poly, (size, size)))
            
    return mask

def _detector_boxes(pipe, x, conf_threshold: float) -> np.ndarray:
    """Per-instance fire boxes [K,4] (x0,y0,x1,y1) at network resolution from the
    pipeline's CenterNet detector (no snake)."""
    _, _, H, W = x.shape
    feats = pipe.detector.extract_feat(x)
    chp, whp, offp = pipe.detector.bbox_head(feats)
    res = pipe.detector.bbox_head.predict_by_feat(
        chp, whp, offp,
        batch_img_metas=[{"img_shape": (H, W), "batch_input_shape": (H, W),
                          "border": (0, H, 0, W), "scale_factor": (1., 1.)}],
        rescale=False, with_nms=False)
    inst = res[0]
    keep = inst.scores.cpu().numpy() > conf_threshold
    return inst.bboxes.cpu().numpy()[keep]


def _gac_predict_from_boxes(pipe, frame, size, device, conf_threshold: float) -> np.ndarray:
    """deep_snake_gac: detector finds per-instance boxes, then a CLASSICAL geodesic
    active contour (no learning) evolves a box-seeded level set to the fire
    boundary on the fire-energy map. Tests whether classical evolution beats the
    learned offset-snake from the same per-instance boxes."""
    from flame.gac import run_gac, GACConfig
    x = image_tensor(frame.rgb, size, device)
    boxes = _detector_boxes(pipe, x, conf_threshold)
    if len(boxes) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    # Seed: a filled rectangle per detected box (at network res), eroded slightly
    # so GAC grows out to the true boundary rather than starting over-sized.
    seed = np.zeros((size, size), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes.astype(int):
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 > x0 and y1 > y0:
            seed[y0:y1, x0:x1] = 255
    # Run GAC on a size-resolution copy of the frame. fire_energy needs the RGB at
    # the same resolution as the seed.
    from flame.data import Frame
    rgb_rs = cv2.resize(frame.rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    f_rs = Frame(frame_id=frame.frame_id, rgb=rgb_rs, thermal_c=None,
                 gt_mask=np.zeros((size, size), np.uint8))
    out = run_gac(f_rs, GACConfig(dilate_factor=1.0), init_mask=seed)
    return (out > 0).astype(np.uint8)


@torch.no_grad()
def predict_native(method: str, model, frame, size: int, device,
                   conf_threshold: float = 0.3) -> np.ndarray:
    """Predicted 0/255 mask at the frame's native resolution."""
    x = image_tensor(frame.rgb, size, device)
    if method == "unet":
        logits = model(x)
        net_mask = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    elif method == "dals":
        phi, _ = model(x)
        net_mask = (torch.sigmoid(phi)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    elif method == "deep_snake_simple":
        net_mask = _snake_predict(model, x, size, device) * 255
    elif method == "deep_snake_paper":
        net_mask = _snake_predict_paper(model, frame, size, device, conf_threshold) * 255
    elif method == "deep_snake_gac":
        net_mask = _gac_predict_from_boxes(model, frame, size, device, conf_threshold) * 255
    h, w = frame.gt_mask.shape
    return cv2.resize(net_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def evaluate_split(method: str, model, frame_ids: list[str], threshold_c: float,
                   size: int, device, write_csv: bool = False,
                   dataset: str = DEFAULT_DATASET, conf_threshold: float = 0.3) -> dict:
    model.eval()
    rows = []
    for fid in frame_ids:
        frame = load_frame(fid, threshold_c=threshold_c, dataset=dataset)
        t0 = time.perf_counter()
        pred = predict_native(method, model, frame, size, device, conf_threshold)
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
        out = RESULTS_DIR / f"{_prefix(dataset)}{method}_per_frame.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  per-frame scores -> {out}")
    return {"iou": float(ious.mean()), "dice": float(dices.mean()), "n": len(rows)}


def compute_loss(method: str, model, batch, device,
                 chamfer_weight: float = 0.05) -> torch.Tensor:
    image = batch["image"].to(device)
    if method == "deep_snake_paper":
        # PER-INSTANCE: PaperSnakeDataset gives one tight GT box + contour per
        # fire component (no full-frame mask). The snake refines each box's
        # octagon toward that component's boundary.
        _, _, H, W = image.shape
        boxes = batch["box"].to(device).float()          # [B, 4] x0,y0,x1,y1
        gt = batch["gt_contour"].to(device)              # [B, N, 2]
        features = model.extract_features(image)
        contours = model.snake(features, boxes, image_size=(W, H))
        # cyclic L1 + Chamfer (anti-collapse). contours[0] is the fixed octagon
        # init; skip it so the loss only supervises the deformed iterations.
        loss = 0
        for c in contours[1:]:
            loss = loss + snake_contour_loss(c, gt, chamfer_weight=chamfer_weight)
        return loss

    mask = batch["mask"].to(device)
    if method == "unet":
        return bce_dice(model(image), mask)
    if method == "dals":
        phi, prob_logits = model(image)
        return dice_loss(phi, mask) + bce_dice(prob_logits, mask)

    coarse, contours = model(image, batch["init_contour"].to(device))
    gt = batch["gt_contour"].to(device)
    loss = bce_dice(coarse, mask)
    for c in contours:
        loss = loss + cyclic_contour_loss(c, gt)
    
    return loss


def train(method: str, args) -> None:
    device = torch.device(args.device)
    splits = make_splits(dataset=args.dataset)
    train_ids, val_ids = splits["train"], splits["val"]
    if args.limit:
        train_ids, val_ids = train_ids[: args.limit], val_ids[: max(1, args.limit // 4)]

    if method == "deep_snake_paper":
        ds_cls = PaperSnakeDataset          # one sample per fire component
    elif method == "deep_snake_simple":
        ds_cls = SnakeDataset               # one sample per frame (largest component)
    else:
        ds_cls = FlameDataset
    train_ds = ds_cls(train_ids, threshold_c=args.threshold_c, size=args.size,
                      augment=True, dataset=args.dataset)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"{method} [{args.dataset}]: {len(train_ds)} train frames, {len(val_ids)} val frames, device={device}")

    model = build_model(method).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_iou = -1.0
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = MODELS_DIR / f"{_prefix(args.dataset)}{method}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = compute_loss(method, model, batch, device,
                                chamfer_weight=args.chamfer_weight)
            loss.backward()
            opt.step()
            running += loss.item()
        val = evaluate_split(method, model, val_ids, args.threshold_c, args.size, device,
                             dataset=args.dataset)
        print(f"  epoch {epoch:3d}  loss={running / max(len(loader), 1):.4f}  "
              f"val IoU={val['iou']:.4f}  Dice={val['dice']:.4f}")
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(model.state_dict(), ckpt)
    print(f"  best val IoU={best_iou:.4f}  ->  {ckpt}")


def run_eval(method: str, args) -> None:
    device = torch.device(args.device)
    if method == "deep_snake_gac":
        # No trained model: detector (for per-instance boxes) + classical GAC.
        print("Initializing detector for deep_snake_gac (classical contour, no snake)...")
        model = DeepSnakePipeline(config_file=args.mmdet_config,
                                  checkpoint_file=args.mmdet_checkpoint,
                                  snake_feat_dim=64, device=device).to(device)
        test_ids = make_splits(dataset=args.dataset)["test"]
        if args.limit:
            test_ids = test_ids[: args.limit]
        print(f"=== {_prefix(args.dataset)}DEEP_SNAKE_GAC  eval on {len(test_ids)} test frames ===")
        summary = evaluate_split(method, model, test_ids, args.threshold_c, args.size,
                                 device, write_csv=True, dataset=args.dataset,
                                 conf_threshold=args.conf_threshold)
        print(f"  test IoU mean={summary['iou']:.4f}  Dice mean={summary['dice']:.4f}")
        return

    ckpt = MODELS_DIR / f"{_prefix(args.dataset)}{method}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}; train first.")
    if method == "deep_snake_paper":
        print("Initializing DeepSnake MMDet Pipeline for Evaluation...")
        model = DeepSnakePipeline(
            config_file=args.mmdet_config, 
            checkpoint_file=args.mmdet_checkpoint,
            snake_feat_dim=64,
            device=device
        ).to(device)
        
        # The training wrapper stores the snake under `self.snake`, so its keys
        # are prefixed 'snake.'. Strip ONLY that leading prefix (not a global
        # replace, which would also mangle 'snake...' substrings elsewhere).
        trained_weights = torch.load(ckpt, map_location=device, weights_only=True)
        snake_weights = {k[len('snake.'):]: v for k, v in trained_weights.items()
                         if k.startswith('snake.')}
        model.snake.load_state_dict(snake_weights)
        
    else:
        model = build_model(method).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    test_ids = make_splits(dataset=args.dataset)["test"]
    if args.limit:
        test_ids = test_ids[: args.limit]
    print(f"=== {_prefix(args.dataset)}{method.upper()}  eval on {len(test_ids)} test frames ===")
    summary = evaluate_split(method, model, test_ids, args.threshold_c, args.size,
                             device, write_csv=True, dataset=args.dataset,
                             conf_threshold=args.conf_threshold)
    print(f"  test IoU mean={summary['iou']:.4f}  Dice mean={summary['dice']:.4f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["unet", "dals", "deep_snake_simple", "deep_snake_paper",
                                          "deep_snake_gac"], required=True)
    ap.add_argument("--dataset", choices=["flame3", "flame1"], default=DEFAULT_DATASET,
                    help="flame3 (thermal GT) or flame1 (hand-labeled PNG masks)")
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=NET_SIZE)
    ap.add_argument("--threshold-c", type=float, default=DEFAULT_THRESHOLD_C)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="Subset frames for a quick run")
    ap.add_argument("--mmdet-config", type=str, default="flame/deep/centernet_flame.py",
                    help="Path to the MMDet CenterNet config")
    ap.add_argument("--mmdet-checkpoint", type=str, default="models/centernet.pth",
                    help="Path to trained MMDet CenterNet weights")
    ap.add_argument("--conf-threshold", type=float, default=0.3,
                    help="CenterNet box confidence threshold for deep_snake_paper eval "
                         "(lower = higher recall, more frames get a prediction)")
    ap.add_argument("--chamfer-weight", type=float, default=0.05,
                    help="Weight of the Chamfer (anti-collapse) term in the "
                         "deep_snake_paper contour loss")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args.method, args)
    else:
        run_eval(args.method, args)


if __name__ == "__main__":
    main()
