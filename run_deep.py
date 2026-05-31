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

from flame.data import DEFAULT_THRESHOLD_C, load_frame
from flame.contour_utils import polygon_to_mask
from flame.metrics import dice, iou
from flame.splits import make_splits
from flame.deep.dataset import NET_SIZE, FlameDataset, SnakeDataset
from flame.deep.losses import bce_dice, cyclic_contour_loss, dice_loss
from flame.deep.unet import UNet
from flame.deep.dals import DALS
from flame.deep.deep_snake_simplified import DeepSnake
from flame.deep.deep_snake import DeepSnakePaper, DeepSnakePipeline

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
N_POINTS = 128
MIN_CC_PX = 30

class DeepSnakeTrainWrapper(torch.nn.Module):
    """
    A lightweight wrapper strictly for training the Snake head.
    Provides deep features without the overhead of an object detector.
    """
    def __init__(self, snake_feat_dim=64):
        super().__init__()
        # 1. Standard Backbone (ResNet18)
        resnet = models.resnet18(pretrained=True)
        
        # Strip the classification head and pooling to keep spatial dimensions
        # Outputs a feature map of shape [B, 512, H/32, W/32]
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2]) 
        
        # 2. Channel Reducer (matches the ResNet output to the Snake's expected input)
        self.reduce = torch.nn.Conv2d(512, snake_feat_dim, kernel_size=1)
        
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

def _snake_predict_paper(model, frame, size, device) -> np.ndarray:
    """RGB-only Deep Snake inference (Handles both Oracle and System evaluation)."""
    x = image_tensor(frame.rgb, size, device)
    
    # Check which model we are currently running
    if isinstance(model, DeepSnakePipeline):
        # PHASE 2: SYSTEM EVALUATION
        # The pipeline handles MMDet box predictions internally
        contours, _ = model(x, conf_threshold=0.3) 
    else:
        # PHASE 1: ORACLE EVALUATION (Training Validation)
        # We manually extract perfect ground truth boxes from the mask
        pos = np.where(frame.gt_mask > 0)
        if len(pos[0]) == 0:
            return np.zeros((size, size), dtype=np.uint8)
            
        ymin, xmin = pos[0].min(), pos[1].min()
        ymax, xmax = pos[0].max(), pos[1].max()
        
        h, w = frame.gt_mask.shape
        scale_x, scale_y = size / w, size / h
        boxes = torch.tensor([[xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y]], device=device).float()
        
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

@torch.no_grad()
def predict_native(method: str, model, frame, size: int, device) -> np.ndarray:
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
        net_mask = _snake_predict_paper(model, frame, size, device) * 255
    h, w = frame.gt_mask.shape
    return cv2.resize(net_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def evaluate_split(method: str, model, frame_ids: list[str], threshold_c: float,
                   size: int, device, write_csv: bool = False) -> dict:
    model.eval()
    rows = []
    for fid in frame_ids:
        frame = load_frame(fid, threshold_c=threshold_c)
        t0 = time.perf_counter()
        pred = predict_native(method, model, frame, size, device)
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
        out = RESULTS_DIR / f"{method}_per_frame.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  per-frame scores -> {out}")
    return {"iou": float(ious.mean()), "dice": float(dices.mean()), "n": len(rows)}


def compute_loss(method: str, model, batch, device) -> torch.Tensor:
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    if method == "unet":
        return bce_dice(model(image), mask)
    if method == "dals":
        phi, prob_logits = model(image)
        return dice_loss(phi, mask) + bce_dice(prob_logits, mask)
    if method == "deep_snake_paper":
        B, _, H, W = image.shape
        
        # 1. Get ground truth bounding boxes directly from the mask
        boxes = []
        for i in range(B):
            pos = torch.where(mask[i, 0] > 0)
            if len(pos[0]) > 0:
                ymin, xmin = pos[0].min(), pos[1].min()
                ymax, xmax = pos[0].max(), pos[1].max()
                boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
            else:
                # Fallback if mask is completely empty
                boxes.append(torch.tensor([0, 0, W, H])) 
        boxes = torch.stack(boxes).to(device).float()
        
        # 2. Extract features (Assuming 'model' here is a wrapper that contains 
        # both your backbone and DeepSnakePaper, or just run the backbone first)
        features = model.extract_features(image) 
        
        # 3. Forward pass
        contours = model.snake(features, boxes, image_size=(W, H))
        
        gt = batch["gt_contour"].to(device)
        
        # 4. Loss calculation (No coarse mask loss anymore!)
        loss = 0
        for c in contours:
            loss = loss + cyclic_contour_loss(c, gt)
            
        return loss
    
    coarse, contours = model(image, batch["init_contour"].to(device))
    gt = batch["gt_contour"].to(device)
    loss = bce_dice(coarse, mask)
    for c in contours:
        loss = loss + cyclic_contour_loss(c, gt)
    
    return loss


def train(method: str, args) -> None:
    device = torch.device(args.device)
    splits = make_splits()
    train_ids, val_ids = splits["train"], splits["val"]
    if args.limit:
        train_ids, val_ids = train_ids[: args.limit], val_ids[: max(1, args.limit // 4)]

    ds_cls = SnakeDataset if method == "deep_snake_simple" or method == "deep_snake_paper" else FlameDataset
    train_ds = ds_cls(train_ids, threshold_c=args.threshold_c, size=args.size, augment=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"{method}: {len(train_ds)} train frames, {len(val_ids)} val frames, device={device}")

    model = build_model(method).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_iou = -1.0
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = MODELS_DIR / f"{method}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = compute_loss(method, model, batch, device)
            loss.backward()
            opt.step()
            running += loss.item()
        val = evaluate_split(method, model, val_ids, args.threshold_c, args.size, device)
        print(f"  epoch {epoch:3d}  loss={running / max(len(loader), 1):.4f}  "
              f"val IoU={val['iou']:.4f}  Dice={val['dice']:.4f}")
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(model.state_dict(), ckpt)
    print(f"  best val IoU={best_iou:.4f}  ->  {ckpt}")


def run_eval(method: str, args) -> None:
    device = torch.device(args.device)
    ckpt = MODELS_DIR / f"{method}.pt"
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
        
        # We only want the weights belonging to the 'snake' submodule
        trained_weights = torch.load(ckpt, map_location=device)
        snake_weights = {k.replace('snake.', ''): v for k, v in trained_weights.items() if k.startswith('snake.')}
        model.snake.load_state_dict(snake_weights)
        
    else:
        model = build_model(method).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    test_ids = make_splits()["test"]
    if args.limit:
        test_ids = test_ids[: args.limit]
    print(f"=== {method.upper()}  eval on {len(test_ids)} test frames ===")
    summary = evaluate_split(method, model, test_ids, args.threshold_c, args.size,
                             device, write_csv=True)
    print(f"  test IoU mean={summary['iou']:.4f}  Dice mean={summary['dice']:.4f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["unet", "dals", "deep_snake_simple", "deep_snake_paper"], required=True)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=NET_SIZE)
    ap.add_argument("--threshold-c", type=float, default=DEFAULT_THRESHOLD_C)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="Subset frames for a quick run")
    ap.add_argument("--mmdet-config", type=str, default="centernet_config.py", help="Path to MMDet config")
    ap.add_argument("--mmdet-checkpoint", type=str, default="centernet.pth", help="Path to MMDet weights")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args.method, args)
    else:
        run_eval(args.method, args)


if __name__ == "__main__":
    main()
