"""Visualize deep_snake_paper outputs to show WHY it scores low.

For a set of example frames, render a row: RGB | GT mask | detector boxes (with
max confidence) | final snake prediction. Saves a PNG per frame + a montage.
"""
import sys
import numpy as np
import cv2
import torch

from code.flame.data import load_frame
from code.flame.deep.deep_snake import DeepSnakePipeline
from code.flame.contour_utils import polygon_to_mask
from code.flame.metrics import iou as iou_fn

SIZE = 512
DET_CFG = "flame/deep/centernet_flame1.py"
DET_CKPT = "results/rerun_flame1_20260601_193349/centernet/epoch_10.pth"
SNAKE_CKPT = "models/flame1_deep_snake_paper.pt"

# Confidence threshold from argv (default 0.3). Output dir is threshold-tagged so
# different sweeps don't overwrite each other.
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
OUT_DIR = f"results/rerun_flame1_20260601_193349/viz_conf{CONF:.2f}"

EXAMPLES = {
    "image_1996": "fired, best IoU",
    "image_1997": "fired, good IoU",
    "image_2002": "fired, good IoU",
    "image_2001": "fired, good IoU",
    "image_1736": "detector MISS (large fire)",
    "image_1729": "detector MISS (large fire)",
    "image_1714": "detector MISS (large fire)",
    "image_1702": "fired but wrong place",
    "image_1703": "fired but wrong place",
    "image_1727": "fired but wrong place",
}


def image_tensor(rgb, size, device):
    r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(r).float().permute(2, 0, 1)[None].to(device) / 255.0


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"
    pipe = DeepSnakePipeline(DET_CFG, DET_CKPT, snake_feat_dim=64, device=device).to(device).eval()
    tw = torch.load(SNAKE_CKPT, map_location=device, weights_only=True)
    pipe.snake.load_state_dict({k[len("snake."):]: v for k, v in tw.items() if k.startswith("snake.")})

    panels = []
    for fid, label in EXAMPLES.items():
        frame = load_frame(fid, dataset="flame1")
        rgb = cv2.resize(frame.rgb, (SIZE, SIZE))
        gt = cv2.resize(frame.gt_mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        x = image_tensor(frame.rgb, SIZE, device)

        # Raw detector confidence (to explain misses): inspect predicted boxes pre-threshold.
        with torch.no_grad():
            feats = pipe.detector.extract_feat(x)
            chp, whp, offp = pipe.detector.bbox_head(feats)
            res = pipe.detector.bbox_head.predict_by_feat(
                chp, whp, offp,
                batch_img_metas=[{"img_shape": (SIZE, SIZE), "batch_input_shape": (SIZE, SIZE),
                                  "border": (0, SIZE, 0, SIZE), "scale_factor": (1., 1.)}],
                rescale=False, with_nms=False)
            scores = res[0].scores.cpu().numpy()
            boxes = res[0].bboxes.cpu().numpy()
            max_conf = float(scores.max()) if len(scores) else 0.0
            kept = boxes[scores > CONF]

            # Final prediction through the real pipeline.
            contours, _ = pipe(x, conf_threshold=CONF)
        pred = np.zeros((SIZE, SIZE), np.uint8)
        if len(contours):
            for poly in contours[-1].detach().cpu().numpy():
                if cv2.contourArea(poly.astype(np.float32)) >= 30:
                    pred = np.maximum(pred, polygon_to_mask(poly, (SIZE, SIZE)))
        score = iou_fn(pred * 255, gt)

        # Build a 4-panel row.
        def col(img, title):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img.copy()
            cv2.rectangle(img, (0, 0), (SIZE, 28), (0, 0, 0), -1)
            cv2.putText(img, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            return img

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        def overlay(base, mask, color):
            """Tint mask region + draw its outline on a copy of base (visible even
            for tiny fire blobs, unlike a small white speck on black)."""
            out = base.copy()
            m = (mask > 0).astype(np.uint8)
            tint = np.zeros_like(out); tint[m > 0] = color
            out = cv2.addWeighted(out, 1.0, tint, 0.5, 0)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, 2)
            return out

        det_vis = rgb_bgr.copy()
        for b in kept.astype(int):
            cv2.rectangle(det_vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        det_title = f"det boxes>{CONF:.2f}: {len(kept)} (max conf {max_conf:.2f})"

        row = np.hstack([
            col(rgb_bgr, f"{fid}: {label}"),
            col(overlay(rgb_bgr, gt, (255, 100, 0)), "GT (blue)"),   # GT in blue
            col(det_vis, det_title),
            col(overlay(rgb_bgr, pred * 255, (0, 0, 255)),           # pred in red
                f"snake pred red (IoU {score:.3f})"),
        ])
        out = f"{OUT_DIR}/{fid}.png"
        cv2.imwrite(out, row)
        panels.append(row)
        print(f"{fid}: max_conf={max_conf:.3f} boxes>0.3={len(kept)} pred_px={int((pred>0).sum())} IoU={score:.3f}  -> {out}")

    # Lay out as a grid: 2 example-rows per montage row (so 10 examples -> 5x2).
    per_row = 2
    grid_rows = []
    for i in range(0, len(panels), per_row):
        chunk = panels[i:i + per_row]
        while len(chunk) < per_row:  # pad last row if odd
            chunk.append(np.zeros_like(panels[0]))
        grid_rows.append(np.hstack(chunk))
    montage = np.vstack(grid_rows)
    # Downscale so the attachment is a reasonable size.
    scale = min(1.0, 2600 / montage.shape[1])
    if scale < 1.0:
        montage = cv2.resize(montage, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUT_DIR}/montage.png", montage)
    print(f"montage -> {OUT_DIR}/montage.png  ({montage.shape[1]}x{montage.shape[0]})")


if __name__ == "__main__":
    main()
