"""Extract FLAME-2 frames from the paired video zip into a FLAME-1-style layout.

FLAME-2 ships as 7 paired (colorized-IR, RGB) MP4s inside
``data/FLAME2/#1-7) All Video Pairs.zip`` (Zip64/deflate64 -> needs 7z, not
unzip). There is no raw Celsius thermal, so we derive a thermal-hot GT mask from
the IR palette (flame.data.flame2_fire_mask) under the user-chosen
"resize-and-pair" policy: each timestamped IR/RGB pair is assumed co-registered,
the IR-derived mask is resized into the RGB frame, and any FOV mismatch is
accepted as label noise.

For each pair we:
  1. extract the two MP4s with 7z (one pair at a time, deleted after),
  2. decode RGB and IR at ~1 fps (synchronized by timestamp),
  3. derive the GT mask from each IR frame, resize it to the RGB frame,
  4. keep frames with non-empty GT (fire present, matching FLAME-3's fire subset),
  5. save  images/v{vid}_{idx}.jpg  Masks/v{vid}_{idx}.png  ir/v{vid}_{idx}.jpg .

Run:  python scripts/extract_flame2.py            # all 7 pairs, 1 fps
      python scripts/extract_flame2.py --fps 1 --videos 6 7
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from code.flame.data import (FLAME2_DIR, FLAME2_RGB_DIR, FLAME2_MASK_DIR,
                        FLAME2_IR_DIR, flame2_fire_mask)

ZIP = FLAME2_DIR / "#1-7) All Video Pairs.zip"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _decode(video: Path, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}",
          str(out_dir / "f_%06d.png")])
    return sorted(out_dir.glob("f_*.png"))


def process_pair(vid: int, fps: float, jpg_quality: int = 95) -> tuple[int, int]:
    with tempfile.TemporaryDirectory(dir=FLAME2_DIR) as td:
        td = Path(td)
        # 1) pull the two MP4s for this pair out of the zip.
        _run(["7z", "e", "-y", f"-o{td}", str(ZIP),
              f"#{vid}) IR Video {vid}.MP4", f"#{vid}) RGB Video {vid}.MP4"])
        ir_mp4 = td / f"#{vid}) IR Video {vid}.MP4"
        rgb_mp4 = td / f"#{vid}) RGB Video {vid}.MP4"
        # 2) decode both at fps, synchronized by timestamp.
        ir_frames = _decode(ir_mp4, td / "ir", fps)
        rgb_frames = _decode(rgb_mp4, td / "rgb", fps)
        n = min(len(ir_frames), len(rgb_frames))
        kept = 0
        for i in range(n):
            ir = cv2.imread(str(ir_frames[i]))
            rgb = cv2.imread(str(rgb_frames[i]))
            if ir is None or rgb is None:
                continue
            mask = flame2_fire_mask(ir)                       # in IR coords
            if mask.max() == 0:
                continue                                       # fire subset only
            mask_rgb = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            fid = f"v{vid}_{i:06d}"
            cv2.imwrite(str(FLAME2_RGB_DIR / f"{fid}.jpg"), rgb,
                        [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            cv2.imwrite(str(FLAME2_MASK_DIR / f"{fid}.png"), (mask_rgb > 0).astype("uint8"))
            cv2.imwrite(str(FLAME2_IR_DIR / f"{fid}.jpg"), ir,
                        [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            kept += 1
    return kept, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0, help="sampling rate (frames/sec)")
    ap.add_argument("--videos", type=int, nargs="*", default=list(range(1, 8)))
    args = ap.parse_args()
    if not ZIP.exists():
        raise FileNotFoundError(f"FLAME-2 zip not found: {ZIP}")
    for d in (FLAME2_RGB_DIR, FLAME2_MASK_DIR, FLAME2_IR_DIR):
        d.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    for vid in args.videos:
        kept, sampled = process_pair(vid, args.fps)
        total_kept += kept
        print(f"video {vid}: kept {kept}/{sampled} fire frames "
              f"(running total {total_kept})", flush=True)
    print(f"DONE: {total_kept} FLAME-2 fire frames -> {FLAME2_RGB_DIR}")


if __name__ == "__main__":
    main()
