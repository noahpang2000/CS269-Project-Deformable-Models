import json
import cv2
import numpy as np
from pathlib import Path
from flame.data import load_frame
from flame.splits import make_splits

def create_coco_json(frame_ids, output_path):
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fire"}] # Only 1 class for your dataset
    }
    
    ann_id = 1
    for img_id, fid in enumerate(frame_ids):
        frame = load_frame(fid)
        h, w = frame.gt_mask.shape
        
        # Add image info
        coco_dict["images"].append({
            "id": img_id,
            "file_name": f"{fid}.jpg", # Ensure this matches your actual image filenames
            "width": w,
            "height": h
        })
        
        # Find all distinct fire instances (using Connected Components)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((frame.gt_mask > 0).astype(np.uint8))
        
        for lbl in range(1, num):
            if stats[lbl, cv2.CC_STAT_AREA] < 30: # MIN_CC_PX
                continue
                
            x, y, bw, bh, area = stats[lbl]
            
            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1, # "fire"
                "bbox": [int(x), int(y), int(bw), int(bh)], # COCO uses [x, y, width, height]
                "area": int(area),
                "iscrowd": 0
            })
            ann_id += 1
            
    with open(output_path, 'w') as f:
        json.dump(coco_dict, f)

# Run for both splits
splits = make_splits()
create_coco_json(splits["train"], "flame3_coco/annotations/train.json")
create_coco_json(splits["val"], "flame3_coco/annotations/val.json")