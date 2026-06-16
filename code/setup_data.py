import os
import zipfile
import shutil
import random
from pathlib import Path

# --- Configuration ---
# Update these if your downloaded zip files have slightly different names
IMAGES_ZIP = "Images_zip.zip" 
MASKS_ZIP = "Masks.zip"

# Split Ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# Test ratio is automatically the remaining 10%

def create_directory_structure():
    """Builds the pristine PyTorch-ready folder hierarchy."""
    base_dir = Path("data")
    splits = ["train", "val", "test"]
    subdirs = ["images", "masks"]

    for split in splits:
        for subdir in subdirs:
            (base_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    return base_dir

def process_dataset():
    # 1. Sanity Check
    for z in [IMAGES_ZIP, MASKS_ZIP]:
        if not os.path.exists(z):
            print(f"ERROR: Could not find '{z}'. Please ensure both zips are in this folder.")
            return

    # 2. Extract Both Zips to Temporary Folders
    temp_img_dir = Path("temp_images_raw")
    temp_mask_dir = Path("temp_masks_raw")
    
    print(f"Extracting {IMAGES_ZIP}...")
    with zipfile.ZipFile(IMAGES_ZIP, 'r') as zip_ref:
        zip_ref.extractall(temp_img_dir)
        
    print(f"Extracting {MASKS_ZIP}...")
    with zipfile.ZipFile(MASKS_ZIP, 'r') as zip_ref:
        zip_ref.extractall(temp_mask_dir)

    # 3. Gather Files (Searching recursively in case the zips have internal folders)
    # We ignore hidden OS files like .DS_Store
    images = [f for f in temp_img_dir.rglob("*.*") if f.is_file() and not f.name.startswith('.')]
    masks = [f for f in temp_mask_dir.rglob("*.*") if f.is_file() and not f.name.startswith('.')]

    print(f"Found {len(images)} images and {len(masks)} masks.")

    # 4. The Matching Engine
    # Map the masks into a dictionary using their 'stem' (filename without extension)
    mask_dict = {m.stem: m for m in masks}
    
    paired_data = []
    missing_masks = 0

    for img_path in images:
        # If your masks have a suffix like 'image_0_mask.png', change this to: key = img_path.stem + "_mask"
        key = img_path.stem
        
        if key in mask_dict:
            paired_data.append((img_path, mask_dict[key]))
        else:
            missing_masks += 1

    if missing_masks > 0:
        print(f"WARNING: {missing_masks} images did not have a matching mask and were dropped.")

    # 5. Shuffle and Split
    random.seed(42) # Guarantees reproducible splits if you ever re-run the script
    random.shuffle(paired_data)

    total = len(paired_data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = paired_data[:train_end]
    val_data = paired_data[train_end:val_end]
    test_data = paired_data[val_end:]

    print(f"\nDistributing Data:")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # 6. Move to the PyTorch Directory Structure
    base_dir = create_directory_structure()

    def move_files(dataset_split, split_name):
        for img_path, mask_path in dataset_split:
            # Copy to final destination
            shutil.copy(img_path, base_dir / split_name / "images" / img_path.name)
            shutil.copy(mask_path, base_dir / split_name / "masks" / mask_path.name)

    print("Copying files to splits... (this might take a minute)")
    move_files(train_data, "train")
    move_files(val_data, "val")
    move_files(test_data, "test")

    # 7. Cleanup
    print("Cleaning up temporary extraction folders...")
    shutil.rmtree(temp_img_dir)
    shutil.rmtree(temp_mask_dir)
    
    print("\n✓ Pipeline complete. Data is formatted and ready for the PyTorch DataLoader.")

if __name__ == "__main__":
    process_dataset()