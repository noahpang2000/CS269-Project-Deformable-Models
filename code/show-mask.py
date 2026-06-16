import cv2
import numpy as np

# Load the mask as a grayscale image
mask = cv2.imread('00024.TIFF', cv2.IMREAD_GRAYSCALE)

# Option A: Simple multiplication (if you know the max value is exactly 1)
# visual_mask = mask * 255

# Option B: Thresholding (Safest approach for any non-zero value)
_, visual_mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

# Save or display the new mask
cv2.imwrite('visible_mask_flame3.png', visual_mask)