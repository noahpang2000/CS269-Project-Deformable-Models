import tifffile as tiff
import numpy as np
import cv2

# tifffile cleanly extracts the raw temperature array
thermal_data = tiff.imread('00024.TIFF')

# Create a boolean mask where temp > 150, then scale to 0-255
mask = (thermal_data > 150).astype(np.uint8) * 255

# Save the resulting 8-bit image
cv2.imwrite('fire_mask_150C.png', mask)