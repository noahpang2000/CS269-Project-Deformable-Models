import numpy as np
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.color import rgb2gray
from shapely.geometry import Polygon

class ClassicalSnakeBaseline:
    def __init__(self, alpha=0.015, beta=10.0, w_line=0.0, w_edge=1.0, gamma=0.001):
        """
        Initializes the Kass 1988 Active Contour parameters.
        alpha: Elasticity (tension)
        beta: Rigidity (stiffness)
        w_line/w_edge: Weights for the external image energy (attraction to edges)
        gamma: Explicit time stepping parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.w_line = w_line
        self.w_edge = w_edge
        self.gamma = gamma

    def generate_initial_contour(self, gt_mask, dilation_factor=1.15):
        """
        Takes a binary ground truth mask and creates a contour dilated by a specific factor.
        """
        # 1. Extract the ground truth contour
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
            
        # Get the largest contour (assuming one main fire front)
        main_contour = max(contours, key=cv2.contourArea).squeeze()
        
        # 2. Calculate the centroid to scale outward from
        M = cv2.moments(main_contour)
        if M['m00'] == 0:
            return main_contour
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid = np.array([cx, cy])
        
        # 3. Dilate the contour coordinates outward by the dilation factor (e.g., 15% = 1.15)
        # Note: skimage active_contour expects coordinates in (y, x) format
        dilated_contour = centroid + dilation_factor * (main_contour - centroid)
        dilated_contour_yx = np.fliplr(dilated_contour) 
        
        return dilated_contour_yx

    def predict(self, image, initial_contour_yx, max_num_iter=2500):
        """
        Runs the active contour physics simulation on the image.
        """
        # Preprocess: Convert to grayscale and apply Gaussian blur to smooth local minima (smoke/noise)
        img_gray = rgb2gray(image)
        img_smooth = gaussian(img_gray, sigma=3.0, preserve_range=False)
        
        # Run the physics simulation
        snake_yx = active_contour(
            img_smooth,
            initial_contour_yx,
            alpha=self.alpha,
            beta=self.beta,
            w_line=self.w_line,
            w_edge=self.w_edge,
            gamma=self.gamma,
            max_num_iter=max_num_iter
        )
        
        # Convert back to (x, y) for evaluation
        return np.fliplr(snake_yx)

    def calculate_iou(self, pred_contour_xy, gt_mask):
        """
        Calculates Intersection over Union (IoU) using Shapely for polygon math.
        """
        # Extract ground truth polygon
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        gt_poly = Polygon(max(contours, key=cv2.contourArea).squeeze())
        
        # Create prediction polygon
        pred_poly = Polygon(pred_contour_xy)
        
        if not pred_poly.is_valid:
            pred_poly = pred_poly.buffer(0) # Fix self-intersections caused by node bunching
            
        # Calculate standard IoU
        intersection = gt_poly.intersection(pred_poly).area
        union = gt_poly.union(pred_poly).area
        
        if union == 0:
            return 0.0
            
        return intersection / union