from methods import ClassicalSnakeBaseline
import numpy as np
    

# --- Mock Data Setup ---
image = cv2.imread("fire_frame_001.jpg")
gt_mask = cv2.imread("fire_mask_001.png", 0) / 255

def GridSearch():
    # --- Grid Search Definition ---
    alphas = [0.01, 0.015, 0.05]
    betas = [5.0, 10.0, 50.0]
    best_iou = 0
    best_params = {}

    print("Starting Grid Search...")

    for a in alphas:
        for b in betas:
            # Initialize the baseline with current grid parameters
            model = ClassicalSnakeBaseline(alpha=a, beta=b)
            
            # Dilate ground truth by 15% for initialization
            init_contour = model.generate_initial_contour(gt_mask, dilation_factor=1.15)
            
            # Run the physics engine
            final_snake = model.predict(image, init_contour)
            
            # Calculate metric
            iou = model.calculate_iou(final_snake, gt_mask)
            
            print(f"Alpha: {a} | Beta: {b} | IoU: {iou:.4f}")
            
            if iou > best_iou:
                best_iou = iou
                best_params = {'alpha': a, 'beta': b}

    print(f"Optimal Parameters: {best_params} with IoU: {best_iou:.4f}")

if __name__ == "__main__":
    GridSearch()