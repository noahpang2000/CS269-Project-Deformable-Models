import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.apis import init_detector

class CircConv(nn.Module):
    """1D convolution with circular padding -- preserves closed-loop topology."""
    def __init__(self, cin: int, cout: int, k: int = 9):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv1d(cin, cout, k)

    def forward(self, x):
        x = torch.cat([x[..., -self.pad:], x, x[..., :self.pad]], dim=-1)
        return self.conv(x)

class CircResBlock(nn.Module):
    """1D Residual Block with circular convolutions (Faithful to paper's contour network)."""
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.conv1 = CircConv(channels, channels, k)
        self.conv2 = CircConv(channels, channels, k)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

class ResidualSnakeHead(nn.Module):
    """Upgraded Snake Head using Residual Blocks instead of a shallow sequential stack."""
    def __init__(self, feat_c: int, hidden: int = 128, n_blocks: int = 3):
        super().__init__()
        # Input channel size = image feature channels + 2 (for x, y coordinates)
        self.in_conv = CircConv(feat_c + 2, hidden, k=1)
        
        # Deep Residual backbone for the 1D contour
        self.blocks = nn.ModuleList([CircResBlock(hidden) for _ in range(n_blocks)])
        
        # Output layer predicts (dx, dy) offsets
        self.out_conv = CircConv(hidden, 2, k=1)

    def forward(self, feat_at_verts, coords_norm):  # [B, Cf, N], [B, 2, N]
        x = torch.cat([feat_at_verts, coords_norm], dim=1)
        x = F.relu(self.in_conv(x))
        
        for block in self.blocks:
            x = block(x)
            
        return self.out_conv(x)  # offsets [B, 2, N]

def get_octagon_from_box(boxes: torch.Tensor, n_points: int = 128) -> torch.Tensor:
    """
    Rule-based initialization: Replaces U-Net coarse mask.
    Takes bounding boxes [B, 4] (x_min, y_min, x_max, y_max) and generates
    uniformly sampled octagons [B, N, 2] to be used as explicit geometric priors.
    """
    B = boxes.shape[0]
    octagons = []
    
    for i in range(B):
        xmin, ymin, xmax, ymax = boxes[i]
        w, h = xmax - xmin, ymax - ymin
        
        # Define extreme points (top, bottom, left, right centers)
        # Note: The actual paper uses a CNN to predict these, but to simplify the 
        # refactor, we generate a tight octagon directly from the box dimensions.
        top = torch.tensor([(xmin + xmax)/2, ymin])
        bottom = torch.tensor([(xmin + xmax)/2, ymax])
        left = torch.tensor([xmin, (ymin + ymax)/2])
        right = torch.tensor([xmax, (ymin + ymax)/2])
        
        # Define octagon corners (e.g., moving 1/4 width/height from extremes)
        corners = [
            torch.tensor([xmin + w/4, ymin]),      # Top-Left
            torch.tensor([xmax - w/4, ymin]),      # Top-Right
            torch.tensor([xmax, ymin + h/4]),      # Right-Top
            torch.tensor([xmax, ymax - h/4]),      # Right-Bottom
            torch.tensor([xmax - w/4, ymax]),      # Bottom-Right
            torch.tensor([xmin + w/4, ymax]),      # Bottom-Left
            torch.tensor([xmin, ymax - h/4]),      # Left-Bottom
            torch.tensor([xmin, ymin + h/4]),      # Left-Top
        ]
        
        # Interpolate points along the 8 edges to get exactly `n_points`
        pts_per_edge = n_points // 8
        contour = []
        for j in range(8):
            start = corners[j]
            end = corners[(j + 1) % 8]
            # Linear interpolation
            t = torch.linspace(0, 1, pts_per_edge + 1)[:-1].unsqueeze(1)
            edge_pts = start * (1 - t) + end * t
            contour.append(edge_pts)
            
        octagons.append(torch.cat(contour, dim=0))
        
    return torch.stack(octagons).to(boxes.device)  # [B, N, 2]

class DeepSnakePaper(nn.Module):
    """
    Decoupled DeepSnake. It no longer owns the U-Net. 
    It expects deep features (from CenterNet/Deformable CNN) and a bounding box prior.
    """
    def __init__(self, feat_dim: int = 64, n_iter: int = 3):
        super().__init__()
        self.n_iter = n_iter
        self.snakes = nn.ModuleList([ResidualSnakeHead(feat_dim) for _ in range(n_iter)])

    def _sample(self, feat, contour, size):
        # size is now [W, H]
        size_tensor = torch.tensor(size, device=contour.device)
        # Normalizes coordinates to [-1, 1] for grid_sample
        grid = (contour / (size_tensor - 1)) * 2 - 1
        sampled = F.grid_sample(feat, grid.unsqueeze(1), align_corners=True)  # [B, C, 1, N]
        return sampled.squeeze(2)

    def forward(self, features, boxes, image_size: int, n_points: int = 128):
        """
        features: [B, C, H, W] - High quality features from a backbone
        boxes: [B, 4] - Bounding box coordinates from a detector
        """
        # 1. Rule-based geometric prior (replaces U-Net mask)
        contour = get_octagon_from_box(boxes, n_points)
        
        outputs = [contour]
        size_tensor = torch.tensor(image_size, device=contour.device)

        # 2. Iterative Deformation
        for head in self.snakes:
            # Extract features at the current vertex locations
            verts_feat = self._sample(features, contour, image_size)     # [B, C, N]
            
            # Normalize contour coordinates for the network
            coords_n = ((contour / (size_tensor - 1)) * 2 - 1).transpose(1, 2) # [B, 2, N]
            
            # Predict offsets
            offset = head(verts_feat, coords_n).transpose(1, 2)          # [B, N, 2]
            
            # Apply deformation
            contour = contour + offset
            outputs.append(contour)
            
        return outputs

class DeepSnakePipeline(nn.Module):
    def __init__(self, config_file: str, checkpoint_file: str, snake_feat_dim: int = 64, device: str = 'cuda'):
        super().__init__()
        
        # 1. Load the pre-trained MMDet CenterNet
        # init_detector handles building the architecture and loading the weights
        self.detector = init_detector(config_file, checkpoint_file, device=device)
        
        # Freeze the detector if you only want to train the Snake head right now
        for param in self.detector.parameters():
            param.requires_grad = False
            
        # 2. Initialize your decoupled DeepSnake module
        self.snake = DeepSnakePaper(feat_dim=snake_feat_dim)
        
        self.device = device

    def forward(self, img_tensor, conf_threshold=0.3):
        """
        img_tensor: [B, 3, H, W] normalized image tensor
        """
        B, _, H, W = img_tensor.shape
        
        # ==========================================
        # STEP 1: Extract Features from MMDet
        # ==========================================
        multi_scale_features = self.detector.extract_feat(img_tensor)
        snake_features = multi_scale_features[0] # Shape: [B, C, H_f, W_f]
        
        # ==========================================
        # STEP 2: Extract Bounding Boxes
        # ==========================================
        
        cls_scores, bbox_preds = self.detector.bbox_head(multi_scale_features)
            
        # Decode the raw head predictions into actual image coordinates
        batch_metas = [{'img_shape': (H, W), 'scale_factor': (1., 1.)} for _ in range(B)]
        
        results_list = self.detector.bbox_head.predict_by_feat(
            cls_scores, bbox_preds, 
            batch_img_metas=batch_metas, 
            cfg=self.detector.test_cfg
        )
        
        # ==========================================
        # STEP 3: Align Features with Boxes
        # ==========================================
        all_valid_boxes = []
        all_expanded_features = []
        batch_indices = [] # Optional: Keeps track of which image each object belongs to
        
        # Process predictions per image
        for i in range(B):
            pred_instances = results_list[i]
            
            # Filter by confidence threshold
            keep_idx = pred_instances.scores > conf_threshold
            valid_boxes = pred_instances.bboxes[keep_idx]
            N_i = len(valid_boxes)
            
            if N_i > 0:
                all_valid_boxes.append(valid_boxes)
                
                # Grab the feature map for THIS specific image (snake_features[i] is [C, H_f, W_f])
                # Add a batch dim, then copy it N_i times to match the number of boxes
                feat_i = snake_features[i].unsqueeze(0).expand(N_i, -1, -1, -1)
                all_expanded_features.append(feat_i)
                
                # Track original image index
                batch_indices.extend([i] * N_i)
        
        # If no objects detected in the ENTIRE batch, return early
        if len(all_valid_boxes) == 0:
            print("No objects detected above threshold.")
            return [], []

        # Flatten everything across the batch (N_total = sum of all N_i)
        batched_boxes = torch.cat(all_valid_boxes, dim=0)          # [N_total, 4]
        batched_features = torch.cat(all_expanded_features, dim=0) # [N_total, C, H_f, W_f]

        # ==========================================
        # STEP 4: Pass to DeepSnake
        # ==========================================
        # The Snake head now treats N_total as the batch dimension
        contours = self.snake(
            features=batched_features, 
            boxes=batched_boxes, 
            image_size=(W, H) 
        )
        
        return contours, batch_indices