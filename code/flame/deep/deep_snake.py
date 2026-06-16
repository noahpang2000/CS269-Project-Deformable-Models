import torch
import torch.nn as nn
import torch.nn.functional as F
# mmdet is only needed for the CenterNet detector in DeepSnakePipeline (the
# detector-based "paper" variant). Import it lazily so the mmdet-free pieces
# (the snake head DeepSnakePaper, used for training) and the rest of run_deep.py
# still work in environments without mmdet installed.

class CircConv(nn.Module):
    """1D convolution with circular padding -- preserves closed-loop topology."""
    def __init__(self, cin: int, cout: int, k: int = 9):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv1d(cin, cout, k)

    def forward(self, x):
        # NB: with k=1 the pad is 0, and x[..., -0:] would return the WHOLE
        # tensor (not empty), tripling the vertex dim. Guard the pad=0 case.
        if self.pad > 0:
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
    def __init__(self, feat_dim: int = 64, n_iter: int = 3, offset_scale: float = 32.0):
        super().__init__()
        self.n_iter = n_iter
        # The heads emit small (~sub-pixel) raw offsets; scale them to real pixels
        # so a few iterations can actually deform an ~80px octagon. Without this
        # the snake moves vertices ~1px and is effectively a no-op.
        self.offset_scale = offset_scale
        self.snakes = nn.ModuleList([ResidualSnakeHead(feat_dim) for _ in range(n_iter)])

    @staticmethod
    def _wh_tensor(image_size, device) -> torch.Tensor:
        """Normalize image_size (int -> square, or (W, H) tuple) to a [W, H] tensor."""
        if isinstance(image_size, (int, float)):
            wh = (float(image_size), float(image_size))
        else:
            wh = (float(image_size[0]), float(image_size[1]))
        return torch.tensor(wh, device=device)

    def _sample(self, feat, contour, size_tensor):
        # size_tensor: [W, H] on the contour's device
        # Normalizes coordinates to [-1, 1] for grid_sample
        grid = (contour / (size_tensor - 1)) * 2 - 1
        sampled = F.grid_sample(feat, grid.unsqueeze(1), align_corners=True)  # [B, C, 1, N]
        return sampled.squeeze(2)

    def forward(self, features, boxes, image_size, n_points: int = 128):
        """
        features: [B, C, H, W] - High quality features from a backbone
        boxes: [B, 4] - Bounding box coordinates from a detector
        image_size: int (square) or (W, H) tuple
        """
        # 1. Rule-based geometric prior (replaces U-Net mask)
        contour = get_octagon_from_box(boxes, n_points)

        outputs = [contour]
        size_tensor = self._wh_tensor(image_size, contour.device)

        # 2. Iterative Deformation
        for head in self.snakes:
            # Extract features at the current vertex locations
            verts_feat = self._sample(features, contour, size_tensor)    # [B, C, N]

            # Normalize contour coordinates for the network
            coords_n = ((contour / (size_tensor - 1)) * 2 - 1).transpose(1, 2) # [B, 2, N]
            
            # Predict offsets (scaled from the head's small raw output to pixels)
            offset = head(verts_feat, coords_n).transpose(1, 2) * self.offset_scale  # [B, N, 2]

            # Apply deformation
            contour = contour + offset
            outputs.append(contour)
            
        return outputs

class DeepSnakePipeline(nn.Module):
    def __init__(self, config_file: str, checkpoint_file: str, snake_feat_dim: int = 64, device: str = 'cuda'):
        super().__init__()
        
        # 1. Load the pre-trained MMDet CenterNet
        # init_detector handles building the architecture and loading the weights
        from mmdet.apis import init_detector  # lazy: only the detector path needs mmdet
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
        
        # CenterNet's head (mmdet 3.x) returns THREE prediction lists, not two:
        # (center_heatmap_preds, wh_preds, offset_preds). predict_by_feat takes
        # exactly those three and runs decode (+NMS) internally.
        center_heatmap_preds, wh_preds, offset_preds = \
            self.detector.bbox_head(multi_scale_features)

        # Decode the raw head predictions into actual image coordinates.
        # CenterNet's decoder reads 'batch_input_shape', 'border' and
        # 'scale_factor'. 'border' is the RandomCenterCropPad offset
        # (top, bottom, left, right); at inference with no crop-pad it is the
        # full frame, so the predicted boxes map straight back to image coords.
        batch_metas = [{'img_shape': (H, W),
                        'batch_input_shape': (H, W),
                        'border': (0, H, 0, W),
                        'scale_factor': (1., 1.)} for _ in range(B)]

        # CenterNet dedupes via local-maximum heatmap peaks (topk /
        # local_maximum_kernel in test_cfg), not box NMS, so with_nms=False.
        results_list = self.detector.bbox_head.predict_by_feat(
            center_heatmap_preds, wh_preds, offset_preds,
            batch_img_metas=batch_metas,
            rescale=False,
            with_nms=False,
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