# MMDetection 3.x config: CenterNet (ResNet-18) fine-tuned for single-class
# "fire" box detection. This detector is trained SEPARATELY from the snake; at
# test time DeepSnakePipeline uses it to produce boxes for the snake to deform.
#
# Build the COCO data first:  python -m flame.deep.create_coco
# Train (from repo root):
#   python flame/mmdetection/tools/train.py flame/deep/centernet_flame.py
#
# Paths are relative to the repo root (the CWD you launch training from).

# _base_ is resolved relative to THIS file's directory (flame/deep/), so the
# mmdetection clone at flame/mmdetection/ is one level up.
_base_ = '../mmdetection/configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py'

# 1. Single class ("fire").
model = dict(bbox_head=dict(num_classes=1))

# 2. Dataset paths — must match flame3_coco/ produced by create_coco.py.
dataset_type = 'CocoDataset'
data_root = 'flame3_coco/'
metainfo = dict(classes=('fire',))

# NOTE: the base CenterNet config wraps the train CocoDataset in a
# RepeatDataset(times=5). The wrapper ignores ann_file/data_root, so the real
# paths MUST be set on the inner `dataset`, not the wrapper.
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(  # RepeatDataset wrapper
        dataset=dict(  # inner CocoDataset
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/train.json',
            data_prefix=dict(img='images/'),
        ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

# 3. Evaluators point at the val annotations.
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/val.json',
                     metric='bbox')
test_evaluator = val_evaluator

# 4. Training schedule (mmdet 3.x uses optim_wrapper + *_cfg loops).
#
# CRITICAL: the base LR (0.01) is calibrated for base_batch_size=128 (8 GPU x
# 16). We train at batch_size=4 on one GPU, so the LR must be scaled by the
# linear rule: 0.01 * 4/128 = 3.125e-4. Running the unscaled 0.01 collapses the
# CenterNet center-heatmap head into a degenerate constant (every predicted box
# gets the same low score -> mAP 0). We keep the base config's clip_grad and
# LinearLR warmup (do NOT clobber the whole optim_wrapper) and let mmdet's
# auto_scale_lr do the batch scaling so this stays correct if batch size changes.
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
auto_scale_lr = dict(enable=True, base_batch_size=128)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))
