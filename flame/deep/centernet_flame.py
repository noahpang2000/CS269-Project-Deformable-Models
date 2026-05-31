# Inherit the base CenterNet architecture with a ResNet-18 backbone
_base_ = './mmdetection/configs/centernet/centernet_resnet18_dcnv2_140e_coco.py'

# 1. Modify the model for a single class ("fire")
model = dict(
    bbox_head=dict(num_classes=1)
)

# 2. Update Dataset Paths
dataset_type = 'CocoDataset'
data_root = 'flame3_coco/' # The folder containing your JSONs and images
classes = ('fire',)

data = dict(
    samples_per_gpu=4, # Batch size per GPU
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/' # Folder with your raw RGB images
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'images/'
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'images/'
    )
)

# 3. Tuning parameters (Optional)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=50) # Match your 50 epochs