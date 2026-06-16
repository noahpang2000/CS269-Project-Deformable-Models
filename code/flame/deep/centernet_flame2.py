# FLAME-2 CenterNet config: identical to centernet_flame.py but pointed at the
# flame2_coco/ dataset (built by `python -m flame.deep.create_coco --dataset flame2`).
#
# Train (from repo root, in the flame-snake env):
#   python flame/mmdetection/tools/train.py flame/deep/centernet_flame2.py
_base_ = './centernet_flame.py'

data_root = 'flame2_coco/'

train_dataloader = dict(dataset=dict(dataset=dict(data_root=data_root)))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator
