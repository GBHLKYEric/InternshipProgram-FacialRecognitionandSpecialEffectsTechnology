"""MMDetection 3.3 / MMCV 2.1 / MMEngine 0.10; use its separate Linux environment.

RetinaNet is a practical detector baseline, not the RetinaFace algorithm.
Paths are relative to the project root. COCO AP is not WIDER's official subset AP.
"""
_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(bbox_head=dict(num_classes=1))
metainfo = dict(classes=('face',), palette=[(255, 100, 100)])
train_dataloader = dict(batch_size=2, num_workers=2, dataset=dict(
    data_root='data/', ann_file='wider/train.json', data_prefix=dict(img='WIDER_train/images/'), metainfo=metainfo))
val_dataloader = dict(batch_size=1, num_workers=2, dataset=dict(
    data_root='data/', ann_file='wider/val.json', data_prefix=dict(img='WIDER_val/images/'), metainfo=metainfo))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file='data/wider/val.json', metric='bbox', classwise=True)
test_evaluator = val_evaluator
train_cfg = dict(max_epochs=12)
optim_wrapper = dict(optimizer=dict(lr=.00125))
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2))
work_dir = 'runs/wider_retinanet'
