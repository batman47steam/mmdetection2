_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'CustomDataset'
data_root = 'datasets/CDNET/Source/'

class_name = ('car', 'person')
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60), (119, 11, 32)]
)

model = dict(
    data_preprocessor=dict(
        type='CostumDetDataPreprocessor'),
    backbone=dict(
        type='ResNet_Concate',
        frozen_stages=-1),
    bbox_head=dict(
        num_classes=num_classes)
)

train_pipeline = [
    dict(type='LoadMultiImageFromFile', file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiResize', scale=(608,608), keep_ratio=True),
    dict(type='MultiRandomFlip', prob=0.5),
    dict(type='CostumPackDetInputs')
]

test_pipeline = [
    dict(type='LoadMultiImageFromFile', file_client_args={{_base_.file_client_args}}),
    dict(type='MultiResize', scale=(640,640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CostumPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, mim_size=0),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img=''),
    pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root+'test.json')
test_evaluator = val_evaluator


max_epochs = 23

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.000125, weight_decay=0.0001))



# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
# 命令行 --auto_scale_lr 才会开启
auto_scale_lr = dict(base_batch_size=128)

# 每间隔相应的step打印日志
default_hooks = dict(logger=dict(type='LoggerHook', interval=100))

#visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])