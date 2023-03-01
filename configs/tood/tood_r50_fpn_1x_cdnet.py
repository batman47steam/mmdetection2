_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CustomDataset'
data_root = 'datasets/CDNET/Source/'

class_name = ('car', 'person')
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60), (119, 11, 32)]
)

# model settings
model = dict(
    type='TOOD',
    data_preprocessor=dict(
        type='CostumDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet2',
        depth=50,
        deep_stem=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='TOODHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


train_pipeline = [
    dict(type='LoadMultiImageFromFile', file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiResize', scale=(640,640), keep_ratio=True),
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
    batch_size=8,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, mim_size=0),
            indices=5000,
            pipeline=train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img=''),
    pipeline=test_pipeline))
test_dataloader = val_dataloader  # 先这样，在test的时候还是要换成test.json

val_evaluator = dict(ann_file=data_root+'val.json')
test_evaluator = val_evaluator

max_epochs = 24

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.000125, momentum=0.9, weight_decay=0.0001))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[18, 24],
        gamma=0.1)
]

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

default_hooks = dict(logger=dict(type='LoggerHook', interval=100))

#visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])



