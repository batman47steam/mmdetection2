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
    palette=[(220,20,60), (119,11,32)]
)


# model settings
model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='CostumDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet2',
        deep_stem=True,
        depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='CTResNetNeck',
        in_channels=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=num_classes,
        in_channels=64,
        feat_channels=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

train_pipeline = [
    dict(
        type='LoadMultiImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiPhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='MultiRandomCenterCropPad',
        # The cropped images are padded into squares during training,
        # but may be less than crop_size.
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    # Make sure the output is always crop_size.
    dict(type='MultiResize', scale=(512, 512), keep_ratio=True),
    dict(type='MultiRandomFlip', prob=0.5),
    dict(type='CostumPackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadMultiImageFromFile',
        to_float32=True,
        file_client_args={{_base_.file_client_args}}),
    # don't need Resize
    dict(
        type='MultiRandomCenterCropPad',
        ratios=None,
        border=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_mode=True,
        test_pad_mode=['logical_or', 31],
        test_pad_add_pix=1),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CostumPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'border'))
]

# Use RepeatDataset to speed up training
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=0),
            pipeline=train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img=''),
    pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root+'val.json')
test_evaluator = val_evaluator
# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

max_epochs = 28

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.000125, momentum=0.9, weight_decay=0.0001))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[18, 24],  # the real step is [18*5, 24*5]
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)  # the real epoch is 28*5=140

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)

# 每间隔相应的step打印日志
default_hooks = dict(logger=dict(type='LoggerHook', interval=100))



