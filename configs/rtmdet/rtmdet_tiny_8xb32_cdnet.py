_base_ = './rtmdet_s_8xb32-300e_coco.py'

#checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

dataset_type = 'CustomDataset'
data_root = 'datasets/CDNET/Source/'

class_name = ('car', 'person')
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220,20,60), (119,11,32)]
)

file_client_args = dict(backend='disk')
channels = [128, 256, 512]

model = dict(
    data_preprocessor=dict(
        type='CostumDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='ResNet_Concate',
        depth=18,
        num_stages=4,
        out_indices=(1,2,3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=channels, out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False, num_classes=num_classes))

train_pipeline = [
    dict(
        type='LoadMultiImageFromFile',
        file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiRandomCenterCropPad',
        # The cropped images are padded into squares during training,
        # but may be less than crop_size.
        crop_size=(640, 640),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    # Make sure the output is always crop_size.
    dict(type='MultiResize', scale=(640, 640), keep_ratio=True),
    dict(type='MultiRandomFlip', prob=0.5),
    dict(type='CostumPackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadMultiImageFromFile',
        to_float32=True,
        file_client_args=file_client_args),
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

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=0),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img=''),
    pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root+'val.json', proposal_nums=(100, 1, 10))
test_evaluator = val_evaluator

max_epochs = 24
base_lr = 0.004
interval = 1

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=interval)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=200),
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]