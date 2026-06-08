import sys
import os
import datetime
sys.path.append(os.path.abspath('.'))


timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
work_dir = f'./work_dirs/seg_mvit_mutual_vit_80k_{timestamp}'

default_scope = 'mmseg'
custom_imports = dict(imports=['mutualvit_mmseg'], allow_failed_imports=False)


crop_size = (224, 224) 


model = dict(
    type='EncoderDecoder',
    # --- THE MISSING BLOCK ---
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53], 
        std=[58.395, 57.12, 57.375],    
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size
    ),
    # -------------------------
    backbone=dict(
        #type='MutualViTBackbone',
        type='ViTBackbone', 
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=9,
        heads=12
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=192,  
        in_index=0,       
        channels=256,     
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=21,    
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# --- Data Pipelines ---
dataset_type = 'PascalVOCDataset'
data_root = './datasets/VOCdevkit/VOC2012'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=False), 
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# --- Dataloaders ---
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=4000, by_epoch=False, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100)
)