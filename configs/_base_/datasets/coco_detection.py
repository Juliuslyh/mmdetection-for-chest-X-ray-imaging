# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='GaussNoise',
        var_limit=0.5,
        mean=0.0,
        per_channel=True,
        p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomShift', shift_ratio=0.5),
    #dict(type='Rotate', level=10, max_rotate_angle=10, prob=0.5),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True)
    # ),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='./dataset_pvc/train_dataset.json',
        img_prefix='./dataset_pvc/img_train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='./dataset_pvc/valid_dataset.json',
        img_prefix='./dataset_pvc/img_valid/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='./dataset_pvc/test_dataset.json',
        img_prefix='./dataset_pvc/img_test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
