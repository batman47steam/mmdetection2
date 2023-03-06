_base_ = ['../../../configs/tood/tood_r50_fpn_1x_cdnet.py']

custom_imports = dict(imports=[
    'projects.assigner_visualization.detectors',
    'projects.assigner_visualization.dense_heads'
])

model = dict(
    type='SingleStageDetectorAssigner', bbox_head=dict(type='TOODHeadAssigner')
)