_base_ = ['../../../work_dirs/atss_r18_fpn_1xb1_cdnet/atss_r18_fpn_1xb1_cdnet.py']

custom_imports = dict(imports=[
    'projects.assigner_visualization.detectors',
    'projects.assigner_visualization.dense_heads'
])

model = dict(
    type='SingleStageDetectorAssigner', bbox_head=dict(type='ATSSHeadAssigner'))

