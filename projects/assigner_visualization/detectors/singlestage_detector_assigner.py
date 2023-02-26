# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmdet.models import SingleStageDetector
from mmdet.registry import MODELS
from projects.assigner_visualization.dense_heads import ATSSHeadAssigner


@MODELS.register_module()
class SingleStageDetectorAssigner(SingleStageDetector):

    def assign(self, data: dict) -> Union[dict, list]:
        """Calculate assigning results from a batch of inputs and data
        samples.This function is provided to the `assigner_visualization.py`
        script.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            dict: A dictionary of assigning components.
        """
        assert isinstance(data, dict)
        assert len(data['inputs']) == 1, 'Only support batchsize == 1'
        data = self.data_preprocessor(data, True) # 由config指定的data_preprocessor
        available_assigners = (ATSSHeadAssigner)
        if isinstance(self.bbox_head, available_assigners): # 这里的data_samples是一个list
            data['feats'] = self.extract_feat(data['inputs']) # 完成backbone + neck部分的特征提取,应该是只有动态的方法才会需要得到网络输出的结果
        inputs_hw = data['inputs'].shape[-2:]
        assign_results = self.bbox_head.assign(data, inputs_hw)
        return assign_results

