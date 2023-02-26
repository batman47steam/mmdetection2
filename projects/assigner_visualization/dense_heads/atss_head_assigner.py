# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
from mmdet.utils import InstanceList
from torch import Tensor

from mmdet.models import ATSSHead
from mmdet.registry import MODELS

from mmdet.models.utils import unpack_gt_instances
from mmdet.models.utils import multi_apply, images_to_levels, unmap

# 这个逻辑可能不需要，因为本来mmdet里和mmyolo里batch的形式就不太一样
#from mmyolo.models.utils import gt_instances_preprocess

@MODELS.register_module()
class ATSSHeadAssigner(ATSSHead):

    def assign_by_gt_and_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: List[dict],
        inputs_hw: Union[Tensor, tuple] = (640, 640)
    ) -> dict:
        """Calculate the assigning results based on the gt and features
        extracted by the detection head.
        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            inputs_hw (Union[Tensor, tuple]): Height and width of inputs size.
        Returns:
            dict[str, Tensor]: A dictionary of assigning results.
        """
        num_imgs = len(batch_img_metas)
        device = cls_scores[0].device

        current_featmap_sizes = [ # 按道理是不需要cls_scores的，这里主要是利用他来得到featuremap_sizes
            cls_score.shape[2:] for cls_score in cls_scores
        ]

        anchor_list, valid_flag_list = self.get_anchors(
            current_featmap_sizes, batch_img_metas, device=device
        )

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply( # 这个是直接从atss的部分复制过来的
            self._get_targets_single, # 应该是只要get_targets_single就可以了
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=True)


        # 反正运行到这里，我已经有了pos_inds了，我知道哪些样本是正样本了
        # 1. pos_inds 对应的level 2. 在level上具体的坐标 3. pos_inds对应的label 4. pos_inds对应的anchor 5. pos_inds对应的gtbox
        pos_inds = pos_inds_list[0] # 正样本的anchor point在所有的anchor中的序号，不是某一层
        all_labels = all_labels[0]
        targets = all_bbox_targets[0][pos_inds] # 需要回归的bbox的大小

        prior_offset = self.prior_generator.center_offset # 目前没什么用，最后画图的时候需要

        level_inds = torch.zeros_like(all_labels)
        img_inds = torch.zeros_like(all_labels)
        level_nums = [0] + [f[0] * f[1] for f in current_featmap_sizes]
        for i in range(len(level_nums) - 1):
            level_nums[i + 1] = level_nums[i] + level_nums[i + 1]
            level_inds[level_nums[i]:level_nums[i + 1]] = i
        level_inds_pos = level_inds[pos_inds] # 判断分配的正样本到底是在哪个level

        img_inds = img_inds[pos_inds]
        labels = all_labels[pos_inds]

        gt_bboxes = batch_gt_instances[0].bboxes
        # 这里的意义估计是这样的，就是我的targets里面每个pos_inds都会对应一个正样本
        # 应该是得到到底和哪个正样本匹配，targets[0].shape[0]=4, targets[0]=[0,354,72,526]
        matched_gt_inds = torch.tensor(
            [((t == gt_bboxes).sum(dim=1) == t.shape[0]).nonzero()[0]
             for t in targets],
            device=device)



        inputs_hw = batch_img_metas[0]['batch_input_shape']
        assign_results = []
        for i in range(self.prior_generator.num_levels):
            retained_inds = level_inds_pos == i # 正样本是不是在这个level上
            if not retained_inds.any(): # 这个是没有正样的时候，坐标都是0
                assign_results_prior = {
                    'stride':
                    self.prior_generator.base_sizes[i],
                    'grid_x_inds':
                    torch.zeros([0], dtype=torch.int64).to(device),
                    'grid_y_inds':
                    torch.zeros([0], dtype=torch.int64).to(device),
                    'img_inds':
                    torch.zeros([0], dtype=torch.int64).to(device),
                    'class_inds':
                    torch.zeros([0], dtype=torch.int64).to(device),
                    'retained_gt_inds':
                    torch.zeros([0], dtype=torch.int64).to(device),
                    'prior_ind':
                    0,
                    'offset':
                    prior_offset
                }
            else:
                w = inputs_hw[1] // self.prior_generator.base_sizes[i]

                retained_pos_inds = pos_inds[retained_inds] - level_nums[i] # 如果说正样本在这个level上，就进一步得到正样本具体的坐标
                grid_y_inds = retained_pos_inds // w # grid_y_inds和grid_x_inds对应的就是正负样本的坐标
                grid_x_inds = retained_pos_inds - retained_pos_inds // w * w
                assign_results_prior = {
                    'stride': self.prior_generator.base_sizes[i],
                    'grid_x_inds': grid_x_inds,
                    'grid_y_inds': grid_y_inds,
                    'img_inds': img_inds[retained_inds],
                    'class_inds': labels[retained_inds],
                    'retained_gt_inds': matched_gt_inds[retained_inds],
                    'prior_ind': 0,
                    'offset': prior_offset
                }
            assign_results.append([assign_results_prior])
        return assign_results

    def assign(self, batch_data_samples: Union[list, dict],
               inputs_hw: Union[tuple, torch.Size]) -> dict:
        """Calculate assigning results. This function is provided to the
        `assigner_visualization.py` script.

        Args:
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            inputs_hw: Height and width of inputs size

        Returns:
            dict: A dictionary of assigning components.
        """
        if isinstance(batch_data_samples, list):
            raise NotImplementedError(
                'assigning results_list is not implemented')
        else:
            # 不用完全和mmyolo一样，只要在这里提供必要的信息就可以了
            cls_scores, bbox_preds, _ = self(batch_data_samples['feats'])
            outputs = unpack_gt_instances(batch_data_samples['data_samples'])
            (batch_gt_instances, batch_gt_instances_ignore,
             batch_img_metas) = outputs
            assign_inputs = (cls_scores, bbox_preds, batch_gt_instances, batch_img_metas,
                             batch_gt_instances_ignore, inputs_hw)
            # assign_inputs = (cls_scores, bbox_preds,
            #                  batch_data_samples['bboxes_labels'],
            #                  batch_data_samples['img_metas'], inputs_hw)
        assign_results = self.assign_by_gt_and_feat(*assign_inputs)
        return assign_results