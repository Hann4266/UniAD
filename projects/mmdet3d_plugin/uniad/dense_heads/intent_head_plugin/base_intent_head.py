#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import copy
import torch.nn as nn
# from mmdet.models import  build_loss
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
# import torch.nn.functional as F
# from mmcv.runner import force_fp32

class BaseIntentHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseIntentHead, self).__init__()
        pass

    def _build_loss(self, loss_cls):
        loss_cls = copy.deepcopy(loss_cls) if loss_cls is not None else {}
        self.loss_cls_cfg = loss_cls

        self.loss_type = loss_cls.get("type", "FocalLoss")
        self.use_sigmoid = loss_cls.get("use_sigmoid", True)
        self.focal_gamma = float(loss_cls.get("gamma", 2.0))
        self.focal_alpha = float(loss_cls.get("alpha", 0.25))
        self.loss_weight = float(loss_cls.get("loss_weight", 1.0))
        self.ped_loss_weight = float(loss_cls.get("ped_loss_weight", 2.0))

        cw = loss_cls.get("class_weight", None)
        if cw is not None:
            self.register_buffer(
                "intent_class_weight",
                torch.tensor(cw, dtype=torch.float32),
                persistent=False
            )
        else:
            self.intent_class_weight = None

    def _build_layers(self, transformerlayers, det_layer_num):
        """
        Build the layers of the motion prediction module.

        Args:
            transformerlayers (dict): A dictionary containing the parameters for the transformer layers.
            det_layer_num (int): The number of detection layers.

        Returns:
            None
        """
        self.intentformer = build_transformer_layer_sequence(
            transformerlayers)
        self.layer_track_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * det_layer_num, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True)
        )
        self.boxes_query_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
    
    def _init_layers(self):
        # intent cls branch (same style as motion head: clones per decoder layer)
        cls_branch = []
        cls_branch += [nn.Linear(self.embed_dims, self.embed_dims),
                       nn.LayerNorm(self.embed_dims),
                       nn.ReLU(inplace=True)]
        for _ in range(self.num_cls_fcs - 1):
            cls_branch += [nn.Linear(self.embed_dims, self.embed_dims),
                           nn.LayerNorm(self.embed_dims),
                           nn.ReLU(inplace=True)]
        cls_branch += [nn.Linear(self.embed_dims, self.num_intent)]
        cls_branch = nn.Sequential(*cls_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        num_pred = self.intentformer.num_layers
        self.intent_cls_branches = _get_clones(cls_branch, num_pred)
    def _extract_tracking_centers(self, bbox_results, bev_range):
        """
        extract the bboxes centers and normized according to the bev range
        
        Args:
            bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
            bev_range (List[float]): A list of float values representing the bird's eye view range.

        Returns:
            torch.Tensor: A tensor representing normized centers of the detection bounding boxes.
        """
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            xy = bboxes.gravity_center[:, :2]
            x_norm = (xy[:, 0] - bev_range[0]) / \
                (bev_range[3] - bev_range[0])
            y_norm = (xy[:, 1] - bev_range[1]) / \
                (bev_range[4] - bev_range[1])
            det_bbox_posembed.append(
                torch.cat([x_norm[:, None], y_norm[:, None]], dim=-1))
        return torch.stack(det_bbox_posembed)