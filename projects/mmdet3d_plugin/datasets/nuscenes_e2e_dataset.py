#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import copy
import numpy as np
import torch
import mmcv
import json
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from os import path as osp
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .eval_utils.nuscenes_eval import NuScenesEval_custom
from .eval_utils.nuscenes_eval_track import TrackingEval
from .eval_utils.nuscenes_eval_motion import MotionEval
from nuscenes.eval.common.config import config_factory
import tempfile
from mmcv.parallel import DataContainer as DC
import random
import pickle
from prettytable import PrettyTable

from nuscenes import NuScenes
from projects.mmdet3d_plugin.datasets.data_utils.vector_map import VectorizedLocalMap
from projects.mmdet3d_plugin.datasets.data_utils.rasterize import preprocess_map
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
from projects.mmdet3d_plugin.datasets.data_utils.trajectory_api import NuScenesTraj
from .data_utils.data_utils import lidar_nusc_box_to_global, obtain_map_info, output_to_nusc_box, output_to_nusc_box_det
from nuscenes.prediction import convert_local_coords_to_global

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
def intent_label(index, action_array):
    intent_dic={
        "Crossing" : 2,
        "TURN_RIGHT": 3,
        "TURN_LEFT": 4,
        "LANE_CHANGE_RIGHT": 5,
        "LANE_CHANGE_LEFT": 6,
        "STOPPED": 0,
        "MOVING": 1,
        "Stopped": 0,
        "Moving": 1
    }
    if index < 0 or index >= len(action_array)-1:
        return -1

    action = action_array[index]

    end = min(len(action_array), index + 5)
    future_actions = action_array[index + 1:end]
    if len(future_actions) == 0 or all(a == "na" for a in future_actions):
        return -1

    action_set = {"Crossing", "TURN_RIGHT", "TURN_LEFT", "LANE_CHANGE_RIGHT", "LANE_CHANGE_LEFT"}
    
    if action in action_set:
        for next_action in future_actions:
            if next_action != action and next_action!= "na":
                return intent_dic[next_action]
        return intent_dic[action]
    elif action == "STOPPED" or action == "Stopped":
        next_moving = False
        for next_action in future_actions:
            if next_action in action_set:
                return intent_dic[next_action]
            elif next_action=="MOVING" or next_action=="Moving":
                next_moving = True
        return intent_dic["MOVING"] if next_moving else intent_dic["STOPPED"]
    elif action == "MOVING" or action == "Moving":
        next_stop = False
        for next_action in future_actions:
            if next_action in action_set:
                return intent_dic[next_action]
            elif next_action=="STOPPED" or next_action == "Stopped":
                next_moving = True
        return intent_dic["STOPPED"] if next_stop else intent_dic["MOVING"]
    else:
        return -1

        
            

def mask_for_polygons(canvas_size, fov_degree=35):
    """
    Create FOV triangle mask.
    Triangle exteriors = 1, interiors = 0
    
    Triangle vertices in array indices:
    - (0, 100): bottom center (row 0, col 100)
    - (100, 30): top left (row 100, col 30)
    - (100, 170): top right (row 100, col 170)
    
    Returns:
        mask of shape (100, 200) where:
        - Inside triangle = 0
        - Outside triangle = 1
    """
    import cv2
    from shapely.geometry import Polygon
    
    # Create mask (100, 200)
    mask = np.zeros((100, 200), dtype=np.uint8)
    
    # Triangle vertices as (col, row) for cv2 (x, y format)
    # But you specified as array positions (row, col)
    # Converting (row, col) -> (col, row) for cv2:
    apex = (100, 0)      # array pos (0, 100) -> cv2 point (100, 0)
    left_point = (30, 100)   # array pos (100, 30) -> cv2 point (30, 100)
    right_point = (170, 100) # array pos (100, 170) -> cv2 point (170, 100)
    
    # Create triangle
    triangle = np.array([apex, left_point, right_point], dtype=np.int32)
    
    # Fill triangle with 1
    cv2.fillPoly(mask, [triangle], 1)
    
    # Invert: exteriors = 1, interiors = 0
    mask = 1 - mask
    
    return mask
def plot_image_and_lanes(image_paths, vectors, out_path="lane_vis.png", dpi=200):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    xlim = (-51.2, 51.2)
    ylim = (-51.2, 51.2)

    # ---- Plot 1: image ----
    ax_img = axes[0]
    img_path = f"/zihan-west-vol/UniAD/data/nuscenes/{image_paths[0]}"

    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax_img.imshow(img)
        ax_img.set_title("Front Image")
    else:
        ax_img.text(0.5, 0.5, f"Image not found:\n{img_path}", ha="center", va="center")
        ax_img.set_title("Image (missing)")
    ax_img.axis("off")

    # ---- Plot 2: merged lanes by type with colors ----
    ax = axes[1]
    ax.set_title("Lanes (type 0/1/2 merged)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    type_color = {0: "r", 1: "g", 2: "b"}
    type_label = {0: "type 0", 1: "type 1", 2: "type 2"}
    used_label = set()

    for v in vectors:
        t = int(v.get("type", -1))
        if t not in type_color:
            continue

        pts = np.asarray(v["pts"])  # [N,2]
        color = type_color[t]

        # show legend only once per type
        label = type_label[t] if t not in used_label else None
        used_label.add(t)

        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, marker=".", markersize=3, label=label)
        ax.scatter(pts[0, 0], pts[0, 1], color=color, s=20)

    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
def plot_gt_masks(image_paths, gt_masks, gt_labels, out_path="mask_vis.png", dpi=200):
    fig, axes = plt.subplots(3, 3, figsize=(22, 22))
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.cpu().numpy()
    ax_img = axes[0,0]
    img_path = f"/zihan-west-vol/UniAD/data/nuscenes/{image_paths[0]}"
    type_color = {0: "r", 1: "g", 2: "b", 3:"y", 4:"black",5: "gray"}
    plot_position = {0:(0, 1), 1:(0, 2), 2:(1, 0), 3:(1, 1), 4:(1, 2),5:(2, 0)}
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax_img.imshow(img)
        ax_img.set_title("Front Image")
    else:
        ax_img.text(0.5, 0.5, f"Image not found:\n{img_path}", ha="center", va="center")
        ax_img.set_title("Image (missing)")
    ax_img.axis("off")
    h, w = gt_masks.shape[1], gt_masks.shape[2]
    unique_labels = np.unique(gt_labels)
    for label in unique_labels:
        row, col = plot_position[label]
        ax = axes[row, col]
        class_indices = np.where(gt_labels == label)[0]
        combined_mask = np.zeros((h, w), dtype=float)
        for idx in class_indices:
            combined_mask = np.maximum(combined_mask, gt_masks[idx])
        ax.imshow(combined_mask, cmap='gray', vmin=0, vmax=1, 
                 aspect='equal', origin='lower')
        # ax.set_xlabel('X: [-51.2m, +51.2m]', fontsize=9)
        # ax.set_ylabel('Y: [0m, 51.2m]', fontsize=9)
        ax.set_xticks([0, w/2, w])
        ax.set_xticklabels(['-51.2', '0', '+51.2'], fontsize=8)
        ax.set_yticks([0, h/2, h])
        ax.set_yticklabels(['-51.2', '0', '+51.2'], fontsize=8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    

def plot_gt_3d(image_paths, gt_3d, gt_labels, out_path="mask_vis.png", dpi=200):
    colors = {
        -1: '#808080',  # Gray - ignore
        0: '#FF0000',   # Red - class 0
        1: '#00FF00',   # Green - class 1
        2: '#0000FF',   # Blue - class 2
        3: '#FFFF00',   # Yellow - class 3
        4: '#FF00FF',   # Magenta - class 4
        5: '#00FFFF',   # Cyan - class 5
        6: '#FFA500',   # Orange - class 6
        7: '#800080',   # Purple - class 7
        8: '#FFC0CB',   # Pink - class 8
        9: '#A52A2A',   # Brown - class 9
    }
    xlim = (-51.2, 51.2)
    ylim = (-51.2, 51.2)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_img = axes[0]
    img_path = f"/zihan-west-vol/UniAD/data/nuscenes/{image_paths[0]}"

    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax_img.imshow(img)
        ax_img.set_title("Front Image")
    else:
        ax_img.text(0.5, 0.5, f"Image not found:\n{img_path}", ha="center", va="center")
        ax_img.set_title("Image (missing)")
    ax_img.axis("off")
    ax = axes[1]
    if hasattr(gt_3d, 'tensor'):
        bboxes = gt_3d.tensor.cpu().numpy()
    elif hasattr(gt_3d, 'numpy'):
        bboxes = gt_3d.cpu().numpy()
    else:
        bboxes = np.array(gt_3d)
    
    # Convert labels to numpy
    if hasattr(gt_labels, 'tensor'):
        labels = gt_labels.tensor.cpu().numpy()
    elif hasattr(gt_labels, 'numpy'):
        labels = gt_labels.cpu().numpy()
    else:
        labels = np.array(gt_labels)
    for i in range(len(labels)):
        color = colors[labels[i]]
        bbox_3d = bboxes[i]
        x, y = bbox_3d[0], bbox_3d[1]  # Center x, y
        ax.scatter(x, y, c=color, s=100)
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
@DATASETS.register_module()
class NuScenesE2EDataset(NuScenesDataset):
    r"""NuScenes E2E Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self,
                 queue_length=4,
                 bev_size=(200, 200),
                 patch_size=(102.4, 102.4),
                 canvas_size=(200, 200),
                 overlap_test=False,
                 predict_steps=12,
                 planning_steps=6,
                 past_steps=4,
                 fut_steps=4,
                 use_nonlinear_optimizer=False,
                 lane_ann_file=None,
                 eval_mod=None,

                 # For debug
                 is_debug=False,
                 len_debug=30,

                 # Occ dataset
                 enbale_temporal_aug=False,
                 occ_receptive_field=3,
                 occ_n_future=4,
                 occ_filter_invalid_sample=False,
                 occ_filter_by_valid_flag=False,

                 file_client_args=dict(backend='disk'),
                 *args, 
                 **kwargs):
        # init before super init since it is called in parent class
        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient(**file_client_args)
        intent_file = "/zihan-west-vol/UniAD/data/nuscenes/unified_label_outputs_v2/all_scenes_compact.json"
        with open(intent_file, 'r') as f:
            self.intent_data = json.load(f)
        self.is_debug = is_debug
        self.len_debug = len_debug
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.scene_token = None
        self.lane_infos = self.load_annotations(lane_ann_file) \
            if lane_ann_file else None
        self.eval_mod = eval_mod

        self.use_nonlinear_optimizer = use_nonlinear_optimizer

        self.nusc = NuScenes(version=self.version,
                             dataroot=self.data_root, verbose=True)

        self.map_num_classes = 3
        if canvas_size[0] == 50:
            self.thickness = 1
        # elif canvas_size[0] == 200:

        elif canvas_size[0] == 100 or canvas_size[0] == 200:
            self.thickness = 2
        else:
            assert False
        self.angle_class = 36
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        ps = max(patch_size[0],patch_size[1])
        cs = max(canvas_size[0],canvas_size[1])
        self.map_patch_size = (ps,ps)
        self.map_canvas_size = (cs,cs)
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot=self.data_root, map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot=self.data_root, map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot=self.data_root, map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot=self.data_root, map_name='singapore-queenstown'),
        }
        self.vector_map = VectorizedLocalMap(
            self.data_root,
            patch_size=self.patch_size,
            canvas_size=self.canvas_size)
        self.traj_api = NuScenesTraj(self.nusc,
                                     self.predict_steps,
                                     self.planning_steps,
                                     self.past_steps,
                                     self.fut_steps,
                                     self.with_velocity,
                                     self.CLASSES,
                                     self.box_mode_3d,
                                     self.use_nonlinear_optimizer)

        # Occ
        self.enbale_temporal_aug = enbale_temporal_aug
        assert self.enbale_temporal_aug is False

        self.occ_receptive_field = occ_receptive_field  # past + current
        self.occ_n_future = occ_n_future  # future only
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        self.occ_only_total_frames = 7  # NOTE: hardcode, not influenced by planning

    def __len__(self):
        if not self.is_debug:
            return len(self.data_infos)
        else:
            return self.len_debug

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if self.file_client_args['backend'] == 'disk':
            # data_infos = mmcv.load(ann_file)
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        elif self.file_client_args['backend'] == 'petrel':
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        else:
            assert False, 'Invalid file_client_args!'
        return data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation. 
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_globals_3d: gt_globals of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame (list)
        """
        data_queue = []
        self.enbale_temporal_aug = False
        if self.enbale_temporal_aug:
            # temporal aug
            prev_indexs_list = list(range(index-self.queue_length, index))
            random.shuffle(prev_indexs_list)
            prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
            input_dict = self.get_data_info(index)
        else:
            # ensure the first and final frame in same scene
            final_index = index
            first_index = index - self.queue_length + 1
            if first_index < 0:
                return None
            if self.data_infos[first_index]['scene_token'] != \
                    self.data_infos[final_index]['scene_token']:
                return None
            # current timestamp
            input_dict = self.get_data_info(final_index)
            prev_indexs_list = list(reversed(range(first_index, final_index)))
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        # plot_gt_3d(input_dict['img_filename'], example['gt_bboxes_3d'].data, example['gt_labels_3d'].data, out_path=f"/zihan-west-vol/UniAD/projects/work_dirs/3d_vis/{frame_idx}.png", dpi=250)
        assert example['gt_labels_3d'].data.shape[0] == example['gt_fut_traj'].shape[0]
        assert example['gt_labels_3d'].data.shape[0] == example['gt_past_traj'].shape[0]

        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        data_queue.insert(0, example)

        # retrieve previous infos

        for i in prev_indexs_list:
            if self.enbale_temporal_aug:
                i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            assert example['gt_labels_3d'].data.shape[0] == example['gt_fut_traj'].shape[0]
            assert example['gt_labels_3d'].data.shape[0] == example['gt_past_traj'].shape[0]
            data_queue.insert(0, copy.deepcopy(example))
        data_queue = self.union2one(data_queue)
        return data_queue

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_labels_3d: gt_labels of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame(list)
        """

        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_dict = {}
        for key, value in example.items():
            if 'l2g' in key:
                data_dict[key] = to_tensor(value[0])
            else:
                data_dict[key] = value
        return data_dict

    def union2one(self, queue):
        """
        convert sample dict into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
        gt_labels_intent_list = [to_tensor(each['gt_labels_intent']) for each in queue]
        gt_sdc_label_list = [each['gt_sdc_label'].data for each in queue]
        gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]
        gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
        gt_past_traj_list = [to_tensor(each['gt_past_traj']) for each in queue]
        gt_past_traj_mask_list = [
            to_tensor(each['gt_past_traj_mask']) for each in queue]
        gt_sdc_bbox_list = [each['gt_sdc_bbox'].data for each in queue]
        l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]
        l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]
        timestamp_list = [to_tensor(each['timestamp']) for each in queue]
        gt_fut_traj = to_tensor(queue[-1]['gt_fut_traj'])
        gt_fut_traj_mask = to_tensor(queue[-1]['gt_fut_traj_mask'])
        gt_sdc_fut_traj = to_tensor(queue[-1]['gt_sdc_fut_traj'])
        gt_sdc_fut_traj_mask = to_tensor(queue[-1]['gt_sdc_fut_traj_mask'])
        gt_future_boxes_list = queue[-1]['gt_future_boxes']
        gt_future_labels_list = [to_tensor(each)
                                 for each in queue[-1]['gt_future_labels']]

        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        queue['gt_labels_3d'] = DC(gt_labels_3d_list)
        queue['gt_labels_intent'] = DC(gt_labels_intent_list)
        queue['gt_sdc_label'] = DC(gt_sdc_label_list)
        queue['gt_inds'] = DC(gt_inds_list)
        queue['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
        queue['gt_sdc_bbox'] = DC(gt_sdc_bbox_list, cpu_only=True)
        queue['l2g_r_mat'] = DC(l2g_r_mat_list)
        queue['l2g_t'] = DC(l2g_t_list)
        queue['timestamp'] = DC(timestamp_list)
        queue['gt_fut_traj'] = DC(gt_fut_traj)
        queue['gt_fut_traj_mask'] = DC(gt_fut_traj_mask)
        queue['gt_past_traj'] = DC(gt_past_traj_list)
        queue['gt_past_traj_mask'] = DC(gt_past_traj_mask_list)
        queue['gt_future_boxes'] = DC(gt_future_boxes_list, cpu_only=True)
        queue['gt_future_labels'] = DC(gt_future_labels_list)
        return queue

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - gt_inds (np.ndarray): Instance ids of ground truths.
                - gt_fut_traj (np.ndarray): .
                - gt_fut_traj_mask (np.ndarray): .
        """
        info = self.data_infos[index]
        frame_idx = int(info['frame_idx'])
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_inds'][mask]
        gt_tokens = info['gt_ins_tokens'][mask]
        gt_labels_intent = []
        agent_keys = self.intent_data[info['scene_token']].keys()
        for gt_token in gt_tokens:
            if gt_token not in agent_keys:
                gt_labels_intent.append(-1)
            else:
                # print("find key")
                intent_array = self.intent_data[info['scene_token']][gt_token]["labels"]
                gt_labels_intent.append(intent_label(frame_idx, intent_array))
        gt_labels_intent = np.array(gt_labels_intent)
        # print(gt_labels_intent)
        sample = self.nusc.get('sample', info['token'])
        ann_tokens = np.array(sample['anns'])[mask]
        assert ann_tokens.shape[0] == gt_bboxes_3d.shape[0]

        gt_fut_traj, gt_fut_traj_mask, gt_past_traj, gt_past_traj_mask = self.traj_api.get_traj_label(
            info['token'], ann_tokens)

        sdc_vel = self.traj_api.sdc_vel_info[info['token']]
        gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(sdc_vel)
        gt_sdc_fut_traj, gt_sdc_fut_traj_mask = self.traj_api.get_sdc_traj_label(
            info['token'])

        sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(
            info['token'])

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_labels_intent=gt_labels_intent,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
            gt_fut_traj=gt_fut_traj,
            gt_fut_traj_mask=gt_fut_traj_mask,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=gt_sdc_fut_traj,
            gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask,
            command=command,
        )
        assert gt_fut_traj.shape[0] == gt_labels_3d.shape[0]
        assert gt_past_traj.shape[0] == gt_labels_3d.shape[0]
        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        patch_mask = mask_for_polygons(self.canvas_size, fov_degree = 35)
        # semantic format
        lane_info = self.lane_infos[index] if self.lane_infos else None
        # panoptic format
        location = self.nusc.get('log', self.nusc.get(
            'scene', info['scene_token'])['log_token'])['location']
        vectors = self.vector_map.gen_vectorized_samples(location,
                                                         info['ego2global_translation'],
                                                         info['ego2global_rotation'])
        semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(vectors,
                                                                                       self.patch_size,
                                                                                       self.canvas_size,
                                                                                       self.map_num_classes,
                                                                                       self.thickness,
                                                                                       self.angle_class)
        instance_masks = np.rot90(instance_masks, k=-1, axes=(1, 2))
        instance_masks = torch.tensor(instance_masks.copy())
        gt_labels = []
        gt_bboxes = []
        gt_masks = []
        for cls in range(self.map_num_classes):
            for i in np.unique(instance_masks[cls]):
                if i == 0:
                    continue
                gt_mask = (instance_masks[cls] == i).to(torch.uint8)
                gt_mask[patch_mask == 1] = 0
                ys, xs = np.where(gt_mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
                gt_labels.append(cls)
                gt_bboxes.append(gt_bbox)
                gt_masks.append(gt_mask)
        map_mask = obtain_map_info(self.nusc,
                                   self.nusc_maps,
                                   info,
                                   patch_size=self.map_patch_size,
                                   canvas_size=self.map_canvas_size,
                                   layer_names=['lane_divider', 'road_divider'])
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = torch.tensor(map_mask.copy())
        
        for i, gt_mask in enumerate(map_mask[:-1]):
            gt_mask = gt_mask[100:, :]
            gt_mask[patch_mask == 1] = 0
            ys, xs = np.where(gt_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
            gt_labels.append(i + self.map_num_classes)
            gt_bboxes.append(gt_bbox)
            gt_masks.append(gt_mask)
        if len(gt_bboxes) == 0:
            gt_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.long)
            gt_masks  = torch.zeros((0,self.canvas_size[1] , self.canvas_size[0]), dtype=torch.uint8) 
        else:
            gt_bboxes = torch.tensor(np.stack(gt_bboxes))
            gt_labels = torch.tensor(gt_labels)
            gt_masks  = torch.stack(gt_masks)  #（0，100，200）
        # gt_labels = torch.tensor(gt_labels)
        # gt_bboxes = torch.tensor(np.stack(gt_bboxes))
        # gt_masks = torch.stack(gt_masks)
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            map_filename=lane_info['maps']['map_mask'] if lane_info else None,
            gt_lane_labels=gt_labels,
            gt_lane_bboxes=gt_bboxes,
            gt_lane_masks=gt_masks,
        )

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        input_dict.update(
            dict(
                l2g_r_mat=l2g_r_mat.astype(np.float32),
                l2g_t=l2g_t.astype(np.float32)))

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))
        # plot_image_and_lanes(image_paths, vectors, out_path=f"/zihan-west-vol/UniAD/projects/work_dirs/map_vis_v2/{image_paths[0]}.png", dpi=250)
        # plot_gt_masks(image_paths, gt_masks, gt_labels,out_path=f"/zihan-west-vol/UniAD/projects/work_dirs/map_mask/{image_paths[0]}.png", dpi=250)
        
        # if not self.test_mode:
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        # plot_gt_3d(image_paths, annos['gt_bboxes_3d'], annos['gt_labels_3d'], out_path=f"/zihan-west-vol/UniAD/projects/work_dirs/3d_vis/{image_paths[0]}.png", dpi=250)
        if 'sdc_planning' in input_dict['ann_info'].keys():
            input_dict['sdc_planning'] = input_dict['ann_info']['sdc_planning']
            input_dict['sdc_planning_mask'] = input_dict['ann_info']['sdc_planning_mask']
            input_dict['command'] = input_dict['ann_info']['command']
       
        
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        # TODO: Warp all those below occupancy-related codes into a function
        prev_indices, future_indices = self.occ_get_temporal_indices(
            index, self.occ_receptive_field, self.occ_n_future)

        # ego motions of all frames are needed
        all_frames = prev_indices + [index] + future_indices

        # whether invalid frames is present
        # 
        has_invalid_frame = -1 in all_frames[:self.occ_only_total_frames]
        # NOTE: This can only represent 7 frames in total as it influence evaluation
        input_dict['occ_has_invalid_frame'] = has_invalid_frame
        input_dict['occ_img_is_valid'] = np.array(all_frames) >= 0

        # might have None if not in the same sequence
        future_frames = [index] + future_indices

        # get lidar to ego to global transforms for each curr and fut index
        occ_transforms = self.occ_get_transforms(
            future_frames)  # might have None
        input_dict.update(occ_transforms)

        # for (current and) future frames, detection labels are needed
        # generate detection labels for current + future frames
        input_dict['occ_future_ann_infos'] = \
            self.get_future_detection_infos(future_frames)
        return input_dict

    def get_future_detection_infos(self, future_frames):
        detection_ann_infos = []
        for future_frame in future_frames:
            if future_frame >= 0:
                detection_ann_infos.append(
                    self.occ_get_detection_ann_info(future_frame),
                )
            else:
                detection_ann_infos.append(None)
        return detection_ann_infos

    def occ_get_temporal_indices(self, index, receptive_field, n_future):
        current_scene_token = self.data_infos[index]['scene_token']

        # generate the past
        previous_indices = []

        for t in range(- receptive_field + 1, 0):
            index_t = index + t
            if index_t >= 0 and self.data_infos[index_t]['scene_token'] == current_scene_token:
                previous_indices.append(index_t)
            else:
                previous_indices.append(-1)  # for invalid indices

        # generate the future
        future_indices = []

        for t in range(1, n_future + 1):
            index_t = index + t
            if index_t < len(self.data_infos) and self.data_infos[index_t]['scene_token'] == current_scene_token:
                future_indices.append(index_t)
            else:
                # NOTE: How to deal the invalid indices???
                future_indices.append(-1)

        return previous_indices, future_indices

    def occ_get_transforms(self, indices, data_type=torch.float32):
        """
        get l2e, e2g rotation and translation for each valid frame
        """
        l2e_r_mats = []
        l2e_t_vecs = []
        e2g_r_mats = []
        e2g_t_vecs = []

        for index in indices:
            if index == -1:
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
            else:
                info = self.data_infos[index]
                l2e_r = info['lidar2ego_rotation']
                l2e_t = info['lidar2ego_translation']
                e2g_r = info['ego2global_rotation']
                e2g_t = info['ego2global_translation']

                l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix)
                e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix)

                l2e_r_mats.append(l2e_r_mat.to(data_type))
                l2e_t_vecs.append(torch.tensor(l2e_t).to(data_type))
                e2g_r_mats.append(e2g_r_mat.to(data_type))
                e2g_t_vecs.append(torch.tensor(e2g_t).to(data_type))

        res = {
            'occ_l2e_r_mats': l2e_r_mats,
            'occ_l2e_t_vecs': l2e_t_vecs,
            'occ_e2g_r_mats': e2g_r_mats,
            'occ_e2g_t_vecs': e2g_t_vecs,
        }

        return res

    def occ_get_detection_ann_info(self, index):
        info = self.data_infos[index].copy()
        gt_bboxes_3d = info['gt_boxes'].copy()
        gt_names_3d = info['gt_names'].copy()
        gt_ins_inds = info['gt_inds'].copy()

        gt_vis_tokens = info.get('visibility_tokens', None)

        if self.use_valid_flag:
            gt_valid_flag = info['valid_flag']
        else:
            gt_valid_flag = info['num_lidar_pts'] > 0

        assert self.occ_filter_by_valid_flag is False
        if self.occ_filter_by_valid_flag:
            gt_bboxes_3d = gt_bboxes_3d[gt_valid_flag]
            gt_names_3d = gt_names_3d[gt_valid_flag]
            gt_ins_inds = gt_ins_inds[gt_valid_flag]
            gt_vis_tokens = gt_vis_tokens[gt_valid_flag]

        # cls_name to cls_id
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity']
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            # gt_names=gt_names_3d,
            gt_inds=gt_ins_inds,
            gt_vis_tokens=gt_vis_tokens,
        )

        return anns_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        nusc_map_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']

            if 'map' in self.eval_mod:
                map_annos = {}
                for key, value in det['ret_iou'].items():
                    map_annos[key] = float(value.numpy()[0])
                    nusc_map_annos[sample_token] = map_annos

            if 'boxes_3d' not in det:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                                       mapped_class_names,
                                                       self.eval_detection_configs,
                                                       self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                # center_ = box.center.tolist()
                # change from ground height to center height
                # center_[2] = center_[2] + (box.wlh.tolist()[2] / 2.0)
                if name not in ['car', 'truck', 'bus', 'trailer', 'motorcycle',
                                'bicycle', 'pedestrian', ]:
                    continue

                box_ego = boxes_ego[keep_idx[i]]
                trans = box_ego.center
                if 'traj' in det:
                    traj_local = det['traj'][keep_idx[i]].numpy()[..., :2]
                    traj_scores = det['traj_scores'][keep_idx[i]].numpy()
                else:
                    traj_local = np.zeros((0,))
                    traj_scores = np.zeros((0,))
                traj_ego = np.zeros_like(traj_local)
                rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi/2)
                for kk in range(traj_ego.shape[0]):
                    traj_ego[kk] = convert_local_coords_to_global(
                        traj_local[kk], trans, rot)

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=box.token,
                    predict_traj=traj_ego,
                    predict_traj_score=traj_scores,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
            'map_results': nusc_map_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)

        return result_files, tmp_dir

    def _format_bbox_det(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']

            if det is None:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box_det(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                                       mapped_class_names,
                                                       self.eval_detection_configs,
                                                       self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc_det.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results_det(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results_det')
        else:
            tmp_dir = None

        result_files = self._format_bbox_det(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 planning_evaluation_strategy="uniad"):
        """Evaluation in nuScenes protocol.
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if isinstance(results, dict):
            if 'occ_results_computed' in results.keys():
                occ_results_computed = results['occ_results_computed']
                out_metrics = ['iou']

                # pan_eval
                if occ_results_computed.get('pq', None) is not None:
                    out_metrics = ['iou', 'pq', 'sq', 'rq']

                print("Occ-flow Val Results:")
                for panoptic_key in out_metrics:
                    print(panoptic_key)
                    # HERE!! connect
                    print(' & '.join(
                        [f'{x:.1f}' for x in occ_results_computed[panoptic_key]]))

                if 'num_occ' in occ_results_computed.keys() and 'ratio_occ' in occ_results_computed.keys():
                    print(
                        f"num occ evaluated:{occ_results_computed['num_occ']}")
                    print(
                        f"ratio occ evaluated: {occ_results_computed['ratio_occ'] * 100:.1f}%")
            if 'planning_results_computed' in results.keys():
                planning_results_computed = results['planning_results_computed']
                planning_tab = PrettyTable()
                planning_tab.title = f"{planning_evaluation_strategy}'s definition planning metrics"
                planning_tab.field_names = [
                    "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
                for key in planning_results_computed.keys():
                    value = planning_results_computed[key]
                    row_value = []
                    row_value.append(key)
                    for i in range(len(value)):
                        if planning_evaluation_strategy == "stp3":
                            row_value.append("%.4f" % float(value[: i + 1].mean()))
                        elif planning_evaluation_strategy == "uniad":
                            row_value.append("%.4f" % float(value[i]))
                        else:
                            raise ValueError(
                                "planning_evaluation_strategy should be uniad or spt3"
                            )
                    planning_tab.add_row(row_value)
                print(planning_tab)
            results = results['bbox_results']  # get bbox_results

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        result_files_det, tmp_dir = self.format_results_det(
            results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(
                    result_files[name], result_files_det[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(
                result_files, result_files_det)

        if 'map' in self.eval_mod:
            drivable_intersection = 0
            drivable_union = 0
            lanes_intersection = 0
            lanes_union = 0
            divider_intersection = 0
            divider_union = 0
            crossing_intersection = 0
            crossing_union = 0
            contour_intersection = 0
            contour_union = 0
            for i in range(len(results)):
                drivable_intersection += results[i]['ret_iou']['drivable_intersection']
                drivable_union += results[i]['ret_iou']['drivable_union']
                lanes_intersection += results[i]['ret_iou']['lanes_intersection']
                lanes_union += results[i]['ret_iou']['lanes_union']
                divider_intersection += results[i]['ret_iou']['divider_intersection']
                divider_union += results[i]['ret_iou']['divider_union']
                crossing_intersection += results[i]['ret_iou']['crossing_intersection']
                crossing_union += results[i]['ret_iou']['crossing_union']
                contour_intersection += results[i]['ret_iou']['contour_intersection']
                contour_union += results[i]['ret_iou']['contour_union']
            results_dict.update({'drivable_iou': float(drivable_intersection / drivable_union),
                                 'lanes_iou': float(lanes_intersection / lanes_union),
                                 'divider_iou': float(divider_intersection / divider_union),
                                 'crossing_iou': float(crossing_intersection / crossing_union),
                                 'contour_iou': float(contour_intersection / contour_union)})

            print(results_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _evaluate_single(self,
                         result_path,
                         result_path_det,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        # TODO: fix the evaluation pipelines

        output_dir = osp.join(*osp.split(result_path)[:-1])
        output_dir_det = osp.join(output_dir, 'det')
        output_dir_track = osp.join(output_dir, 'track')
        output_dir_motion = osp.join(output_dir, 'motion')
        mmcv.mkdir_or_exist(output_dir_det)
        mmcv.mkdir_or_exist(output_dir_track)
        mmcv.mkdir_or_exist(output_dir_motion)

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        detail = dict()
        if 'det' in self.eval_mod:
            self.nusc_eval = NuScenesEval_custom(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path_det,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_det,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos
            )
            self.nusc_eval.main(plot_examples=0, render_curves=False)
            # record metrics
            metrics = mmcv.load(
                osp.join(
                    output_dir_det,
                    'metrics_summary.json'))
            metric_prefix = f'{result_name}_NuScenes'
            for name in self.CLASSES:
                for k, v in metrics['label_aps'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_AP_dist_{}'.format(
                        metric_prefix, name, k)] = val
                for k, v in metrics['label_tp_errors'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
                for k, v in metrics['tp_errors'].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}'.format(metric_prefix,
                                          self.ErrNameMapping[k])] = val
            detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
            detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']

        if 'track' in self.eval_mod:
            cfg = config_factory("tracking_nips_2019")
            self.nusc_eval_track = TrackingEval(
                config=cfg,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_track,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root
            )
            self.nusc_eval_track.main()
            # record metrics
            metrics = mmcv.load(
                osp.join(
                    output_dir_track,
                    'metrics_summary.json'))
            keys = ['amota', 'amotp', 'recall', 'motar',
                    'gt', 'mota', 'motp', 'mt', 'ml', 'faf',
                    'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
            for key in keys:
                detail['{}/{}'.format(metric_prefix, key)] = metrics[key]

        # if 'map' in self.eval_mod:
        #     for i, ret_iou in enumerate(ret_ious):
        #         detail['iou_{}'.format(i)] = ret_iou

        if 'motion' in self.eval_mod:
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type='motion_category'
            )
            print('-'*50)
            print(
                'Evaluate on motion category, merge class for vehicles and pedestrians...')
            print('evaluate standard motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='standard')
            print('evaluate motion mAP-minFDE metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='motion_map')
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='epa')
            print('-'*50)
            print('Evaluate on detection category...')
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type='detection_category'
            )
            print('evaluate standard motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='standard')
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='motion_map')
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='epa')

        return detail
