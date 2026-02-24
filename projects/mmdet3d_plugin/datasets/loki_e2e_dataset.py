#---------------------------------------------------------------------------------#
# LOKI Dataset for UniAD (v2 - rewritten to follow NuScenes patterns)            #
# Fixes: class name mismatch, gt_inds pop, DC wrapping, can_bus alignment        #
#---------------------------------------------------------------------------------#

import copy
import numpy as np
import torch
import pickle
import os
import random

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import Custom3DDataset
from mmcv.parallel import DataContainer as DC

# Evaluation imports — reuse nuScenes eval algorithms
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import (
    DetectionBox, DetectionMetrics, DetectionMetricDataList)
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.common.utils import center_distance
from pyquaternion import Quaternion
from prettytable import PrettyTable

# Tracking evaluation imports
from nuscenes.eval.tracking.data_classes import (
    TrackingBox, TrackingConfig, TrackingMetrics,
    TrackingMetricData, TrackingMetricDataList)
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import (
    MOT_METRIC_MAP, AVG_METRIC_MAP, TRACKING_METRICS)
from nuscenes.eval.tracking.utils import print_final_metrics
from collections import defaultdict


# Half-FOV for LOKI's 60° front camera (used to filter GT outside camera view)
_LOKI_HALF_FOV_RAD = np.deg2rad(60.0 / 2.0)


def _in_fov(center_xy, half_fov=_LOKI_HALF_FOV_RAD):
    """Check if a 2D point is within the front-camera FOV.

    In the rotated lidar frame: +y = forward, +x = right.
    Returns True if the point is in front AND within ±half_fov of +y.
    """
    x, y = float(center_xy[0]), float(center_xy[1])
    return y > 0 and abs(np.arctan2(x, y)) <= half_fov


class LokiDetectionConfig:
    """Lightweight replacement for nuscenes DetectionConfig.

    The upstream DetectionConfig asserts class_range keys == the 10
    nuScenes detection classes.  This class drops that check so we
    can use LOKI's own class names.
    """

    def __init__(self, class_range, dist_ths, dist_th_tp,
                 min_recall, min_precision, max_boxes_per_sample,
                 mean_ap_weight):
        assert dist_th_tp in dist_ths
        self.class_range = class_range
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight
        self.class_names = list(class_range.keys())


class LokiDetectionBox(DetectionBox):
    """DetectionBox that accepts arbitrary class names.

    The upstream DetectionBox asserts detection_name is one of the 10
    nuScenes DETECTION_NAMES.  We skip that so LOKI class names work.
    """

    def __init__(self, **kwargs):
        from nuscenes.eval.common.data_classes import EvalBox
        # Bypass DetectionBox.__init__ and call EvalBox directly
        EvalBox.__init__(
            self,
            sample_token=kwargs.get('sample_token', ''),
            translation=kwargs.get('translation', (0, 0, 0)),
            size=kwargs.get('size', (0, 0, 0)),
            rotation=kwargs.get('rotation', (0, 0, 0, 0)),
            velocity=kwargs.get('velocity', (0, 0)),
            ego_translation=kwargs.get('ego_translation', (0, 0, 0)),
            num_pts=kwargs.get('num_pts', -1),
        )
        self.detection_name = kwargs.get('detection_name', '')
        self.detection_score = kwargs.get('detection_score', -1.0)
        self.attribute_name = kwargs.get('attribute_name', '')


# ------------------------------------------------------------------------------- #
# PKL stores lowercase mapped names ('car', 'pedestrian', 'truck', etc.)
# Config uses LOKI-native capitalized names ('Car', 'Pedestrian', 'Truck', etc.)
# This map bridges the two.
# ------------------------------------------------------------------------------- #
PKL_TO_CONFIG = {
    'car': 'Car',
    'truck': 'Truck',
    'bus': 'Bus',
    'pedestrian': 'Pedestrian',
    'motorcycle': 'Motorcyclist',
    'bicycle': 'Bicyclist',
}


@DATASETS.register_module()
class LokiE2EDataset(Custom3DDataset):
    """LOKI End-to-End Dataset for UniAD perception pipeline (v2).

    Rewritten to closely follow NuScenesE2EDataset patterns, fixing:
        - Class name case mismatch (pkl lowercase vs config capitalized)
        - gt_inds pop from ann_info (uses LoadAnnotations3D_E2E)
        - Pre-wrapping in DataContainer (now raw, except sdc fields)
        - Proper can_bus delta computation in union2one

    Key differences from NuScenesE2EDataset:
        - Single front-view camera (num_cams=1) instead of 6 surround cameras
        - No NuScenes SDK dependency
        - No HD map data (dummy map labels)
        - Perception-only: zero-filled trajectory/planning/occupancy labels
        - ego2global_rotation is 3x3 matrix (not quaternion)
        - Timestamps in seconds (not microseconds)
    """

    CLASSES = ('Pedestrian', 'Car', 'Bus', 'Truck',
               'Van', 'Motorcyclist', 'Bicyclist', 'Other')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root='',
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 test_mode=False,
                 queue_length=3,
                 bev_size=(200, 200),
                 predict_steps=12,
                 planning_steps=6,
                 past_steps=4,
                 fut_steps=4,
                 use_valid_flag=True,
                 filter_empty_gt=True,

                 # Occ settings (kept for pipeline compat, unused)
                 occ_receptive_field=3,
                 occ_n_future=4,
                 occ_filter_invalid_sample=False,

                 # Debug
                 is_debug=False,
                 len_debug=30,
                 **kwargs):
        # --- Manual init (cannot use Custom3DDataset.__init__ directly) ---
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        from mmdet3d.core.bbox import get_box_type
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.CLASSES = classes if classes else self.__class__.CLASSES
        self.modality = modality or dict(
            use_lidar=False, use_camera=True,
            use_radar=False, use_map=False, use_external=False)
        self.use_valid_flag = use_valid_flag
        self.with_velocity = True

        self.queue_length = queue_length
        self.bev_size = bev_size
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps

        self.occ_receptive_field = occ_receptive_field
        self.occ_n_future = occ_n_future
        self.occ_filter_invalid_sample = occ_filter_invalid_sample

        self.is_debug = is_debug
        self.len_debug = len_debug

        # Load annotations
        self.data_infos = self._load_annotations(ann_file)

        # Build scene index for temporal access
        self._build_scene_index()

        # Set up pipeline
        if pipeline is not None:
            from mmdet3d.datasets.pipelines import Compose
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        # Flag for dataloader grouping (all same aspect ratio)
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #

    def _load_annotations(self, ann_file):
        """Load annotations from pickle file."""
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(
            data['infos'], key=lambda e: (e['scene_token'], e['frame_idx'])))
        self.metadata = data['metadata']
        return data_infos

    def _build_scene_index(self):
        """Build mapping from scene_token to list of indices."""
        self.scene_to_indices = {}
        for idx, info in enumerate(self.data_infos):
            scene = info['scene_token']
            if scene not in self.scene_to_indices:
                self.scene_to_indices[scene] = []
            self.scene_to_indices[scene].append(idx)

    def __len__(self):
        if self.is_debug:
            return min(self.len_debug, len(self.data_infos))
        return len(self.data_infos)

    # ------------------------------------------------------------------ #
    #  get_ann_info  (mirrors NuScenesE2EDataset.get_ann_info)
    # ------------------------------------------------------------------ #

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        - Maps pkl lowercase names to config capitalized names via PKL_TO_CONFIG
        - Uses zero-filled trajectories (no traj_api)
        - Generates SDC pseudo-box inline (no NuScenes SDK)

        Returns:
            dict: Annotation info with gt_bboxes_3d, gt_labels_3d,
                  gt_inds, trajectories, SDC info, planning placeholders.
        """
        info = self.data_infos[index]

        # Validity mask (same as NuScenes)
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0

        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_inds'][mask]
        gt_camera_visible = info.get('gt_camera_visible', np.ones(len(info['gt_boxes']), dtype=bool))[mask]

        # Map pkl lowercase names to config capitalized names
        gt_labels_3d = []
        for cat in gt_names_3d:
            config_name = PKL_TO_CONFIG.get(cat, None)
            if config_name and config_name in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(config_name))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # Velocity: already in gt_boxes columns 7,8 (no NaN handling needed)
        # LOKI pkl already has velocity in gt_boxes, so no concatenation needed.

        # Rotate 90° CCW to align with nuScenes convention:
        # LOKI: x=forward, y=left/right → nuScenes: x=left/right, y=forward
        # Transform: new_x = -old_y, new_y = old_x
        if len(gt_bboxes_3d) > 0:
            old_x = gt_bboxes_3d[:, 0].copy()
            old_y = gt_bboxes_3d[:, 1].copy()
            gt_bboxes_3d[:, 0] = -old_y   # new x = -old y
            gt_bboxes_3d[:, 1] = old_x    # new y = old x
            gt_bboxes_3d[:, 6] += 0.5 * np.pi  # yaw rotated CCW by 90°
            if gt_bboxes_3d.shape[-1] >= 9:
                old_vx = gt_bboxes_3d[:, 7].copy()
                old_vy = gt_bboxes_3d[:, 8].copy()
                gt_bboxes_3d[:, 7] = -old_vy  # new vx = -old vy
                gt_bboxes_3d[:, 8] = old_vx   # new vy = old vx

        # Create LiDARInstance3DBoxes (same as NuScenes)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1] if len(gt_bboxes_3d) > 0 else 9,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # Zero-filled trajectories (NuScenes uses traj_api.get_traj_label)
        n_valid = len(gt_labels_3d)
        # Intent labels are unavailable in LOKI. Keep a dummy vector so
        # UniAD pipeline transforms that expect this key remain compatible.
        gt_labels_intent = np.zeros((n_valid,), dtype=np.int64)
        gt_fut_traj = np.zeros((n_valid, self.predict_steps, 2), dtype=np.float32)
        gt_fut_traj_mask = np.zeros((n_valid, self.predict_steps, 2), dtype=np.float32)
        gt_past_traj = np.zeros((n_valid, self.past_steps + self.fut_steps, 2), dtype=np.float32)
        gt_past_traj_mask = np.zeros((n_valid, self.past_steps + self.fut_steps, 2), dtype=np.float32)

        # SDC (ego vehicle) info -- DC-wrapped, matching NuScenes traj_api.generate_sdc_info()
        # SDC is at origin, yaw=pi (facing +y after 90° CCW rotation, same as nuScenes)
        sdc_vel = np.zeros(2, dtype=np.float32)
        psudo_sdc_bbox = np.array(
            [[0.0, 0.0, 0.0, 4.08, 1.73, 1.56, np.pi,
              sdc_vel[0], sdc_vel[1]]], dtype=np.float32)
        gt_sdc_bbox = LiDARInstance3DBoxes(
            psudo_sdc_bbox, box_dim=9,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        # DC-wrapped just like NuScenes traj_api.generate_sdc_info returns
        gt_sdc_bbox = DC(gt_sdc_bbox, cpu_only=True)
        # SDC label: index of 'Car' in CLASSES
        car_idx = self.CLASSES.index('Car') if 'Car' in self.CLASSES else 0
        gt_sdc_label = DC(to_tensor(np.array([car_idx])))

        gt_sdc_fut_traj = np.zeros((1, self.predict_steps, 2), dtype=np.float32)
        gt_sdc_fut_traj_mask = np.zeros((1, self.predict_steps, 2), dtype=np.float32)

        # Planning placeholders
        sdc_planning = np.zeros((1, self.planning_steps, 3), dtype=np.float32)
        sdc_planning_mask = np.zeros((1, self.planning_steps, 2), dtype=np.float32)
        command = 2  # FORWARD

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_labels_intent=gt_labels_intent,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,  # Will be POPPED by LoadAnnotations3D_E2E (fixes BUG #2)
            gt_camera_visible=gt_camera_visible,
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

    # ------------------------------------------------------------------ #
    #  get_data_info  (mirrors NuScenesE2EDataset.get_data_info)
    # ------------------------------------------------------------------ #

    def get_data_info(self, index):
        """Get data info dict for a single frame.

        Closely follows NuScenesE2EDataset.get_data_info but:
        - Single camera (list of 1) instead of 6
        - No NuScenes map/vector_map data
        - ego2global_rotation is 3x3 (not quaternion)
        - can_bus pre-computed in pkl (no quaternion conversion)
        - Dummy occ transforms (no real occ data)
        """
        info = self.data_infos[index]

        # --- Dummy map labels (NuScenes has nusc_maps, vector_map, etc.) ---
        gt_lane_labels = torch.zeros(0, dtype=torch.long)
        gt_lane_bboxes = torch.zeros((0, 4), dtype=torch.float32)
        gt_lane_masks = torch.zeros(
            (0, self.bev_size[0], self.bev_size[1]), dtype=torch.uint8)

        # --- Build input_dict (same structure as NuScenes) ---
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'].copy(),
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'],  # seconds (NuScenes divides by 1e6)
            gt_lane_labels=gt_lane_labels,
            gt_lane_bboxes=gt_lane_bboxes,
            gt_lane_masks=gt_lane_masks,
        )

        # l2g_t is in global frame, unchanged by lidar rotation
        input_dict['l2g_t'] = info['l2g_t'].astype(np.float32)

        # -------------------------------------------------------------- #
        # 90° CCW rotation matrix: transforms new lidar → old lidar
        # Our coord change: new_x = -old_y, new_y = old_x
        # Inverse (new→old): old_x = new_y, old_y = -new_x
        # -------------------------------------------------------------- #
        R_inv_3x3 = np.array([
            [ 0., 1., 0.],
            [-1., 0., 0.],
            [ 0., 0., 1.],
        ], dtype=np.float64)
        R_inv_4x4 = np.eye(4, dtype=np.float64)
        R_inv_4x4[:3, :3] = R_inv_3x3

        # Rotate l2g_r_mat: l2g_new = l2g_old @ R_inv
        l2g_r_mat = info['l2g_r_mat'].astype(np.float64)
        input_dict['l2g_r_mat'] = (l2g_r_mat @ R_inv_3x3).astype(np.float32)

        # Camera data: single camera as list of 1 (NuScenes iterates info['cams'])
        if self.modality['use_camera']:
            # Rotate lidar2img and lidar2cam: M_new = M_old @ R_inv
            lidar2img = info['lidar2img'].copy().astype(np.float64)
            lidar2cam = info['lidar2cam'].copy().astype(np.float64)
            input_dict.update(dict(
                img_filename=[info['img_filename']],
                lidar2img=[(lidar2img @ R_inv_4x4).astype(np.float64)],
                cam_intrinsic=[info['cam_intrinsic'].copy()],
                lidar2cam=[(lidar2cam @ R_inv_4x4).astype(np.float64)],
            ))

        # Annotations (matches NuScenes pattern: get_ann_info -> ann_info)
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        # Copy planning to top level (same as NuScenes lines 548-551)
        if 'sdc_planning' in annos:
            input_dict['sdc_planning'] = annos['sdc_planning']
            input_dict['sdc_planning_mask'] = annos['sdc_planning_mask']
            input_dict['command'] = annos['command']

        # Camera visibility flag (for ObjectCameraVisibleFilter)
        if 'gt_camera_visible' in annos:
            input_dict['gt_camera_visible'] = annos['gt_camera_visible']

        # can_bus: position [:3] is in global frame (unchanged).
        # Yaw [-1] is ego heading in global frame (unchanged).
        # The l2g_r_mat rotation handles converting global deltas
        # to the new lidar frame in the transformer.

        # --- Occ placeholders (NuScenes calls get_occ_data_infos) ---
        occ_frames = self._get_occ_temporal_indices(index)
        input_dict['occ_has_invalid_frame'] = any(f < 0 for f in occ_frames)
        input_dict['occ_img_is_valid'] = np.array([f >= 0 for f in occ_frames])

        n_future = self.occ_n_future + 1  # current + future
        input_dict['occ_l2e_r_mats'] = [torch.eye(3, dtype=torch.float32)] * n_future
        input_dict['occ_l2e_t_vecs'] = [torch.zeros(3, dtype=torch.float32)] * n_future
        input_dict['occ_e2g_r_mats'] = [
            torch.from_numpy(info['ego2global_rotation']).float()] * n_future
        input_dict['occ_e2g_t_vecs'] = [
            torch.from_numpy(info['ego2global_translation']).float()] * n_future

        # Future detection ann infos (for occ flow - all None since occ disabled)
        input_dict['occ_future_ann_infos'] = [None] * n_future

        return input_dict

    # ------------------------------------------------------------------ #
    #  Temporal helpers
    # ------------------------------------------------------------------ #

    def _get_occ_temporal_indices(self, index):
        """Get temporal indices for occ (7 frames total)."""
        current_scene = self.data_infos[index]['scene_token']
        total_frames = 7
        indices = []
        for t in range(-(self.occ_receptive_field - 1), self.occ_n_future + 1):
            idx_t = index + t
            if 0 <= idx_t < len(self.data_infos) and \
               self.data_infos[idx_t]['scene_token'] == current_scene:
                indices.append(idx_t)
            else:
                indices.append(-1)
        return indices[:total_frames]

    # ------------------------------------------------------------------ #
    #  prepare_train_data  (mirrors NuScenesE2EDataset.prepare_train_data)
    # ------------------------------------------------------------------ #

    def pre_pipeline(self, results):
        """Initialization before data pipeline."""
        results['img_prefix'] = ''
        results['seg_prefix'] = None
        results['proposal_file'] = None
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation with temporal queue.

        Mirrors NuScenesE2EDataset.prepare_train_data exactly.
        """
        data_queue = []

        # Ensure first and last frame are in the same scene
        final_index = index
        first_index = index - self.queue_length + 1
        if first_index < 0:
            return None
        if self.data_infos[first_index]['scene_token'] != \
                self.data_infos[final_index]['scene_token']:
            return None

        # Process current (final) frame
        input_dict = self.get_data_info(final_index)
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        if example is None:
            return None

        # Validate shapes (same assertions as NuScenes lines 209, 232-233)
        assert example['gt_labels_3d'].data.shape[0] == example['gt_fut_traj'].shape[0]
        assert example['gt_labels_3d'].data.shape[0] == example['gt_past_traj'].shape[0]

        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None

        data_queue.insert(0, example)

        # Retrieve previous frames (in reverse order)
        prev_indexs_list = list(reversed(range(first_index, final_index)))
        for i in prev_indexs_list:
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and \
               input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if example is None:
                    return None
                if self.filter_empty_gt and \
                        (example is None or
                         ~(example['gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            assert example['gt_labels_3d'].data.shape[0] == example['gt_fut_traj'].shape[0]
            assert example['gt_labels_3d'].data.shape[0] == example['gt_past_traj'].shape[0]
            data_queue.insert(0, copy.deepcopy(example))

        data_queue = self.union2one(data_queue)
        return data_queue

    def prepare_test_data(self, index):
        """Prepare test data for a single frame."""
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_dict = {}
        for key, value in example.items():
            if 'l2g' in key:
                data_dict[key] = to_tensor(value[0]) if isinstance(
                    value, (list, tuple)) else to_tensor(value)
            else:
                data_dict[key] = value
        return data_dict

    # ------------------------------------------------------------------ #
    #  union2one  (mirrors NuScenesE2EDataset.union2one exactly)
    # ------------------------------------------------------------------ #

    def union2one(self, queue):
        """Merge temporal queue into a single sample dict.

        Directly mirrors NuScenesE2EDataset.union2one, including the
        can_bus delta computation for temporal BEV alignment.
        """
        imgs_list = [each['img'].data for each in queue]
        gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
        gt_sdc_label_list = [each['gt_sdc_label'].data for each in queue]
        gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]
        gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
        gt_past_traj_list = [to_tensor(each['gt_past_traj']) for each in queue]
        gt_past_traj_mask_list = [
            to_tensor(each['gt_past_traj_mask']) for each in queue]
        gt_sdc_bbox_list = [each['gt_sdc_bbox'].data for each in queue]
        l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]
        l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]
        timestamp_list = [
            torch.tensor([each['timestamp']], dtype=torch.float64)
            for each in queue]

        gt_fut_traj = to_tensor(queue[-1]['gt_fut_traj'])
        gt_fut_traj_mask = to_tensor(queue[-1]['gt_fut_traj_mask'])
        gt_sdc_fut_traj = to_tensor(queue[-1]['gt_sdc_fut_traj'])
        gt_sdc_fut_traj_mask = to_tensor(queue[-1]['gt_sdc_fut_traj_mask'])

        # Dummy future boxes for occ (empty, same as NuScenes structure)
        gt_future_boxes_list = []
        gt_future_labels_list = []

        # can_bus delta computation (identical to NuScenes lines 291-306)
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

    # ------------------------------------------------------------------ #
    #  __getitem__  (identical to NuScenes)
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx):
        """Get item from dataset."""
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _rand_another(self, idx):
        """Randomly get another sample."""
        pool = list(range(len(self)))
        pool.remove(idx)
        return random.choice(pool) if pool else idx

    # ------------------------------------------------------------------ #
    #  Evaluation — nuScenes-style detection metrics for LOKI
    # ------------------------------------------------------------------ #

    def evaluate(self, results, **kwargs):
        """Evaluate detection results using nuScenes-style metrics.

        Reuses nuscenes.eval.detection.algo (accumulate, calc_ap, calc_tp)
        to compute per-class AP at multiple distance thresholds, mAP,
        and TP error metrics (ATE, ASE, AOE, AVE).

        All comparisons are done in the rotated lidar frame (the frame
        the model operates in), so no global-frame transform is needed.

        Args:
            results: Either a list of per-frame detection dicts, or a dict
                with 'bbox_results' key containing such a list.

        Returns:
            dict: Detection metrics (per-class AP, mAP, NDS, TP errors).
        """
        if isinstance(results, dict):
            bbox_results = results.get('bbox_results', [])
        else:
            bbox_results = results

        assert len(bbox_results) == len(self), (
            f'Results length {len(bbox_results)} != dataset length {len(self)}')

        # --- detection evaluation ---
        eval_cfg = self._get_eval_config()
        gt_boxes = self._build_gt_eval_boxes(eval_cfg)
        pred_boxes = self._build_pred_eval_boxes(bbox_results, eval_cfg)
        detail = self._run_detection_eval(gt_boxes, pred_boxes, eval_cfg)

        # --- tracking evaluation ---
        track_detail = self._run_tracking_eval(bbox_results)
        detail.update(track_detail)

        return detail

    # ----- eval helpers ------------------------------------------------ #

    def _get_eval_config(self):
        """Create a DetectionConfig for LOKI classes.

        Uses the same distance thresholds and AP calculation parameters
        as the standard nuScenes CVPR-2019 config, but with LOKI class
        names and appropriate detection ranges for a single front camera.
        """
        class_range = {name: 50.0 for name in self.CLASSES}
        # Shorter range for small / vulnerable road users
        for short_cls in ('Pedestrian', 'Bicyclist', 'Other'):
            if short_cls in class_range:
                class_range[short_cls] = 40.0

        return LokiDetectionConfig(
            class_range=class_range,
            dist_ths=[0.5, 1.0, 2.0, 4.0],
            dist_th_tp=2.0,
            min_recall=0.1,
            min_precision=0.1,
            max_boxes_per_sample=500,
            mean_ap_weight=5,
        )

    def _build_gt_eval_boxes(self, eval_cfg):
        """Build ground-truth EvalBoxes from the val pkl.

        Applies the same 90-deg CCW rotation used in get_ann_info() so
        that GT and predictions are in the same (rotated-lidar) frame.
        """
        gt_eval = EvalBoxes()

        for idx in range(len(self)):
            info = self.data_infos[idx]
            token = info['token']

            mask = info['valid_flag'] if self.use_valid_flag \
                else info['num_lidar_pts'] > 0
            boxes_raw = info['gt_boxes'][mask].copy()
            names_raw = info['gt_names'][mask]

            # --- 90-deg CCW rotation (identical to get_ann_info) ---
            if len(boxes_raw) > 0:
                ox, oy = boxes_raw[:, 0].copy(), boxes_raw[:, 1].copy()
                boxes_raw[:, 0] = -oy
                boxes_raw[:, 1] = ox
                boxes_raw[:, 6] += 0.5 * np.pi
                if boxes_raw.shape[-1] >= 9:
                    ovx, ovy = boxes_raw[:, 7].copy(), boxes_raw[:, 8].copy()
                    boxes_raw[:, 7] = -ovy
                    boxes_raw[:, 8] = ovx

            sample_boxes = []
            for i in range(len(boxes_raw)):
                det_name = PKL_TO_CONFIG.get(names_raw[i], None)
                if det_name is None or det_name not in self.CLASSES:
                    continue

                center = boxes_raw[i, :3]
                if np.linalg.norm(center[:2]) > eval_cfg.class_range.get(
                        det_name, 50.0):
                    continue

                # Skip GT outside the front camera's 60° FOV
                if not _in_fov(center[:2]):
                    continue

                quat = Quaternion(axis=[0, 0, 1],
                                  radians=float(boxes_raw[i, 6]))
                vel = (float(boxes_raw[i, 7]),
                       float(boxes_raw[i, 8])) \
                    if boxes_raw.shape[-1] >= 9 else (0.0, 0.0)

                sample_boxes.append(LokiDetectionBox(
                    sample_token=token,
                    translation=tuple(center.tolist()),
                    size=tuple(boxes_raw[i, 3:6].tolist()),
                    rotation=tuple(quat.elements.tolist()),
                    velocity=vel,
                    ego_translation=tuple(center.tolist()),  # ego at origin
                    num_pts=-1,
                    detection_name=det_name,
                    detection_score=-1.0,
                    attribute_name='',
                ))

            gt_eval.add_boxes(token, sample_boxes)

        return gt_eval

    def _build_pred_eval_boxes(self, results, eval_cfg):
        """Build prediction EvalBoxes from model outputs.

        Model outputs are already in the rotated-lidar frame.
        """
        pred_eval = EvalBoxes()

        for sample_id, det in enumerate(results):
            token = self.data_infos[sample_id]['token']
            sample_boxes = []

            if not isinstance(det, dict) or 'boxes_3d' not in det:
                pred_eval.add_boxes(token, sample_boxes)
                continue

            # Prefer detection-specific outputs (pre-tracking)
            boxes_3d = det.get('boxes_3d_det', det['boxes_3d'])
            scores_3d = det.get('scores_3d_det', det['scores_3d'])
            labels_3d = det.get('labels_3d_det', det['labels_3d'])

            if hasattr(boxes_3d, 'tensor'):
                tensor = boxes_3d.tensor
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                centers = boxes_3d.gravity_center.cpu().numpy() \
                    if boxes_3d.gravity_center.is_cuda \
                    else boxes_3d.gravity_center.numpy()
                dims = boxes_3d.dims.cpu().numpy() \
                    if boxes_3d.dims.is_cuda \
                    else boxes_3d.dims.numpy()
                yaws = boxes_3d.yaw.cpu().numpy() \
                    if boxes_3d.yaw.is_cuda \
                    else boxes_3d.yaw.numpy()
                vels = tensor[:, 7:9].numpy() if tensor.shape[-1] > 7 \
                    else np.zeros((len(boxes_3d), 2))
            else:
                pred_eval.add_boxes(token, sample_boxes)
                continue

            scores = scores_3d.cpu().numpy() if hasattr(scores_3d, 'cpu') \
                else np.asarray(scores_3d)
            labels = labels_3d.cpu().numpy() if hasattr(labels_3d, 'cpu') \
                else np.asarray(labels_3d)

            for i in range(len(boxes_3d)):
                label_idx = int(labels[i])
                if label_idx < 0 or label_idx >= len(self.CLASSES):
                    continue
                det_name = self.CLASSES[label_idx]
                if np.linalg.norm(centers[i, :2]) > eval_cfg.class_range.get(
                        det_name, 50.0):
                    continue

                # Skip predictions outside the front camera's 60° FOV
                if not _in_fov(centers[i, :2]):
                    continue

                quat = Quaternion(axis=[0, 0, 1],
                                  radians=float(yaws[i]))
                sample_boxes.append(LokiDetectionBox(
                    sample_token=token,
                    translation=tuple(centers[i].tolist()),
                    size=tuple(dims[i].tolist()),
                    rotation=tuple(quat.elements.tolist()),
                    velocity=(float(vels[i, 0]), float(vels[i, 1])),
                    ego_translation=tuple(centers[i].tolist()),
                    num_pts=-1,
                    detection_name=det_name,
                    detection_score=float(scores[i]),
                    attribute_name='',
                ))

            # Cap boxes per sample (nuscenes convention)
            if len(sample_boxes) > eval_cfg.max_boxes_per_sample:
                sample_boxes.sort(key=lambda b: b.detection_score,
                                  reverse=True)
                sample_boxes = sample_boxes[:eval_cfg.max_boxes_per_sample]

            pred_eval.add_boxes(token, sample_boxes)

        return pred_eval

    def _run_detection_eval(self, gt_boxes, pred_boxes, eval_cfg):
        """Run nuScenes-style detection evaluation.

        Uses accumulate / calc_ap / calc_tp from the nuscenes-devkit.
        """
        # Ensure both have identical sample-token sets
        all_tokens = set(gt_boxes.sample_tokens) | \
            set(pred_boxes.sample_tokens)
        for t in all_tokens:
            if t not in gt_boxes.boxes:
                gt_boxes.add_boxes(t, [])
            if t not in pred_boxes.boxes:
                pred_boxes.add_boxes(t, [])

        # ---- Step 1: accumulate metric data per class / dist_th ---- #
        print('Accumulating metric data ...')
        md_list = DetectionMetricDataList()
        for cls in eval_cfg.class_names:
            for dist_th in eval_cfg.dist_ths:
                md = accumulate(gt_boxes, pred_boxes, cls,
                                center_distance, dist_th)
                md_list.set(cls, dist_th, md)

        # ---- Step 2: compute AP and TP metrics ---- #
        print('Computing metrics ...')
        metrics = DetectionMetrics(eval_cfg)
        for cls in eval_cfg.class_names:
            for dist_th in eval_cfg.dist_ths:
                md = md_list[(cls, dist_th)]
                ap = calc_ap(md, eval_cfg.min_recall,
                             eval_cfg.min_precision)
                metrics.add_label_ap(cls, dist_th, ap)

            for metric_name in TP_METRICS:
                md = md_list[(cls, eval_cfg.dist_th_tp)]
                if metric_name == 'attr_err':
                    tp = 1.0          # LOKI has no attributes → worst err
                else:
                    tp = calc_tp(md, eval_cfg.min_recall, metric_name)
                metrics.add_label_tp(cls, metric_name, tp)

        # ---- Step 3: print & return ---- #
        detail = self._print_detection_results(metrics, eval_cfg)
        return detail

    def _print_detection_results(self, metrics, eval_cfg):
        """Pretty-print detection results and return a flat metrics dict."""
        print('\n' + '=' * 70)
        print('  LOKI Detection Evaluation Results')
        print('=' * 70)

        # --- per-class AP table ---
        ap_tab = PrettyTable()
        ap_tab.field_names = (['Class']
                              + [f'd={d}' for d in eval_cfg.dist_ths]
                              + ['Mean'])
        for cls in eval_cfg.class_names:
            row = [cls]
            aps = []
            for d in eval_cfg.dist_ths:
                v = metrics.get_label_ap(cls, d)
                row.append(f'{v:.4f}')
                aps.append(v)
            row.append(f'{np.mean(aps):.4f}')
            ap_tab.add_row(row)
        print('\nPer-class Average Precision:')
        print(ap_tab)

        # --- TP error table (skip attr_err) ---
        tp_names = [m for m in TP_METRICS if m != 'attr_err']
        tp_tab = PrettyTable()
        tp_tab.field_names = ['Class'] + tp_names
        for cls in eval_cfg.class_names:
            row = [cls]
            for m in tp_names:
                v = metrics.get_label_tp(cls, m)
                row.append(f'{v:.4f}' if not np.isnan(v) else 'N/A')
            tp_tab.add_row(row)
        print('\nTrue Positive Errors (lower is better):')
        print(tp_tab)

        # --- summary ---
        mean_ap = float(metrics.mean_ap)
        nd_score = float(metrics.nd_score)
        print(f'\nmAP : {mean_ap:.4f}')
        print(f'NDS : {nd_score:.4f}')

        # Mean TP errors
        tp_errors = metrics.tp_errors
        for m in tp_names:
            print(f'm{m.replace("_err","").upper()+"E"}: {tp_errors[m]:.4f}')
        print('=' * 70 + '\n')

        # --- build flat dict for mmdet3d logger ---
        detail = {}
        for cls in eval_cfg.class_names:
            for d in eval_cfg.dist_ths:
                detail[f'{cls}/AP_d{d}'] = float(
                    metrics.get_label_ap(cls, d))
            for m in tp_names:
                v = metrics.get_label_tp(cls, m)
                if not np.isnan(v):
                    detail[f'{cls}/{m}'] = float(v)
        detail['mAP'] = mean_ap
        detail['NDS'] = nd_score
        for m in tp_names:
            detail[f'm{m}'] = float(tp_errors[m])
        return detail

    # ================================================================== #
    #  Tracking Evaluation
    # ================================================================== #

    def _get_tracking_config(self):
        """Create a TrackingConfig for LOKI classes.

        Creating this config sets the module-global TRACKING_NAMES in
        nuscenes.eval.tracking.data_classes, which allows TrackingBox
        to accept LOKI class names.
        """
        tracking_names = list(self.CLASSES)
        class_range = {name: 50.0 for name in tracking_names}
        for short_cls in ('Pedestrian', 'Bicyclist', 'Other'):
            if short_cls in class_range:
                class_range[short_cls] = 40.0

        metric_worst = {
            'amota': 0.0, 'amotp': 2.0, 'recall': 0.0, 'motar': 0.0,
            'mota': 0.0, 'motp': 2.0, 'mt': 0.0, 'ml': -1.0,
            'faf': 500, 'gt': -1, 'tp': 0.0, 'fp': -1.0, 'fn': -1.0,
            'ids': -1.0, 'frag': -1.0, 'tid': 20, 'lgd': 20,
        }

        return TrackingConfig(
            tracking_names=tracking_names,
            pretty_tracking_names={n: n for n in tracking_names},
            tracking_colors={n: 'C0' for n in tracking_names},
            class_range=class_range,
            dist_fcn='center_distance',
            dist_th_tp=2.0,
            min_recall=0.1,
            max_boxes_per_sample=500,
            metric_worst=metric_worst,
            num_thresholds=40,
        )

    def _build_gt_tracks(self, track_cfg):
        """Build GT tracks: {scene_token: {timestamp: [TrackingBox]}}.

        Uses gt_inds from the pkl as tracking IDs and applies the same
        90-deg rotation as get_ann_info().
        """
        tracks = defaultdict(lambda: defaultdict(list))

        for idx in range(len(self)):
            info = self.data_infos[idx]
            scene = info['scene_token']
            timestamp = idx   # sequential index as timestamp

            mask = info['valid_flag'] if self.use_valid_flag \
                else info['num_lidar_pts'] > 0
            boxes_raw = info['gt_boxes'][mask].copy()
            names_raw = info['gt_names'][mask]
            inds_raw = info['gt_inds'][mask]

            # 90-deg CCW rotation (same as get_ann_info / _build_gt_eval_boxes)
            if len(boxes_raw) > 0:
                ox, oy = boxes_raw[:, 0].copy(), boxes_raw[:, 1].copy()
                boxes_raw[:, 0] = -oy
                boxes_raw[:, 1] = ox
                boxes_raw[:, 6] += 0.5 * np.pi
                if boxes_raw.shape[-1] >= 9:
                    ovx, ovy = boxes_raw[:, 7].copy(), boxes_raw[:, 8].copy()
                    boxes_raw[:, 7] = -ovy
                    boxes_raw[:, 8] = ovx

            frame_boxes = []
            for i in range(len(boxes_raw)):
                det_name = PKL_TO_CONFIG.get(names_raw[i], None)
                if det_name is None or det_name not in self.CLASSES:
                    continue
                center = boxes_raw[i, :3]
                if np.linalg.norm(center[:2]) > track_cfg.class_range.get(
                        det_name, 50.0):
                    continue

                # Skip GT outside the front camera's 60° FOV
                if not _in_fov(center[:2]):
                    continue

                quat = Quaternion(axis=[0, 0, 1],
                                  radians=float(boxes_raw[i, 6]))
                vel = (float(boxes_raw[i, 7]),
                       float(boxes_raw[i, 8])) \
                    if boxes_raw.shape[-1] >= 9 else (0.0, 0.0)

                frame_boxes.append(TrackingBox(
                    sample_token=info['token'],
                    translation=tuple(center.tolist()),
                    size=tuple(boxes_raw[i, 3:6].tolist()),
                    rotation=tuple(quat.elements.tolist()),
                    velocity=vel,
                    ego_translation=tuple(center.tolist()),
                    num_pts=-1,
                    tracking_id=str(int(inds_raw[i])),
                    tracking_name=det_name,
                    tracking_score=-1.0,
                ))

            tracks[scene][timestamp] = frame_boxes

        return dict(tracks)

    def _build_pred_tracks(self, results, track_cfg):
        """Build predicted tracks: {scene_token: {timestamp: [TrackingBox]}}.

        Uses track_ids from model output as tracking IDs.
        Falls back to boxes_3d (post-tracking) rather than boxes_3d_det.
        """
        tracks = defaultdict(lambda: defaultdict(list))

        for sample_id, det in enumerate(results):
            info = self.data_infos[sample_id]
            scene = info['scene_token']
            timestamp = sample_id

            if not isinstance(det, dict) or 'boxes_3d' not in det:
                tracks[scene][timestamp] = []
                continue

            # Use post-tracking outputs (boxes_3d, not boxes_3d_det)
            boxes_3d = det['boxes_3d']
            scores_3d = det['scores_3d']
            labels_3d = det['labels_3d']
            track_ids = det.get('track_ids', None)

            if hasattr(boxes_3d, 'tensor'):
                tensor = boxes_3d.tensor
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                centers = boxes_3d.gravity_center.cpu().numpy() \
                    if boxes_3d.gravity_center.is_cuda \
                    else boxes_3d.gravity_center.numpy()
                dims = boxes_3d.dims.cpu().numpy() \
                    if boxes_3d.dims.is_cuda else boxes_3d.dims.numpy()
                yaws = boxes_3d.yaw.cpu().numpy() \
                    if boxes_3d.yaw.is_cuda else boxes_3d.yaw.numpy()
                vels = tensor[:, 7:9].numpy() if tensor.shape[-1] > 7 \
                    else np.zeros((len(boxes_3d), 2))
            else:
                tracks[scene][timestamp] = []
                continue

            scores = scores_3d.cpu().numpy() if hasattr(scores_3d, 'cpu') \
                else np.asarray(scores_3d)
            labels = labels_3d.cpu().numpy() if hasattr(labels_3d, 'cpu') \
                else np.asarray(labels_3d)
            if track_ids is not None:
                tids = track_ids.cpu().numpy() if hasattr(track_ids, 'cpu') \
                    else np.asarray(track_ids)
            else:
                tids = np.arange(len(boxes_3d))

            frame_boxes = []
            for i in range(len(boxes_3d)):
                label_idx = int(labels[i])
                if label_idx < 0 or label_idx >= len(self.CLASSES):
                    continue
                det_name = self.CLASSES[label_idx]
                if np.linalg.norm(centers[i, :2]) > track_cfg.class_range.get(
                        det_name, 50.0):
                    continue

                # Skip predictions outside the front camera's 60° FOV
                if not _in_fov(centers[i, :2]):
                    continue

                quat = Quaternion(axis=[0, 0, 1],
                                  radians=float(yaws[i]))
                frame_boxes.append(TrackingBox(
                    sample_token=info['token'],
                    translation=tuple(centers[i].tolist()),
                    size=tuple(dims[i].tolist()),
                    rotation=tuple(quat.elements.tolist()),
                    velocity=(float(vels[i, 0]), float(vels[i, 1])),
                    ego_translation=tuple(centers[i].tolist()),
                    num_pts=-1,
                    tracking_id=str(int(tids[i])),
                    tracking_name=det_name,
                    tracking_score=float(scores[i]),
                ))

            tracks[scene][timestamp] = frame_boxes

        return dict(tracks)

    def _run_tracking_eval(self, bbox_results):
        """Run nuScenes-style tracking evaluation (AMOTA, MOTP, etc.).

        Builds GT/pred tracks from data_infos and model outputs, then
        runs TrackingEvaluation per class using motmetrics.
        """
        print('\n' + '=' * 70)
        print('  LOKI Tracking Evaluation')
        print('=' * 70)

        track_cfg = self._get_tracking_config()
        gt_tracks = self._build_gt_tracks(track_cfg)
        pred_tracks = self._build_pred_tracks(bbox_results, track_cfg)

        # Ensure pred_tracks has all GT scenes (with empty frames if needed)
        for scene in gt_tracks:
            if scene not in pred_tracks:
                pred_tracks[scene] = {
                    ts: [] for ts in gt_tracks[scene]}
            else:
                for ts in gt_tracks[scene]:
                    if ts not in pred_tracks[scene]:
                        pred_tracks[scene][ts] = []

        # Per-class accumulation (mirrors TrackingEval.evaluate())
        print('Accumulating tracking metrics ...')
        metrics = TrackingMetrics(track_cfg)
        md_list = TrackingMetricDataList()

        for cls in track_cfg.class_names:
            ev = TrackingEvaluation(
                tracks_gt=gt_tracks,
                tracks_pred=pred_tracks,
                class_name=cls,
                dist_fcn=center_distance,
                dist_th_tp=track_cfg.dist_th_tp,
                min_recall=track_cfg.min_recall,
                num_thresholds=TrackingMetricData.nelem,
                metric_worst=track_cfg.metric_worst,
                verbose=False,
            )
            md = ev.accumulate()
            md_list.set(cls, md)

        # Aggregate per-class metrics (same logic as TrackingEval.evaluate)
        for cls in track_cfg.class_names:
            md = md_list[cls]
            if np.all(np.isnan(md.mota)):
                best_idx = None
            else:
                best_idx = int(np.nanargmax(md.mota))

            if best_idx is not None:
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == '':
                        continue
                    val = md.get_metric(metric_name)[best_idx]
                    metrics.add_label_metric(metric_name, cls, val)

            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(
                    md.get_metric(AVG_METRIC_MAP[metric_name]))
                if np.all(np.isnan(values)):
                    val = np.nan
                else:
                    values[np.isnan(values)] = \
                        track_cfg.metric_worst[metric_name]
                    val = float(np.nanmean(values))
                metrics.add_label_metric(metric_name, cls, val)

        return self._print_tracking_results(metrics, track_cfg)

    def _print_tracking_results(self, metrics, track_cfg):
        """Pretty-print tracking results and return a flat dict."""
        serialized = metrics.serialize()

        # Main summary metrics
        main_keys = ['amota', 'amotp', 'recall', 'mota', 'motp',
                      'ids', 'frag', 'tid', 'lgd']
        tab = PrettyTable()
        tab.field_names = ['Metric', 'Value']
        for k in main_keys:
            v = serialized.get(k, np.nan)
            tab.add_row([k.upper(), f'{v:.4f}' if not np.isnan(v) else 'N/A'])
        print('\nOverall Tracking Metrics:')
        print(tab)

        # Per-class AMOTA table
        cls_tab = PrettyTable()
        cls_tab.field_names = ['Class', 'AMOTA', 'AMOTP', 'RECALL',
                                'MOTA', 'IDS']
        for cls in track_cfg.class_names:
            row = [cls]
            for m in ['amota', 'amotp', 'recall', 'mota', 'ids']:
                v = metrics.label_metrics.get(m, {}).get(cls, np.nan)
                row.append(f'{v:.4f}' if not np.isnan(v) else 'N/A')
            cls_tab.add_row(row)
        print('\nPer-class Tracking:')
        print(cls_tab)
        print('=' * 70 + '\n')

        # Build flat dict
        detail = {}
        for k in main_keys:
            v = serialized.get(k, np.nan)
            if not np.isnan(v):
                detail[f'track/{k}'] = float(v)
        for cls in track_cfg.class_names:
            for m in ['amota', 'amotp']:
                v = metrics.label_metrics.get(m, {}).get(cls, np.nan)
                if not np.isnan(v):
                    detail[f'track/{cls}/{m}'] = float(v)
        return detail
