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
            gt_names=gt_names_3d,
            gt_inds=gt_inds,  # Will be POPPED by LoadAnnotations3D_E2E (fixes BUG #2)
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
    #  Evaluation placeholder
    # ------------------------------------------------------------------ #

    def evaluate(self, results, **kwargs):
        """Simple evaluation placeholder."""
        # results from custom_multi_gpu_test is a dict with 'bbox_results' key
        if isinstance(results, dict):
            bbox_results = results.get('bbox_results', [])
        else:
            bbox_results = results

        print(f"[LOKI Eval] Received {len(bbox_results)} results")
        total_boxes = 0
        for det in bbox_results:
            if isinstance(det, dict) and 'boxes_3d' in det:
                total_boxes += len(det['boxes_3d'])
        print(f"[LOKI Eval] Total detected boxes: {total_boxes}")
        avg_boxes = total_boxes / max(len(bbox_results), 1)
        print(f"[LOKI Eval] Avg boxes per frame: {avg_boxes:.1f}")
        return {'num_detections': total_boxes, 'avg_boxes_per_frame': avg_boxes}
