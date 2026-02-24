"""
LOKI-specific data loading and annotation pipelines for UniAD.

Handles single-camera image loading (vs. 6-camera for nuScenes)
and LOKI-specific annotation format conversion.
"""

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class LoadLokiImage(object):
    """Load a single-view image from the LOKI dataset.

    Expects results['img_filename'] to be a list containing one filename.
    Optionally resizes the image to a target size.

    The output follows the same multi-view convention as nuScenes
    (results['img'] is a list of images) but with length 1.

    Args:
        to_float32 (bool): Convert the image to float32. Default: True.
        color_type (str): Color type for loading. Default: 'unchanged'.
        target_size (tuple): (W, H) to resize to. Default: (1600, 900).
    """

    def __init__(self, to_float32=True, color_type='unchanged',
                 target_size=(1600, 900)):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.target_size = target_size  # (W, H)

    def __call__(self, results):
        filename = results['img_filename']
        assert isinstance(filename, (list, tuple)), \
            f"img_filename should be a list, got {type(filename)}"

        images = []
        for img_path in filename:
            img = mmcv.imread(img_path, self.color_type)
            if img is None:
                import warnings
                warnings.warn(f"Skipping corrupted/missing image: {img_path}")
                return None
            # Resize to target
            if self.target_size is not None:
                img = mmcv.imresize(img, self.target_size, return_scale=False)
            if self.to_float32:
                img = img.astype(np.float32)
            images.append(img)

        results['filename'] = filename
        results['img'] = images  # list of 1 image, shape (H, W, 3)
        results['img_shape'] = [img.shape for img in images]
        results['ori_shape'] = [img.shape for img in images]
        results['pad_shape'] = [img.shape for img in images]
        results['scale_factor'] = 1.0

        num_channels = 1 if len(images[0].shape) < 3 else images[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        # Scale lidar2img if we resized
        if self.target_size is not None:
            # Original LOKI resolution: 1920×1208
            orig_w, orig_h = 1920, 1208
            target_w, target_h = self.target_size
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            scale_matrix = np.eye(4, dtype=np.float64)
            scale_matrix[0, 0] = scale_x
            scale_matrix[1, 1] = scale_y
            if 'lidar2img' in results:
                results['lidar2img'] = [
                    scale_matrix @ l2i for l2i in results['lidar2img']]
            if 'cam_intrinsic' in results:
                results['cam_intrinsic'] = [
                    scale_matrix @ ci for ci in results['cam_intrinsic']]

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f'target_size={self.target_size})')


@PIPELINES.register_module()
class LoadLokiAnnotations3D(object):
    """Load LOKI 3D annotations from the pre-computed ann_info dict.

    Handles:
        - 3D bounding boxes (gt_bboxes_3d)
        - Labels (gt_labels_3d)
        - Instance indices (gt_inds)
        - Trajectory placeholders (gt_fut_traj, gt_past_traj, etc.)
        - SDC (ego vehicle) pseudo-annotations

    Args:
        with_bbox_3d (bool): Load 3D bounding boxes. Default: True.
        with_label_3d (bool): Load 3D labels. Default: True.
        with_ins_inds_3d (bool): Load instance indices. Default: True.
        ins_inds_add_1 (bool): Offset instance indices by +1. Default: True.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_ins_inds_3d=True,
                 ins_inds_add_1=True):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_ins_inds_3d = with_ins_inds_3d
        self.ins_inds_add_1 = ins_inds_add_1

    def __call__(self, results):
        ann_info = results.get('ann_info', {})

        if self.with_bbox_3d and 'gt_bboxes_3d' in ann_info:
            results['gt_bboxes_3d'] = ann_info['gt_bboxes_3d']
            results['bbox3d_fields'].append('gt_bboxes_3d')

        if self.with_label_3d and 'gt_labels_3d' in ann_info:
            results['gt_labels_3d'] = ann_info['gt_labels_3d']

        if self.with_ins_inds_3d and 'gt_inds' in ann_info:
            gt_inds = ann_info['gt_inds'].copy()
            if self.ins_inds_add_1:
                gt_inds = gt_inds + 1
            results['gt_inds'] = gt_inds

        # Trajectory placeholders
        for key in ['gt_labels_intent',
                     'gt_fut_traj', 'gt_fut_traj_mask',
                     'gt_past_traj', 'gt_past_traj_mask',
                     'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask']:
            if key in ann_info:
                results[key] = ann_info[key]

        # SDC bbox/label
        if 'gt_sdc_bbox' in ann_info:
            results['gt_sdc_bbox'] = ann_info['gt_sdc_bbox']
        if 'gt_sdc_label' in ann_info:
            results['gt_sdc_label'] = ann_info['gt_sdc_label']

        # Dummy occ future annotations (required by pipeline but unused)
        if 'occ_future_ann_infos' in results:
            results['future_gt_bboxes_3d'] = [None] * len(results['occ_future_ann_infos'])
            results['future_gt_labels_3d'] = [None] * len(results['occ_future_ann_infos'])
            results['future_gt_inds'] = [None] * len(results['occ_future_ann_infos'])
            results['future_gt_vis_tokens'] = [None] * len(results['occ_future_ann_infos'])

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'with_bbox_3d={self.with_bbox_3d}, '
                f'with_label_3d={self.with_label_3d}, '
                f'with_ins_inds_3d={self.with_ins_inds_3d}, '
                f'ins_inds_add_1={self.ins_inds_add_1})')


@PIPELINES.register_module()
class LokiFormatBundle3D(object):
    """Format LOKI data into DataContainers for the model.

    Similar to DefaultFormatBundle3D but without map-specific formatting.
    Handles single-camera image stacking.
    """

    def __init__(self, class_names, with_label=True):
        self.class_names = class_names
        self.with_label = with_label

    def __call__(self, results):
        # Format images: list of HWC → stacked CHW tensor
        if 'img' in results:
            imgs = results['img']
            # imgs is a list of (H, W, C) arrays
            imgs = [img.transpose(2, 0, 1) for img in imgs]
            imgs = np.stack(imgs, axis=0)  # (1, C, H, W)
            results['img'] = DC(
                to_tensor(imgs), stack=True)

        # Format 3D bboxes
        for key in ['gt_bboxes_3d', 'gt_sdc_bbox']:
            if key in results:
                if not isinstance(results[key], DC):
                    results[key] = DC(results[key], cpu_only=True)

        # Format labels
        for key in ['gt_labels_3d']:
            if key in results and self.with_label:
                if not isinstance(results[key], DC):
                    results[key] = DC(to_tensor(results[key]))

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(class_names={self.class_names})'


@PIPELINES.register_module()
class GenerateDummyOccLabels(object):
    """Generate dummy occupancy flow labels for LOKI.

    Since we don't use the occupancy head with LOKI, this generates
    correctly-shaped zero tensors that satisfy the pipeline and model
    interface requirements.

    Args:
        bev_h (int): BEV height. Default: 200.
        bev_w (int): BEV width. Default: 200.
        n_future (int): Number of future frames. Default: 4.
    """

    def __init__(self, bev_h=200, bev_w=200, n_future=4):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.n_future = n_future
        self.n_frames = n_future + 1  # current + future

    def __call__(self, results):
        import torch

        results['gt_occ_has_invalid_frame'] = results.pop(
            'occ_has_invalid_frame', True)
        results['gt_occ_img_is_valid'] = results.pop(
            'occ_img_is_valid', np.array([True] * 7))

        results['gt_segmentation'] = torch.zeros(
            self.n_frames, self.bev_h, self.bev_w, dtype=torch.long)
        results['gt_instance'] = torch.zeros(
            self.n_frames, self.bev_h, self.bev_w, dtype=torch.long)
        results['gt_centerness'] = torch.zeros(
            self.n_frames, 1, self.bev_h, self.bev_w, dtype=torch.float32)
        results['gt_offset'] = 255.0 * torch.ones(
            self.n_frames, 2, self.bev_h, self.bev_w, dtype=torch.float32)
        results['gt_flow'] = 255.0 * torch.ones(
            self.n_frames, 2, self.bev_h, self.bev_w, dtype=torch.float32)
        results['gt_backward_flow'] = 255.0 * torch.ones(
            self.n_frames, 2, self.bev_h, self.bev_w, dtype=torch.float32)

        # Dummy future boxes and labels
        results['gt_future_boxes'] = []
        results['gt_future_labels'] = []

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'bev_h={self.bev_h}, bev_w={self.bev_w}, '
                f'n_future={self.n_future})')


@PIPELINES.register_module()
class PadToMultiCamera(object):
    """Pad single-camera Loki data to multi-camera format for nuScenes-trained models.

    The nuScenes BEVFormer model expects 6 camera views. This transform pads
    a single Loki front camera to 6 slots by filling the remaining 5 with
    black images and identity/zero transform matrices.

    BEV queries in the front region primarily attend to the front camera via
    deformable attention. The 5 dummy cameras produce near-zero backbone
    features and contribute negligibly to the BEV.

    Args:
        target_num_cams (int): Number of camera slots. Default: 6.
    """

    def __init__(self, target_num_cams=6):
        self.target_num_cams = target_num_cams

    def __call__(self, results):
        num_existing = len(results['img'])
        num_pad = self.target_num_cams - num_existing

        if num_pad <= 0:
            return results

        # Pad images with black images matching front camera shape
        front_img = results['img'][0]
        h, w = front_img.shape[:2]
        c = front_img.shape[2] if len(front_img.shape) == 3 else 1
        black = np.zeros_like(front_img)
        results['img'] = results['img'] + [black.copy() for _ in range(num_pad)]

        # Pad img_shape, ori_shape, pad_shape
        for key in ['img_shape', 'ori_shape', 'pad_shape']:
            if key in results and isinstance(results[key], list):
                results[key] = results[key] + [results[key][0]] * num_pad

        # Pad transform matrices with identity 4x4 for dummy cameras
        identity_4x4 = np.eye(4, dtype=np.float64)
        for key in ['lidar2img', 'lidar2cam', 'cam_intrinsic']:
            if key in results and isinstance(results[key], list):
                results[key] = results[key] + [identity_4x4.copy() for _ in range(num_pad)]

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(target_num_cams={self.target_num_cams})'
