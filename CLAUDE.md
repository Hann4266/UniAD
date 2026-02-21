# UniAD + LOKI Dataset Integration

## What This Repo Is
UniAD (Unified Autonomous Driving) adapted to work with the LOKI dataset (single front-camera, no LiDAR point cloud). Originally designed for nuScenes (6 cameras).

## Coordinate Systems
- **Original LOKI pkl**: x=forward, y=lateral. `point_cloud_range = [0, -51.2, -5, 51.2, 51.2, 3]`
- **After rotation (current)**: 90° CCW applied in dataset loader to match nuScenes convention (x=right, y=forward). `point_cloud_range = [-51.2, 0, -5, 51.2, 51.2, 3]`
- Rotation applied in `loki_e2e_dataset.py` `get_data_info()` and `get_ann_info()` — NOT in the pkl itself
- `R_inv` (new→old): `[[0,1,0],[-1,0,0],[0,0,1]]` applied to `lidar2img`, `lidar2cam`, `l2g_r_mat`
- `can_bus` and `l2g_t` are in global frame — unchanged by rotation. The rotated `l2g_r_mat` handles the conversion in the transformer's BEV shift code.

## FOV Filtering (60° front camera)
LOKI has a single front camera with 60° horizontal FOV. GT annotations in the pkl include all 360° objects. Without filtering, objects behind/beside the ego are false negatives during training.

### How it works
- **Rotated frame**: +y = forward, +x = right. Object at (x, y) is in FOV if `y > 0` and `|atan2(x, y)| <= 30°`
- **Training pipeline**: `ObjectFOVFilterTrack` in `transform_3d.py` filters GT boxes after range/name filters
- **Evaluation**: `_in_fov()` helper in `loki_e2e_dataset.py` applied symmetrically to all 4 eval builders (GT detection, pred detection, GT tracking, pred tracking)
- **Impact**: ~57% of GT objects are outside the 60° FOV and get filtered out

### Files modified for FOV filter
- `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` — `ObjectFOVFilterTrack` class
- `projects/mmdet3d_plugin/datasets/pipelines/__init__.py` — export
- `projects/configs/loki/base_loki_perception.py` — added to train_pipeline after `ObjectNameFilterTrack`
- `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py` — `_in_fov()` + filtering in `_build_gt_eval_boxes`, `_build_pred_eval_boxes`, `_build_gt_tracks`, `_build_pred_tracks`

## Key Files

### Dataset
- `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py` — Main dataset class `LokiE2EDataset`. Mirrors `NuScenesE2EDataset` but single-camera. Contains `get_ann_info()`, `get_data_info()`, `prepare_train_data()`, `union2one()`, and full evaluation pipeline with FOV filtering.
- `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` — Pipeline transforms including `ObjectRangeFilterTrack`, `ObjectNameFilterTrack`, `ObjectFOVFilterTrack`
- `projects/mmdet3d_plugin/datasets/pipelines/loki_loading.py` — `LoadLokiImage` (single camera, scales lidar2img if resized from 1920x1208), `GenerateDummyOccLabels`
- `PKL_TO_CONFIG` dict maps lowercase pkl names → capitalized config names (e.g. `'car'→'Car'`)

### Config
- `projects/configs/loki/base_loki_perception.py` — Single config file. 8 classes, BEV 200x100 (w x h), ResNet101+DCN backbone. Train pipeline includes `ObjectFOVFilterTrack(fov_deg=60.0)`.
- `bev_h_=100, bev_w_=200` → grid_length = (0.512, 0.512)

### Model
- `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py` — Full UniAD model (track + motion + occ + planning). `forward_test()` is the inference entry point.
- `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` — Tracking head, `simple_test_track()` for inference.
- `projects/mmdet3d_plugin/uniad/detectors/bevformer.py` — BEVFormer base, `simple_test()` for perception-only.
- `projects/mmdet3d_plugin/uniad/modules/transformer.py` — `PerceptionTransformer`. BEV shift uses `can_bus` delta converted to lidar frame via `inv(l2g_r_mat) @ delta_global`.

### Tools
- `tools/train.py` / `tools/test.py` — Standard mmdet3d train/test. Test requires distributed launch (`torchrun`).
- `tools/create_loki_infos.py` — Generates pkl info files from raw LOKI data.
- `tools/visualize_predictions.py` — **Primary visualization tool.** Combined camera+BEV per frame. Camera: solid green=GT from `label2d_*.json` (pixel-accurate), dashed colored=pred projected from 3D. BEV: white=GT 3D boxes from pkl, dashed colored=pred 3D boxes with heading arrows. Uses original LOKI frame (x=forward, y=lateral). Requires `--data-root` for label2d JSONs.
- `tools/visualize_loki_gt.py` — Visualize GT with range + FOV filtering. Shows camera view (2D labels with KEPT/FILTERED status) and BEV in rotated frame with 60° FOV cone overlay. Applies 90° CCW rotation to boxes before filtering.

### Data
- `data/infos/loki_infos_train.pkl`, `data/infos/loki_infos_val.pkl` — Pre-computed info dicts
- Raw images at `/mnt/storage/loki_data/scenario_XXX/image_XXXX.png`
- Pkl info keys: `token, scene_token, frame_idx, img_filename, pts_filename, lidar2img, cam_intrinsic, lidar2cam, l2g_r_mat, l2g_t, can_bus, gt_boxes, gt_names, gt_labels, gt_inds, gt_velocity, valid_flag, num_lidar_pts, ego2global_rotation, ego2global_translation`
- `pts_filename`: path to PLY point cloud; `num_lidar_pts`: real count of LiDAR points per GT box; `valid_flag`: `num_lidar_pts > 0`
- `gt_boxes` shape: (N, 9) = [x, y, z, l, w, h, yaw, vx, vy]

### Training Outputs
- `work_dirs/base_loki_perception/` — Checkpoints (epoch_1-6.pth), results_epoch6.pkl, tensorboard logs
- Results pkl structure: `{'bbox_results': [{'token', 'boxes_3d', 'scores_3d', 'labels_3d', 'track_scores', 'track_ids', 'boxes_3d_det', 'scores_3d_det', 'labels_3d_det', 'track_bbox_results'}, ...]}`
- Results from epoch 6 are in the OLD coordinate system (before rotation was added)

## LiDAR Depth Supervision
LOKI has synchronized LiDAR point clouds (`pc_XXXX.ply`, ~232K pts/frame, PLY format with x,y,z,intensity).

### How it works
1. `create_loki_infos.py` stores `pts_filename` (path to PLY) and real `num_lidar_pts` per GT box (used for `valid_flag`)
2. `LoadLokiLiDARDepth` pipeline transform: loads PLY, projects to camera image using original lidar2cam + scaled intrinsics → sparse depth map `gt_depth` (H, W)
3. `DepthNet` module (`uniad/modules/depth_net.py`): Conv2d head on FPN level 0 features → categorical depth bins (60 bins, 1-61m)
4. Binary cross-entropy loss between predicted depth distribution and one-hot LiDAR depth, masked to valid pixels only

### Data flow
- Pipeline: `LoadLokiImage` → `LoadLokiLiDARDepth` → ... → `CustomCollect3D` (includes `gt_depth`)
- Dataset: `union2one` preserves `gt_depth` from last frame in queue
- Model: `UniAD.forward_train` passes `gt_depth` → `forward_track_train` → depth loss computed on cached FPN features after tracking loop

### Files modified
- `tools/create_loki_infos.py` — `pts_filename`, real `num_lidar_pts`, `valid_flag` from point counts
- `projects/mmdet3d_plugin/datasets/pipelines/loki_loading.py` — `LoadLokiLiDARDepth`
- `projects/mmdet3d_plugin/datasets/pipelines/__init__.py` — export
- `projects/mmdet3d_plugin/uniad/modules/depth_net.py` — `DepthNet` categorical depth head
- `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` — `depth_sup_cfg`, cached FPN feats, depth loss
- `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py` — passes `gt_depth` to track training
- `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py` — `pts_filename` in `get_data_info`, `gt_depth` in `union2one`
- `projects/configs/loki/base_loki_perception.py` — `depth_sup_cfg`, pipeline, collect keys

### Regenerating pkl (required before training)
```bash
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python tools/create_loki_infos.py \
    --data-root /mnt/storage/loki_data --out-dir data/infos
```

## LOKI vs nuScenes Differences
1. Single camera (not 6) — 60° horizontal FOV
2. Pre-computed transforms in pkl (l2g_r_mat as 3x3 matrix, not quaternion)
3. Zero-filled trajectories (no trajectory prediction data available)
4. Dummy occupancy labels
5. No NuScenes SDK dependency
6. 8 classes: Pedestrian, Car, Bus, Truck, Van, Motorcyclist, Bicyclist, Other
7. Dense scenes (~20+ agents per frame on average)
8. No depth sensor — 3D z values from GPS/IMU are noisy

## Performance Comparison
### LOKI (epoch 6, pre-FOV-filter, single front camera)
- mAP: 0.0952, NDS: 0.1628, AMOTA: 0.1007

### nuScenes UniAD reference (6 cameras)
- mAP: 0.368, AMOTA: 0.349 (UniAD(5) from paper)
- Front_BEV(3) single-cam variant: mAP: 0.33, AMOTA: 0.328

### Gap analysis
The ~4x mAP gap vs nuScenes is expected given: single camera (no multi-view), no LiDAR depth, FOV false negatives (now fixed), noisy z GT, dense LOKI scenes. The FOV filter should significantly improve metrics by removing ~57% of invisible GT objects.

## Train / Eval Commands
```bash
# Train (single GPU)
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -u tools/train.py \
    projects/configs/loki/base_loki_perception.py \
    --deterministic \
    --cfg-options log_config.interval=1

# Train (8 GPU)
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=29500 \
    tools/train.py projects/configs/loki/base_loki_perception.py \
    --launcher pytorch --deterministic --cfg-options log_config.interval=1

# Eval (distributed required)
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=28596 \
    tools/test.py projects/configs/loki/base_loki_perception.py \
    work_dirs/base_loki_perception/epoch_6.pth \
    --launcher pytorch --eval bbox \
    --out work_dirs/base_loki_perception/results.pkl

# Visualize predictions vs GT (camera + BEV)
python tools/visualize_predictions.py \
    --results    work_dirs/base_loki_perception/results_epoch6.pkl \
    --val-pkl    data/infos/loki_infos_val.pkl \
    --data-root  /mnt/storage/loki_data \
    --out-dir    vis_predictions/ \
    --num-frames 20 --score-thresh 0.4

# Visualize GT with FOV filter
python tools/visualize_loki_gt.py \
    --pkl data/infos/loki_infos_val.pkl \
    --data-root /mnt/storage/loki_data \
    --out-dir viz_loki_gt_fov \
    --num-samples 20 --fov-deg 60
```

## Evaluation

### Detection Evaluation
`LokiE2EDataset.evaluate()` in `loki_e2e_dataset.py` computes nuScenes-style detection metrics:
- Per-class AP at distance thresholds [0.5, 1.0, 2.0, 4.0]m
- TP errors: ATE (translation), ASE (scale), AOE (orientation), AVE (velocity)
- mAP, NDS (nuScenes Detection Score, attr_err set to worst since LOKI has no attributes)
- **FOV filter**: Both GT and predictions are filtered to 60° front-camera FOV before metric computation

Uses nuscenes-devkit algorithms (accumulate, calc_ap, calc_tp) with custom `LokiDetectionConfig` and `LokiDetectionBox` that bypass nuScenes class name assertions.

### Tracking Evaluation
Same `evaluate()` also runs tracking eval (AMOTA, MOTP, IDS, etc.):
- Builds GT/pred tracks grouped by (scene, timestamp)
- GT track IDs from `gt_inds` in pkl; pred track IDs from model `track_ids`
- Uses nuscenes `TrackingEvaluation` per class with motmetrics
- Creates `TrackingConfig` with LOKI names (sets global TRACKING_NAMES so TrackingBox accepts them)
- **FOV filter**: Both GT and pred tracks are filtered to 60° FOV

### Evaluation frame
Both GT and predictions are compared in the **rotated lidar frame** (the frame the model operates in). GT boxes from pkl are rotated 90° CCW (same transform as `get_ann_info`) before comparison. No global-frame transform is needed since all per-frame comparisons use the same coordinate system.

### Running eval standalone (without distributed test)
```python
import pickle
from projects.mmdet3d_plugin.datasets.loki_e2e_dataset import LokiE2EDataset
dataset = LokiE2EDataset(ann_file='data/infos/loki_infos_val.pkl', pipeline=None, test_mode=True)
results = pickle.load(open('work_dirs/base_loki_perception/results.pkl', 'rb'))
metrics = dataset.evaluate(results)
```

## Visualization Notes

### z-value accuracy (3D GT boxes)
LOKI has no LiDAR — 3D z values come from GPS/IMU and are noisy for vehicles (off by 0.9–1.9m). This makes projecting 3D GT boxes to camera view unreliable (boxes float above road).
- **Fix**: Camera GT uses raw `label2d_*.json` (pixel-accurate human annotations) instead of projecting 3D boxes.
- **Predictions**: The model learns to predict physically correct z values (closer to −1.5 to −2m for vehicles) even though GT training labels were noisy. Prediction 2D boxes project well.
- BEV GT uses the 3D pkl boxes directly — z error doesn't affect BEV (only x,y,yaw matter).

### Coordinate system for epoch-6 results
- Results in the **OLD system** (before 90° rotation was added to the dataset loader): x=forward, y=lateral.
- Val pkl `gt_boxes` are also in the original LOKI frame (rotation is applied by the loader at runtime, not in the pkl).
- The `lidar2img` in the pkl is for the original 1920×1208 image with assumed FOV=60°, no sensor height offset.
