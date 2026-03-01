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

## Camera Visibility Filter (occlusion-aware)
The FOV filter only checks angular position — agents within the 60° cone but fully occluded by other objects are still kept as GT, creating false negatives. The camera visibility filter cross-references `label3d` (3D boxes, 360°) with `label2d` (2D boxes, only camera-visible agents) using shared UUID track IDs to remove 3D GT objects that have no corresponding 2D annotation.

### How it works
1. `create_loki_infos.py` loads `label2d_*.json` per frame, collects all UUIDs, and stores `gt_camera_visible` (bool array) in the pkl for each frame
2. `ObjectCameraVisibleFilter` pipeline transform filters `gt_bboxes_3d` and all parallel arrays by this flag
3. Runs after `ObjectFOVFilterTrack` in the training pipeline — the FOV filter removes behind/beside agents, the visibility filter removes occluded agents within the FOV
4. All upstream filters (`ObjectRangeFilterTrack`, `ObjectNameFilterTrack`, `ObjectFOVFilterTrack`) propagate `gt_camera_visible` in sync

### Impact
~46% of 3D agents across all scenes have no 2D bbox (checked 100 scenarios, 6967 frames). After FOV filtering removes ~57%, the visibility filter further removes occluded-from-camera agents within the FOV cone.

### Files modified
- `tools/create_loki_infos.py` — loads `label2d_*.json`, stores `gt_camera_visible` in pkl
- `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` — `ObjectCameraVisibleFilter` class + `gt_camera_visible` propagation in all 3 existing filters
- `projects/mmdet3d_plugin/datasets/pipelines/__init__.py` — export
- `projects/configs/loki/base_loki_perception.py` — added after `ObjectFOVFilterTrack`
- `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py` — `gt_camera_visible` in `get_ann_info()` and `get_data_info()`

### Compatibility notes (training stability)
- `gt_labels_intent` KeyError fix:
  - Cause: `ObjectRangeFilterTrack` expects `gt_labels_intent`, but LOKI has no intent labels.
  - Fix: `LokiE2EDataset.get_ann_info()` now always emits dummy intent labels (`zeros`), and annotation loaders pass through `gt_labels_intent` when present.
  - Files: `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py`, `projects/mmdet3d_plugin/datasets/pipelines/loading.py`, `projects/mmdet3d_plugin/datasets/pipelines/loki_loading.py`.
- Resume vs load behavior:
  - Use `resume_from` only when optimizer parameter groups match checkpoint.
  - If model params changed (e.g., adding/removing trainable modules), resume can fail with `different number of parameter groups`; use `load_from` instead.

### Regenerating pkl (required before training)
```bash
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python tools/create_loki_infos.py \
    --data-root /mnt/storage/loki_data --out-dir data/infos
```

## Key Files

### Dataset
- `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py` — Main dataset class `LokiE2EDataset`. Mirrors `NuScenesE2EDataset` but single-camera. Contains `get_ann_info()`, `get_data_info()`, `prepare_train_data()`, `union2one()`, and full evaluation pipeline with FOV filtering.
- `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` — Pipeline transforms including `ObjectRangeFilterTrack`, `ObjectNameFilterTrack`, `ObjectFOVFilterTrack`, `ObjectCameraVisibleFilter`
- `projects/mmdet3d_plugin/datasets/pipelines/loki_loading.py` — `LoadLokiImage` (single camera, scales lidar2img if resized from 1920x1208), `GenerateDummyOccLabels`
- `PKL_TO_CONFIG` dict maps lowercase pkl names → capitalized config names (e.g. `'car'→'Car'`)

### Config
- `projects/configs/loki/base_loki_perception.py` — Single config file. 8 classes, BEV 200x100 (w x h), ResNet101+DCN backbone. Train pipeline includes `ObjectFOVFilterTrack(fov_deg=60.0)` and `ObjectCameraVisibleFilter`.
- `bev_h_=100, bev_w_=200` → grid_length = (0.512, 0.512)

### Model
- `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py` — Full UniAD model (track + motion + occ + planning). `forward_test()` is the inference entry point.
- `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` — Tracking head, `simple_test_track()` for inference.
- `projects/mmdet3d_plugin/uniad/detectors/bevformer.py` — BEVFormer base, `simple_test()` for perception-only.
- `projects/mmdet3d_plugin/uniad/modules/transformer.py` — `PerceptionTransformer`. BEV shift uses `can_bus` delta converted to lidar frame via `inv(l2g_r_mat) @ delta_global`.

### Tools
- `tools/train.py` / `tools/test.py` — Standard mmdet3d train/test. Test requires distributed launch (`torchrun`).
- `tools/create_loki_infos.py` — Generates pkl info files from raw LOKI data.
- `tools/visualize_predictions.py` — **Primary visualization tool.** Combined camera+BEV per frame. Camera: solid green=GT from `label2d_*.json` (pixel-accurate, inherently FOV+visibility filtered), dashed colored=pred projected from 3D. BEV: white=GT 3D boxes filtered by PC range + 60° FOV cone + `gt_camera_visible` (matches eval filtering), dashed colored=pred 3D boxes with heading arrows. Uses original LOKI frame (x=forward, y=lateral). Requires `--data-root` for label2d JSONs.
- `tools/visualize_loki_gt.py` — Visualize GT with range + FOV + camera-visibility filtering. Camera view can show projected yellow boxes for 3D agents removed by no-2D visibility filter; BEV shows final-kept / FOV-filtered / range-filtered / no2D-filtered in rotated frame.

### Data
- `data/infos/loki_infos_train.pkl`, `data/infos/loki_infos_val.pkl` — Pre-computed info dicts
- Raw images at `/mnt/storage/loki_data/scenario_XXX/image_XXXX.png`
- Pkl info keys: `token, scene_token, frame_idx, img_filename, lidar2img, cam_intrinsic, lidar2cam, l2g_r_mat, l2g_t, can_bus, gt_boxes, gt_names, gt_labels, gt_inds, gt_velocity, valid_flag, num_lidar_pts, ego2global_rotation, ego2global_translation, gt_camera_visible`
- `gt_boxes` shape: (N, 9) = [x, y, z, l, w, h, yaw, vx, vy]

### Training Outputs
- `work_dirs/base_loki_perception/` — Checkpoints (epoch_1-6.pth), results_epoch6.pkl, tensorboard logs
- Results pkl structure: `{'bbox_results': [{'token', 'boxes_3d', 'scores_3d', 'labels_3d', 'track_scores', 'track_ids', 'boxes_3d_det', 'scores_3d_det', 'labels_3d_det', 'track_bbox_results'}, ...]}`
- Results from epoch 6 are in the OLD coordinate system (before rotation was added)

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
- mAP: 0.2284, NDS: 0.2472, AMOTA: 0.2619

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

## Zero-Shot nuScenes→LOKI Inference

### Overview
A nuScenes-trained UniAD checkpoint (`work_dirs/zihan_nuscenes_weight/epoch_6.pth`) is run zero-shot on LOKI data. The nuScenes model has 10 classes, 6-camera input, and a `seg_head` (PansegformerHead) for map segmentation. LOKI's single camera is padded to 6 via `PadToMultiCamera` in the test pipeline.

### Config
- `projects/configs/loki/loki_nuscenes_zeroshot.py` — Uses nuScenes architecture (10 classes, seg_head for map) with LOKI data. `PadToMultiCamera` pads single camera to 6. Point cloud range: `[-51.2, 0, -5, 51.2, 51.2, 3]` (rotated frame).

### Class mapping (nuScenes → LOKI)
The nuScenes model outputs labels 0-9 in nuScenes class order, which differs from LOKI's 8-class order:

| Label | nuScenes class         | LOKI class     |
|-------|------------------------|----------------|
| 0     | car                    | Car            |
| 1     | truck                  | Truck          |
| 2     | construction_vehicle   | (no equivalent)|
| 3     | bus                    | Bus            |
| 4     | trailer                | (no equivalent)|
| 5     | barrier                | (no equivalent)|
| 6     | motorcycle             | Motorcyclist   |
| 7     | bicycle                | Bicyclist      |
| 8     | pedestrian             | Pedestrian     |
| 9     | traffic_cone           | (no equivalent)|

### Coordinate system for zeroshot results
Predictions are in the **rotated frame** (x=right, y=forward) — same as the model's operating frame. To visualize alongside GT in the original LOKI frame (x=forward, y=lateral), apply `R_inv`:
- `x_old = y_new`, `y_old = -x_new`, `yaw_old = yaw_new - π/2`
- `vx_old = vy_new`, `vy_old = -vx_new`

### Results files
- `work_dirs/nuscenes_on_loki/results_zeroshot.pkl` — Full val set (4236 frames, 64 scenes), detection only (no `pts_bbox`)
- `work_dirs/nuscenes_on_loki/results_zeroshot_with_map_20.pkl` — 20 frames from scenario_004 only, includes `pts_bbox` (map predictions)
- `work_dirs/nuscenes_on_loki/results_zeroshot_with_map_64scenes.pkl` — 64 frames (1 per scene), includes `pts_bbox`

### Map predictions (pts_bbox)
When `seg_head` is present in the config, results include `pts_bbox` dict per frame:
- `drivable`: `(100, 200)` bool — drivable area BEV mask
- `lane`: `(3, 100, 200)` int64 — lane line masks (divider, crossing, contour)
- `lane_score`: `(3, 100, 200)` float32 — lane confidence scores
- `segm`: `(100, 100, 200)` bool — instance segmentation masks
- `bbox`: `(100, 5)` float32 — bounding boxes
- `panoptic`: tuple — panoptic segmentation

Map stats across 64 scenes: drivable area ~44.5% coverage, lane lines ~3% coverage. All 64 scenes produce non-zero predictions.

### Running zeroshot inference
```bash
# Full val set (4236 frames, ~70 min on 1 GPU, no map output in existing run)
cd /mnt/storage/UniAD && \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29601 \
tools/test.py \
projects/configs/loki/loki_nuscenes_zeroshot.py \
work_dirs/zihan_nuscenes_weight/epoch_6.pth \
--launcher pytorch \
--out work_dirs/nuscenes_on_loki/results_zeroshot.pkl \
--tmpdir /tmp/uniad_zeroshot_collect \
--cfg-options data.workers_per_gpu=1

# Subset (1 frame per scene, 64 frames, ~1 min) — uses trimmed val pkl
cd /mnt/storage/UniAD && \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29602 \
tools/test.py \
projects/configs/loki/loki_nuscenes_zeroshot.py \
work_dirs/zihan_nuscenes_weight/epoch_6.pth \
--launcher pytorch \
--out work_dirs/nuscenes_on_loki/results_zeroshot_with_map_64scenes.pkl \
--tmpdir /tmp/uniad_zeroshot_64s \
--cfg-options data.test.ann_file=data/infos/loki_infos_val_1perscene.pkl data.workers_per_gpu=1
```

### Visualization tools

#### Detection visualization (`tools/visualize_predictions.py`)
Camera+BEV combined figure. For nuScenes zeroshot results, use `--rotated-preds` flag which:
1. Applies `rotate_boxes_to_original()` to convert predicted boxes from rotated→original LOKI frame
2. Switches class names from LOKI 8-class to nuScenes 10-class list (`CLASS_NAMES_NUSCENES`)

**BEV GT filtering** (matches eval pipeline):
- PC range: `x ∈ [0, 51.2]m, y ∈ [−51.2, 51.2]m`
- 60° FOV cone: `x > 0 and |atan2(y, x)| ≤ 30°` (original LOKI frame, via `_in_fov_original()`)
- Camera visibility: `gt_camera_visible[i]` from pkl (skipped gracefully if field absent)

**Token mismatch note (epoch-6 results):** `results_epoch6.pkl` was produced before the current val split — only 261/2482 tokens overlap with `loki_infos_val.pkl`. Use `--tokens` with the overlapping subset:
```bash
python3 -c "
import pickle
r = pickle.load(open('work_dirs/base_loki_perception/results_epoch6.pkl','rb'))
v = pickle.load(open('data/infos/loki_infos_val.pkl','rb'))
vt = {i['token'] for i in v['infos']}
overlap = [p['token'] for p in r['bbox_results'] if p['token'] in vt]
print(' '.join(overlap[:20]))
"
```

```bash
# LOKI-trained model
python tools/visualize_predictions.py \
    --results work_dirs/base_loki_perception/results_epoch6.pkl \
    --val-pkl data/infos/loki_infos_val.pkl \
    --data-root /mnt/storage/loki_data \
    --out-dir vis_predictions/ \
    --score-thresh 0.4 \
    --tokens <overlapping tokens from snippet above>

# Zero-shot nuScenes model
python tools/visualize_predictions.py \
    --results work_dirs/nuscenes_on_loki/results_zeroshot.pkl \
    --val-pkl data/infos/loki_infos_val.pkl \
    --data-root /mnt/storage/loki_data \
    --out-dir work_dirs/nuscenes_on_loki/vis_det_zeroshot_multiscene \
    --score-thresh 0.3 --rotated-preds \
    --num-frames 20
```

#### Map visualization (`tools/visualize_map_predictions.py`)
Camera image on top, 4 separate BEV panels below (drivable, divider, crossing, contour) — avoids overlap. Each panel has a yellow dashed ego midline at x=0 (BEV column 100). Also saves scaled BEV-only images to `bev_only/` subdirectory.

```bash
python tools/visualize_map_predictions.py \
    --results work_dirs/nuscenes_on_loki/results_zeroshot_with_map_64scenes.pkl \
    --loki-infos data/infos/loki_infos_val_1perscene.pkl \
    --data-root /mnt/storage/loki_data/ \
    --out-dir work_dirs/nuscenes_on_loki/vis_map_zeroshot_64scenes \
    --num-samples 64
```

### Subset val pkl
`data/infos/loki_infos_val_1perscene.pkl` — 64 frames (frame 5 from each of the 64 val scenes). Created by grouping val infos by `scene_token` and picking 1 frame per scene. Used for fast inference with map output.
