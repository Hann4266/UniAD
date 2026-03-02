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
- `gt_labels_intent` now populated with real LOKI intent labels from pkl (see Intent Head section below).
- Resume vs load behavior:
  - Use `resume_from` only when optimizer parameter groups match checkpoint.
  - If model params changed (e.g., adding/removing trainable modules), resume can fail with `different number of parameter groups`; use `load_from` instead.

### Regenerating pkl (required before training)
```bash
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python tools/create_loki_infos.py \
    --data-root /mnt/storage/loki_data --out-dir data/infos
```

## Intent Head (Stage 2)

### Overview
Stage 2 training adds an intent prediction head on top of the frozen stage-1 perception model. The intent head predicts per-agent intent (7 classes) from tracked object queries using agent-agent, agent-BEV, and optionally agent-map interactions.

### Intent Classes (7)
| ID | Name | Vehicle Source | Pedestrian Source |
|----|------|---------------|-------------------|
| 0 | STOPPED | Stopped, Parked | Stopped, Waiting to cross |
| 1 | MOVING | Other moving | Moving |
| 2 | LCL | Lane change to the left, Cut in to the left | — |
| 3 | LCR | Lane change to the right, Cut in to the right | — |
| 4 | TL | Turn left | — |
| 5 | TR | Turn right | — |
| 6 | CROSSING | — | Crossing the road |

LOKI class IDs for intent masking: `vehicle_id_list=[1,2,3,4,5,6]` (Car,Bus,Truck,Van,Motorcyclist,Bicyclist), `ped_id_list=[0]` (Pedestrian), `ignore_id_list=[7]` (Other). Vehicles can't CROSS, pedestrians can't LCL/LCR/TL/TR.

### GT Intent Labels in Pkl
- `create_loki_infos.py` extracts `vehicle_state` (for vehicles) and `intended_actions` (for pedestrians) from `label3d_*.txt` CSV columns 11 and 12
- Maps strings → integer class IDs via `VEHICLE_STATE_TO_INTENT` / `PED_ACTION_TO_INTENT` dicts and `loki_intent_label()` function
- Stored as `gt_intent_labels` (int64 array, parallel to `gt_boxes`) in each frame's info dict
- All valid LOKI agents have labeled intent (0% unknown) — `None` values only appear on filtered-out classes like `Road_Entrance_Exit`
- Label distribution (train): MOVING 50%, STOP 40%, CROSS 6%, TL 1.8%, TR 1.6%, LCL 0.3%, LCR 0.3%

### Architecture
- `IntentTransformerDecoder` (3 layers): agent-agent interaction (TransformerDecoderLayer) + agent-BEV deformable attention (`IntentDeformableAttention`) + optional agent-map interaction
- LOKI uses `map_features=False, inter_features=True` (default) → fuser input is 3×D (no map), vs 4×D with map. The main branch `IntentTransformerDecoder` already supports this natively via conditional branches — no code changes needed in modules.py.
- `obj_type_embed`: Embedding(3, D) adds learned type embeddings (0=ped, 1=veh, 2=ignore) to track queries before the decoder — same as main branch.
- Per-layer classification branches: `Linear(D,D) → LN → ReLU → Linear(D,D) → LN → ReLU → Linear(D,7)`
- Loss: masked softmax focal loss with sqrt-inverse-frequency class weights `[1.18, 1.0, 15.49, 12.77, 5.38, 6.32, 2.24]` (computed from LOKI train distribution). `ped_loss_weight=1.0` (LOKI has proportionally more pedestrians than nuScenes).

### Track-to-Intent Mapping
Hungarian matching in the tracking head produces `track_query_matched_idxes[query_i] = gt_j`. GT intent labels are parallel arrays to GT bboxes: `gt_labels_intent[gt_j]` gives intent for GT object j. The loss function uses this mapping to assign GT intent labels to predicted track queries. SDC (ego) is appended with `match_index=-1` (unmatched).

### Stage 2 Freezing
`freeze_img_backbone=True, freeze_img_neck=True, freeze_bn=True, freeze_bev_encoder=True` — only the intent head parameters train.

### Files Modified for Intent Head (LOKI branch)
Principle: **minimal diff from main branch**. The intent head architecture and loss are identical to main; only dataset-specific adaptations are changed.

- `tools/create_loki_infos.py` — Added `VEHICLE_STATE_TO_INTENT`, `PED_ACTION_TO_INTENT`, `loki_intent_label()`, `gt_intent_labels` in pkl output
- `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py` — `get_ann_info()` reads `gt_intent_labels` from pkl (falls back to -1 if missing); `union2one()` propagates `gt_labels_intent` across temporal queue
- `projects/mmdet3d_plugin/uniad/dense_heads/intent_head.py` — **2 minimal changes from main**: (1) `forward_test` bug fix (`intent_scores` → `logits_last` variable name, fix `[bi]` indexing on already-sliced tensors), (2) LOKI intent ID constants in `_build_allowed_mask_and_ignore` (LOKI: 0=STOP,1=MOVING,2=LCL,3=LCR,4=TL,5=TR,6=CROSS vs main: 2=CROSS,3=TR,4=TL,5=LCR,6=LCL)
- `projects/mmdet3d_plugin/uniad/dense_heads/intent_head_plugin/modules.py` — **No changes from main**. Main already handles `map_features=False` via `inter_features` conditional branches.
- `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py` — **1 line added**: `result_seg=[{}]` before `if self.with_seg_head` in `forward_test` (prevents `NameError` when no seg_head but intent_head is present)
- `projects/configs/loki/loki_stage2_intent.py` — New config for LOKI intent training

### Config
- `projects/configs/loki/loki_stage2_intent.py`
- `load_from`: stage-1 checkpoint path (currently `/mnt/storage/UniAD/work_dirs/base_loki_perception/epoch_10.pth`)
- `data_root`: `/root/loki_data/`
- Intent head: `num_intent=7, map_features=False, num_layers=3, det_layer_num=1`

### Commands
```bash
# Regenerate pkl with intent labels (required once after code change)
cd /root/UniAD && python tools/create_loki_infos.py \
    --data-root /root/loki_data --out-dir data/infos

# Stage 2 training (8 GPU)
cd /root/UniAD && PYTHONPATH="$(pwd)/projects:$(pwd):$PYTHONPATH" \
python3 -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=28596 \
    tools/train.py projects/configs/loki/loki_stage2_intent.py \
    --launcher pytorch --deterministic \
    --work-dir /mnt/storage/UniAD/work_dirs/stage_2_base
```

### Data Flow
1. `create_loki_infos.py` → pkl with `gt_intent_labels` per frame
2. `get_ann_info()` reads `gt_intent_labels` from pkl, applies validity mask
3. `LoadAnnotations3D_E2E` with `with_intent_label_3d=True` passes through
4. Pipeline filters (`ObjectRangeFilterTrack`, `ObjectNameFilterTrack`, `ObjectFOVFilterTrack`, `ObjectCameraVisibleFilter`) all propagate `gt_labels_intent` in sync with `gt_bboxes_3d`
5. `union2one()` collects temporal queue, stores as `DC(list[tensor])`
6. `intent_head.loss()` calls `_last_frame_gt_intent_labels()` to extract last frame's labels from queue
7. Hungarian matching indices map predicted tracks → GT objects → GT intent labels
8. Masked softmax focal loss (vehicles can't CROSS, peds can't lane change)

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
- Pkl info keys: `token, scene_token, frame_idx, img_filename, lidar2img, cam_intrinsic, lidar2cam, l2g_r_mat, l2g_t, can_bus, gt_boxes, gt_names, gt_labels, gt_inds, gt_velocity, gt_intent_labels, valid_flag, num_lidar_pts, ego2global_rotation, ego2global_translation, gt_camera_visible`
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

## Fine-tuning nuScenes→LOKI

### Overview
Fine-tune the nuScenes-pretrained UniAD checkpoint on LOKI data. The nuScenes model (10 classes, seg_head for map) is adapted to LOKI's 8-class, single-camera, no-map setting. Only BEVFormer encoder + detection/tracking decoder are trained (same modules as from-scratch LOKI training). Map head is removed entirely.

### Class mapping (nuScenes→LOKI weight adaptation)

| LOKI idx | LOKI class   | nuScenes idx | nuScenes source |
|----------|-------------|-------------|-----------------|
| 0        | Pedestrian  | 8           | pedestrian      |
| 1        | Car         | 0           | car             |
| 2        | Bus         | 3           | bus             |
| 3        | Truck       | 1           | truck           |
| 4        | Van         | 0           | car (Van mapped to Car in pkl) |
| 5        | Motorcyclist| 6           | motorcycle      |
| 6        | Bicyclist   | 7           | bicycle         |
| 7        | Other       | —           | random init     |

Dropped nuScenes classes: construction_vehicle (2), trailer (4), barrier (5), traffic_cone (9).

### Weight adaptation
`tools/adapt_nuscenes_to_loki.py` converts the nuScenes checkpoint:
- Remaps `pts_bbox_head.cls_branches.{0-5}.6.{weight,bias}` from [10,...] → [8,...] using class mapping
- Drops all 578 `seg_head.*` keys (map module, no LOKI map GT)
- Drops optimizer state (param groups changed)
- All other weights (backbone, neck, BEV encoder, decoder, regression heads, query embeddings, memory bank) transfer unchanged

### Config
`projects/configs/loki/loki_finetune_from_nuscenes.py` — Identical architecture to `base_loki_perception.py` (8 classes, no seg_head, single camera) but:
- `load_from` points to the adapted checkpoint
- Lower learning rate (5e-5 vs 2e-4) for fine-tuning stability
- Same GT filters: range, 60° FOV, camera visibility (2D bbox check)

### Trainable modules (same as from-scratch)
- `img_neck` (FPN) — trainable
- `pts_bbox_head.transformer.encoder` (BEVFormer encoder) — trainable
- `pts_bbox_head.transformer.decoder` (detection decoder) — trainable
- `pts_bbox_head.cls_branches` / `reg_branches` — trainable
- `query_embedding`, `reference_points`, `query_interact`, `memory_bank` — trainable
- `img_backbone` (ResNet101) — **frozen** (frozen_stages=4)
- No seg_head / motion_head / occ_head / planning_head

### Commands
```bash
# Step 1: Adapt checkpoint (10 classes → 8 classes, drop seg_head)
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH \
python tools/adapt_nuscenes_to_loki.py \
    --src /mnt/storage/UniAD/work_dirs/zihan_nuscenes_weight/epoch_6.pth \
    --dst work_dirs/nuscenes_adapted_for_loki.pth

# Step 2: Fine-tune (single GPU)
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -u tools/train.py \
    projects/configs/loki/loki_finetune_from_nuscenes.py \
    --deterministic \
    --cfg-options log_config.interval=1

# Step 2 alt: Fine-tune (8 GPU)
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=29500 \
    tools/train.py projects/configs/loki/loki_finetune_from_nuscenes.py \
    --launcher pytorch --deterministic --cfg-options log_config.interval=1

# Step 3: Evaluate
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=28596 \
    tools/test.py projects/configs/loki/loki_finetune_from_nuscenes.py \
    work_dirs/loki_finetune_from_nuscenes/epoch_6.pth \
    --launcher pytorch --eval bbox \
    --out work_dirs/loki_finetune_from_nuscenes/results.pkl
```
