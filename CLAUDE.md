# UniAD + LOKI Dataset Integration

## What This Repo Is
UniAD (Unified Autonomous Driving) adapted to work with the LOKI dataset (single front-camera, no LiDAR point cloud). Originally designed for nuScenes (6 cameras).

## Coordinate Systems
- **Original LOKI pkl**: x=forward, y=lateral. `point_cloud_range = [0, -51.2, -5, 51.2, 51.2, 3]`
- **After rotation (current)**: 90° CCW applied in dataset loader to match nuScenes convention (x=lateral, y=forward). `point_cloud_range = [-51.2, 0, -5, 51.2, 51.2, 3]`
- Rotation applied in `loki_e2e_dataset.py` `get_data_info()` and `get_ann_info()` — NOT in the pkl itself
- `R_inv` (new→old): `[[0,1,0],[-1,0,0],[0,0,1]]` applied to `lidar2img`, `lidar2cam`, `l2g_r_mat`
- `can_bus` and `l2g_t` are in global frame — unchanged by rotation. The rotated `l2g_r_mat` handles the conversion in the transformer's BEV shift code.

## Key Files

### Dataset
- `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py` — Main dataset class `LokiE2EDataset`. Mirrors `NuScenesE2EDataset` but single-camera. Contains `get_ann_info()`, `get_data_info()`, `prepare_train_data()`, `union2one()`.
- `projects/mmdet3d_plugin/datasets/pipelines/loki_loading.py` — `LoadLokiImage` (single camera, scales lidar2img if resized from 1920x1208), `GenerateDummyOccLabels`
- `PKL_TO_CONFIG` dict maps lowercase pkl names → capitalized config names (e.g. `'car'→'Car'`)

### Config
- `projects/configs/loki/base_loki_perception.py` — Single config file. 8 classes, BEV 200x100 (w x h), ResNet101+DCN backbone.
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
- `tools/visualize_loki_gt.py` — Visualize ground truth only (GT 2D boxes + BEV + projection sanity check).

### Data
- `data/infos/loki_infos_train.pkl`, `data/infos/loki_infos_val.pkl` — Pre-computed info dicts
- Raw images at `/mnt/storage/loki_data/scenario_XXX/image_XXXX.png`
- Pkl info keys: `token, scene_token, frame_idx, img_filename, lidar2img, cam_intrinsic, lidar2cam, l2g_r_mat, l2g_t, can_bus, gt_boxes, gt_names, gt_labels, gt_inds, gt_velocity, valid_flag, num_lidar_pts, ego2global_rotation, ego2global_translation`
- `gt_boxes` shape: (N, 9) = [x, y, z, l, w, h, yaw, vx, vy]

### Training Outputs
- `work_dirs/base_loki_perception/` — Checkpoints (epoch_1-6.pth), results_epoch6.pkl, tensorboard logs
- Results pkl structure: `{'bbox_results': [{'token', 'boxes_3d', 'scores_3d', 'labels_3d', 'track_scores', 'track_ids', 'boxes_3d_det', 'scores_3d_det', 'labels_3d_det', 'track_bbox_results'}, ...]}`
- Results from epoch 6 are in the OLD coordinate system (before rotation was added)

## LOKI vs nuScenes Differences
1. Single camera (not 6)
2. Pre-computed transforms in pkl (l2g_r_mat as 3x3 matrix, not quaternion)
3. Zero-filled trajectories (no trajectory prediction data available)
4. Dummy occupancy labels
5. No NuScenes SDK dependency
6. 8 classes: Pedestrian, Car, Bus, Truck, Van, Motorcyclist, Bicyclist, Other

## Train / Eval Commands
```bash
# Train
./tools/uniad_dist_train.sh projects/configs/loki/base_loki_perception.py 1

# Eval (distributed required)
./tools/uniad_dist_eval.sh projects/configs/loki/base_loki_perception.py work_dirs/base_loki_perception/epoch_6.pth 1

# Visualize predictions vs GT (camera + BEV)
python tools/visualize_predictions.py \
    --results    work_dirs/base_loki_perception/results_epoch6.pkl \
    --val-pkl    data/infos/loki_infos_val.pkl \
    --data-root  /mnt/storage/loki_data \
    --out-dir    vis_predictions/ \
    --num-frames 20 --score-thresh 0.4
# Options: --tokens TOKEN1 TOKEN2 ...   --all-frames
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
