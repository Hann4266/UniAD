cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -u tools/train.py \
    projects/configs/loki/base_loki_perception.py \
    --deterministic \
    --cfg-options log_config.interval=1 \
    2>&1 | tee projects/work_dirs/loki/base_loki_perception/train_$(date +%Y%m%d_%H%M%S).log


cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python tools/debug_compare_dataloaders.py

cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/train.py \
    projects/configs/loki/base_loki_perception.py \
    --launcher pytorch \
    --deterministic \
    --cfg-options log_config.interval=1 \
    2>&1 | tee projects/work_dirs/loki/base_loki_perception/train_$(date +%Y%m%d_%H%M%S).log

### Running Nuscenes model zero-shot inference
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

### Running test
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/test.py \
    projects/configs/loki/base_loki_perception.py \
    work_dirs/base_loki_perception/epoch_6.pth \
    --launcher pytorch \
    --eval bbox track \
    --out work_dirs/base_loki_perception/results_epoch6.pkl

### Inference + Detection + Tracking full eval pipeline
cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=28596 \
    tools/test.py \
    projects/configs/loki/base_loki_perception.py \
    work_dirs/base_loki_perception/epoch_6.pth \
    --launcher pytorch \
    --eval bbox \
    --out work_dirs/base_loki_perception/results_rotated_epoch6.pkl \
    2>&1 | tee work_dirs/base_loki_perception/eval_rotated_epoch6.log


### First Evaluation Result:

======================================================================
  LOKI Detection Evaluation Results
======================================================================

Per-class Average Precision:
+--------------+--------+--------+--------+--------+--------+
|    Class     | d=0.5  | d=1.0  | d=2.0  | d=4.0  |  Mean  |
+--------------+--------+--------+--------+--------+--------+
|  Pedestrian  | 0.0003 | 0.0356 | 0.1588 | 0.3142 | 0.1272 |
|     Car      | 0.0136 | 0.0887 | 0.2011 | 0.3337 | 0.1593 |
|     Bus      | 0.0000 | 0.0000 | 0.0000 | 0.0159 | 0.0040 |
|    Truck     | 0.0024 | 0.0728 | 0.1896 | 0.3183 | 0.1458 |
|     Van      | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Motorcyclist | 0.0462 | 0.1761 | 0.3114 | 0.3801 | 0.2285 |
|  Bicyclist   | 0.0003 | 0.0441 | 0.1272 | 0.2152 | 0.0967 |
|    Other     | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
+--------------+--------+--------+--------+--------+--------+

True Positive Errors (lower is better):
+--------------+-----------+-----------+------------+---------+
|    Class     | trans_err | scale_err | orient_err | vel_err |
+--------------+-----------+-----------+------------+---------+
|  Pedestrian  |   0.9866  |   0.2577  |   1.0487   |  1.7651 |
|     Car      |   0.7848  |   0.1928  |   0.2539   |  2.1683 |
|     Bus      |   1.0576  |   0.2905  |   0.1485   |  2.4158 |
|    Truck     |   0.8004  |   0.2582  |   0.1223   |  1.6148 |
|     Van      |   1.0000  |   1.0000  |   1.0000   |  1.0000 |
| Motorcyclist |   0.6222  |   0.2564  |   0.1238   |  1.6036 |
|  Bicyclist   |   0.8241  |   0.2181  |   0.5394   |  2.3450 |
|    Other     |   1.0000  |   1.0000  |   1.0000   |  1.0000 |
+--------------+-----------+-----------+------------+---------+

mAP : 0.0952
NDS : 0.1628
mTRANSE: 0.8845
mSCALEE: 0.4342
mORIENTE: 0.5296
mVELE: 1.7391
======================================================================


======================================================================
  LOKI Tracking Evaluation
======================================================================
Accumulating tracking metrics ...

Overall Tracking Metrics:
+--------+----------+
| Metric |  Value   |
+--------+----------+
| AMOTA  |  0.1007  |
| AMOTP  |  1.7496  |
| RECALL |  0.1737  |
|  MOTA  |  0.0991  |
|  MOTP  |  0.9289  |
|  IDS   | 680.0000 |
|  FRAG  | 735.0000 |
|  TID   |  8.4162  |
|  LGD   | 12.9201  |
+--------+----------+

Per-class Tracking:
+--------------+--------+--------+--------+--------+----------+
|    Class     | AMOTA  | AMOTP  | RECALL |  MOTA  |   IDS    |
+--------------+--------+--------+--------+--------+----------+
|  Bicyclist   | 0.0607 | 1.7775 | 0.1194 | 0.0862 | 30.0000  |
|     Bus      | 0.0000 | 2.0000 | 0.0000 | 0.0000 |   N/A    |
|     Car      | 0.1520 | 1.6280 | 0.2466 | 0.1395 | 84.0000  |
| Motorcyclist | 0.2497 | 1.5380 | 0.2925 | 0.2156 |  8.0000  |
|    Other     |  N/A   |  N/A   |  N/A   |  N/A   |   N/A    |
|  Pedestrian  | 0.0087 | 1.8894 | 0.1332 | 0.0341 | 546.0000 |
|    Truck     | 0.1330 | 1.6645 | 0.2507 | 0.1191 | 12.0000  |
|     Van      |  N/A   |  N/A   |  N/A   |  N/A   |   N/A    |
+--------------+--------+--------+--------+--------+----------+
======================================================================

{'Pedestrian/AP_d0.5': 0.0002581533394952145, 'Pedestrian/AP_d1.0': 0.03559626159406073, 'Pedestrian/AP_d2.0': 0.15878857657520123, 'Pedestrian/AP_d4.0': 0.31424232350820486, 'Pedestrian/trans_err': 0.9865948738330526, 'Pedestrian/scale_err': 0.25772911009875, 'Pedestrian/orient_err': 1.048721435450146, 'Pedestrian/vel_err': 1.7650855020789418, 'Car/AP_d0.5': 0.013568381013869826, 'Car/AP_d1.0': 0.08870366109851562, 'Car/AP_d2.0': 0.2011114747679404, 'Car/AP_d4.0': 0.333719950330202, 'Car/trans_err': 0.7848297221162005, 'Car/scale_err': 0.19283001930213398, 'Car/orient_err': 0.2538592077161936, 'Car/vel_err': 2.168306664313805, 'Bus/AP_d0.5': 0.0, 'Bus/AP_d1.0': 0.0, 'Bus/AP_d2.0': 0.0, 'Bus/AP_d4.0': 0.015901923812362436, 'Bus/trans_err': 1.057618576074681, 'Bus/scale_err': 0.29051658921562307, 'Bus/orient_err': 0.1484839990288613, 'Bus/vel_err': 2.415760242070436, 'Truck/AP_d0.5': 0.002381417188755745, 'Truck/AP_d1.0': 0.07275395146465274, 'Truck/AP_d2.0': 0.1895848477883309, 'Truck/AP_d4.0': 0.3183180145675947, 'Truck/trans_err': 0.8004201056890857, 'Truck/scale_err': 0.25821621144489615, 'Truck/orient_err': 0.12232085395846029, 'Truck/vel_err': 1.6148242541853015, 'Van/AP_d0.5': 0.0, 'Van/AP_d1.0': 0.0, 'Van/AP_d2.0': 0.0, 'Van/AP_d4.0': 0.0, 'Van/trans_err': 1.0, 'Van/scale_err': 1.0, 'Van/orient_err': 1.0, 'Van/vel_err': 1.0, 'Motorcyclist/AP_d0.5': 0.04615476590498459, 'Motorcyclist/AP_d1.0': 0.17610894531573423, 'Motorcyclist/AP_d2.0': 0.3114294775460295, 'Motorcyclist/AP_d4.0': 0.3801391330477175, 'Motorcyclist/trans_err': 0.6222461534838485, 'Motorcyclist/scale_err': 0.2563817958807943, 'Motorcyclist/orient_err': 0.12383769528432904, 'Motorcyclist/vel_err': 1.6036226051159557, 'Bicyclist/AP_d0.5': 0.0003370408428484704, 'Bicyclist/AP_d1.0': 0.044143486017231275, 'Bicyclist/AP_d2.0': 0.12721779589879412, 'Bicyclist/AP_d4.0': 0.21515569897951298, 'Bicyclist/trans_err': 0.8240786506571932, 'Bicyclist/scale_err': 0.21813824566761175, 'Bicyclist/orient_err': 0.539358058536934, 'Bicyclist/vel_err': 2.344968465916306, 'Other/AP_d0.5': 0.0, 'Other/AP_d1.0': 0.0, 'Other/AP_d2.0': 0.0, 'Other/AP_d4.0': 0.0, 'Other/trans_err': 1.0, 'Other/scale_err': 1.0, 'Other/orient_err': 1.0, 'Other/vel_err': 1.0, 'mAP': 0.09517547751881372, 'NDS': 0.16276047246642192, 'mtrans_err': 0.8844735102317577, 'mscale_err': 0.43422649645122613, 'morient_err': 0.5295726562468656, 'mvel_err': 1.7390709667100932, 'track/amota': 0.10070040219855408, 'track/amotp': 1.7495501425428202, 'track/recall': 0.17372541090987118, 'track/mota': 0.09909007535417556, 'track/motp': 0.9289041750734146, 'track/ids': 680.0, 'track/frag': 735.0, 'track/tid': 8.41622240332753, 'track/lgd': 12.920145198414582, 'track/Bicyclist/amota': 0.06074272951287159, 'track/Bicyclist/amotp': 1.7774816800955933, 'track/Bus/amota': 0.0, 'track/Bus/amotp': 2.0, 'track/Car/amota': 0.1520386490879157, 'track/Car/amotp': 1.6280022115460024, 'track/Motorcyclist/amota': 0.24967814824003431, 'track/Motorcyclist/amotp': 1.5379696165677745, 'track/Pedestrian/amota': 0.008701097074631783, 'track/Pedestrian/amotp': 1.8893583134310696, 'track/Truck/amota': 0.13304178927587113, 'track/Truck/amotp': 1.6644890336164821}


## Stage 2: Intent Head Training

### Train command (8 GPU)
```bash
cd /root/UniAD && PYTHONPATH="$(pwd)/projects:$(pwd):$PYTHONPATH" \
python3 -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=28596 \
    tools/train.py projects/configs/loki/loki_stage2_intent.py \
    --launcher pytorch --deterministic \
    --work-dir /mnt/storage/UniAD/work_dirs/stage_2_base
```

### Config: projects/configs/loki/loki_stage2_intent.py
- `load_from`: `/mnt/storage/UniAD/work_dirs/base_loki_perception/epoch_10.pth`
- `data_root`: `/root/loki_data/`
- Frozen: backbone, neck, BN, BEV encoder (only intent head trains)
- `num_intent=7`, `map_features=False`, 3 decoder layers
- Loss: masked softmax focal loss, class_weight=[1.0, 1.39, 3.21, 6.19, 5.44, 10.05, 11.15]

### Intent label mapping (in create_loki_infos.py)
- Vehicles use `vehicle_state` column: Stopped/Parked→0, Other moving→1, LCL/Cut-in-left→2, LCR/Cut-in-right→3, TL→4, TR→5
- Pedestrians use `intended_actions` column: Stopped/Waiting→0, Moving→1, Crossing→6
- All valid agents have labels (0% unknown)
- Train distribution: MOVING 50%, STOP 40%, CROSS 6%, TL 1.8%, TR 1.6%, LCL 0.3%, LCR 0.3%

### Files changed for intent (loki branch)
- `tools/create_loki_infos.py` — gt_intent_labels in pkl
- `projects/mmdet3d_plugin/datasets/loki_e2e_dataset.py` — get_ann_info reads from pkl, union2one propagates
- `projects/mmdet3d_plugin/uniad/dense_heads/intent_head.py` — forward_train/forward_test for no-map
- `projects/mmdet3d_plugin/uniad/dense_heads/intent_head_plugin/modules.py` — map_features=False support
- `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py` — forward_test intent without seg_head
- `projects/configs/loki/loki_stage2_intent.py` — new config

### Regenerate pkl (required once)
```bash
cd /root/UniAD && python tools/create_loki_infos.py \
    --data-root /root/loki_data --out-dir data/infos
```

---

# LOKI Dataset Adaptation for UniAD

## Dataset Characteristics

- **Sensors**: Single front camera only (no LiDAR)
- **HD Maps**: None (dummy zero tensors for map ground truth)
- **Temporal Links**: No `prev_idx`/`next_idx` fields
- **Localization**: Odometry with Euler angles (roll, pitch, yaw)
- **Frame Rate**: 5 FPS

## `input_dict` Fields in `get_data_info()`

**Missing fields** (not in LOKI):
- `pts_filename` - No LiDAR point cloud
- `sweeps` - No LiDAR sweep aggregation
- `prev_idx` / `next_idx` - No temporal frame indexing
- `map_filename` - No HD map files
- Real `gt_lane_labels/bboxes/masks` - Dummy zeros instead

**LOKI specifics:**
- `can_bus.copy()` - Explicit copy to avoid mutation
- `timestamp` - Already in seconds (no conversion needed)
- All map-related ground truth are zero tensors

## Transform Computation: l2g

Pre-computed in `create_loki_infos.py`:

```python
# Parse odometry: [x, y, z, roll, pitch, yaw]
# Odometry files contain ABSOLUTE global poses (not relative deltas)
odom = parse_odom(f"odom_{frame:04d}.txt")

# ego2global transform (direct from odometry)
e2g_t = odom[:3]  # Direct copy of x, y, z (global position)
e2g_r = euler_to_rotation_matrix(odom[3], odom[4], odom[5])  # Euler → matrix

# lidar2ego is identity (no LiDAR sensor)
l2e_r = np.eye(3)
l2e_t = np.zeros(3)

# Compute lidar-to-global (effectively ego-to-global)
l2g_r_mat = l2e_r.T @ e2g_r.T  # = e2g_r.T
l2g_t = l2e_t @ e2g_r.T + e2g_t  # = e2g_t

# Stored in pickle
info['l2g_r_mat'] = l2g_r_mat
info['l2g_t'] = l2g_t
```

**Key insights:** 
- LOKI odometry files contain **absolute global poses**, not relative motion
- Since lidar frame = ego frame, `l2g = ego2global`
- No pose accumulation needed - values are already cumulative

## Class Name Mapping

Pickle stores lowercase, config uses capitalized:
```python
PKL_TO_CONFIG = {
    'car': 'Car',
    'truck': 'Truck', 
    'bus': 'Bus',
    'pedestrian': 'Pedestrian',
    'motorcycle': 'Motorcyclist',
    'bicycle': 'Bicyclist',
}
```

## Camera Intrinsics

Estimated from FOV=60°, resolution 1920×1208:
```python
FX = FY = (1920/2) / tan(30°) ≈ 1663.2
CX = 960.0
CY = 604.0
```

## Files Structure

```
tools/create_loki_infos.py          # Pickle generation (odometry → l2g)
projects/mmdet3d_plugin/datasets/
  └── loki_e2e_dataset.py           # Dataset loader (reads pre-computed l2g)
```


root@tingji-dual-mount-deployment-5bc54fff6-4nf26:/mnt/storage/UniAD# cd /mnt/storage/UniAD && PYTHONPATH=$(pwd):$PYTHONPATH python tools/debug_compare_dataloaders.py

================================================================================
  Building NuScenes dataset
================================================================================
  Symlink exists: /mnt/storage/UniAD/data/nuscenes
  Building NuScenesE2EDataset (this loads NuScenes SDK + maps, may take a minute)...
======
Loading NuScenes tables for version v1.0-trainval...
23 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
2631083 ego_pose,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
1166187 sample_annotation,
4 map,
Done loading in 45.116 seconds.
======
Reverse indexing ...
Done reverse indexing in 9.6 seconds.
======
  NuScenes loaded: 28130 samples
  CLASSES: ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
  Using index: 5
  scene_token: 0d2cc345342a460e94ff54748338ac22
  frame_idx: 5
  timestamp: 1526915245547336

================================================================================
  Building LOKI dataset
================================================================================
  LOKI loaded: 18084 samples
  CLASSES: ['Pedestrian', 'Car', 'Bus', 'Truck', 'Van', 'Motorcyclist', 'Bicyclist', 'Other']
  Using index: 3
  scene_token: scenario_001
  frame_idx: 6
  timestamp: 0.6

================================================================================
  STAGE 1: get_ann_info()
================================================================================

  KEY                            NUSCENES                                      LOKI                                          MATCH
  ------------------------------ --------------------------------------------- --------------------------------------------- -----
  command                        int=2                                         int=2                                         OK
  gt_bboxes_3d                   LiDARInstance3DBoxes [42, 9] torch.float32    LiDARInstance3DBoxes [39, 9] torch.float32    OK
  gt_fut_traj                    ndarray [42, 12, 2] float64                   ndarray [39, 12, 2] float32                   DIFF
  gt_fut_traj_mask               ndarray [42, 12, 2] float64                   ndarray [39, 12, 2] float32                   DIFF
  gt_inds                        ndarray [42] int64                            ndarray [39] int64                            OK
  gt_labels_3d                   ndarray [42] int64                            ndarray [39] int64                            OK
  gt_names                       ndarray [42] <U21                             ndarray [39] <U10                             DIFF
  gt_past_traj                   ndarray [42, 8, 2] float64                    ndarray [39, 8, 2] float32                    DIFF
  gt_past_traj_mask              ndarray [42, 8, 2] float64                    ndarray [39, 8, 2] float32                    DIFF
  gt_sdc_bbox                    DC(cpu_only=True, stack=False) -> LiDARInsta  DC(cpu_only=True, stack=False) -> LiDARInsta  OK
  gt_sdc_fut_traj                ndarray [1, 12, 2] float64                    ndarray [1, 12, 2] float32                    DIFF
  gt_sdc_fut_traj_mask           ndarray [1, 12, 2] float64                    ndarray [1, 12, 2] float32                    DIFF
  gt_sdc_label                   DC(cpu_only=False, stack=False) -> Tensor [1  DC(cpu_only=False, stack=False) -> Tensor [1  OK
  sdc_planning                   ndarray [1, 6, 3] float64                     ndarray [1, 6, 3] float32                     DIFF
  sdc_planning_mask              ndarray [1, 6, 2] float64                     ndarray [1, 6, 2] float32                     DIFF

  MISMATCHES (9):
    'gt_fut_traj':
      NuScenes: ndarray [42, 12, 2] float64
        values: [-7.4570, 23.8516, -7.4584, 23.8416, -7.4587]
      LOKI:     ndarray [39, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_fut_traj_mask':
      NuScenes: ndarray [42, 12, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [39, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_names':
      NuScenes: ndarray [42] <U21
        values: [truck, pedestrian, car, truck, car]
      LOKI:     ndarray [39] <U10
        values: [pedestrian, truck, car, pedestrian, pedestrian]
    'gt_past_traj':
      NuScenes: ndarray [42, 8, 2] float64
        values: [-0.0064, 0.0162, -0.0128, 0.0315, -0.0193]
      LOKI:     ndarray [39, 8, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_past_traj_mask':
      NuScenes: ndarray [42, 8, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [39, 8, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_sdc_fut_traj':
      NuScenes: ndarray [1, 12, 2] float64
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
      LOKI:     ndarray [1, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_sdc_fut_traj_mask':
      NuScenes: ndarray [1, 12, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [1, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'sdc_planning':
      NuScenes: ndarray [1, 6, 3] float64
        values: [0.0011, 0.0001, 1.5708, 0.0003, -0.0001]
      LOKI:     ndarray [1, 6, 3] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'sdc_planning_mask':
      NuScenes: ndarray [1, 6, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [1, 6, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

================================================================================
  STAGE 2: get_data_info()
================================================================================
/mnt/storage/UniAD/projects/mmdet3d_plugin/datasets/eval_utils/map_api.py:2097: ShapelyDeprecationWarning: Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.
  exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
/mnt/storage/UniAD/projects/mmdet3d_plugin/datasets/eval_utils/map_api.py:2098: ShapelyDeprecationWarning: Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.
  interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]

  KEY                            NUSCENES                                      LOKI                                          MATCH
  ------------------------------ --------------------------------------------- --------------------------------------------- -----
  ann_info                       dict keys=['command', 'gt_bboxes_3d', 'gt_fu  dict keys=['command', 'gt_bboxes_3d', 'gt_fu  OK
  cam_intrinsic                  list len=6, [0]: ndarray [4, 4] float64       list len=1, [0]: ndarray [4, 4] float64       OK
  can_bus                        ndarray [18] float64                          ndarray [18] float64                          OK
  command                        int=2                                         int=2                                         OK
  ego2global_rotation            list len=4, [0]: float=0.9997239173348829     ndarray [3, 3] float64                        DIFF
  ego2global_translation         list len=3, [0]: float=2234.259847599768      ndarray [3] float64                           DIFF
  frame_idx                      int=5                                         int=6                                         OK
  gt_lane_bboxes                 Tensor [23, 4] torch.int64                    Tensor [0, 4] torch.float32                   DIFF
  gt_lane_labels                 Tensor [23] torch.int64                       Tensor [0] torch.int64                        OK
  gt_lane_masks                  Tensor [23, 200, 200] torch.uint8             Tensor [0, 100, 200] torch.uint8              OK
  img_filename                   list len=6, [0]: str='./data/nuscenes/sample  list len=1, [0]: str='/mnt/storage/loki_data  OK
  l2g_r_mat                      ndarray [3, 3] float32                        ndarray [3, 3] float32                        OK
  l2g_t                          ndarray [3] float32                           ndarray [3] float32                           OK
  lidar2cam                      list len=6, [0]: ndarray [4, 4] float64       list len=1, [0]: ndarray [4, 4] float64       OK
  lidar2img                      list len=6, [0]: ndarray [4, 4] float64       list len=1, [0]: ndarray [4, 4] float64       OK
  map_filename                   None                                          ---MISSING---                                 MISS
  next_idx                       str='6f0c75b3c00f42768da57f52cb831ef4'        ---MISSING---                                 MISS
  occ_e2g_r_mats                 list len=7, [0]: Tensor [3, 3] torch.float32  list len=7, [0]: Tensor [3, 3] torch.float32  OK
  occ_e2g_t_vecs                 list len=7, [0]: Tensor [3] torch.float32     list len=7, [0]: Tensor [3] torch.float32     OK
  occ_future_ann_infos           list len=7, [0]: dict keys=['gt_bboxes_3d',   list len=7, [0]: None                         DIFF
  occ_has_invalid_frame          bool=False                                    bool=False                                    OK
  occ_img_is_valid               ndarray [9] bool                              ndarray [7] bool                              OK
  occ_l2e_r_mats                 list len=7, [0]: Tensor [3, 3] torch.float32  list len=7, [0]: Tensor [3, 3] torch.float32  OK
  occ_l2e_t_vecs                 list len=7, [0]: Tensor [3] torch.float32     list len=7, [0]: Tensor [3] torch.float32     OK
  prev_idx                       str='234549a607b046539a1a57ac8718edb7'        ---MISSING---                                 MISS
  pts_filename                   str='./data/nuscenes/samples/LIDAR_TOP/n008-  ---MISSING---                                 MISS
  sample_idx                     str='298ae5c93bdf4694a8574444b0153894'        str='scenario_001_frame_0006'                 OK
  scene_token                    str='0d2cc345342a460e94ff54748338ac22'        str='scenario_001'                            OK
  sdc_planning                   ndarray [1, 6, 3] float64                     ndarray [1, 6, 3] float32                     DIFF
  sdc_planning_mask              ndarray [1, 6, 2] float64                     ndarray [1, 6, 2] float32                     DIFF
  sweeps                         list len=10, [0]: dict keys=['data_path', 'e  ---MISSING---                                 MISS
  timestamp                      float=1526915245.547336                       float=0.6                                     OK

  MISMATCHES (11):
    'ego2global_rotation':
      NuScenes: list len=4, [0]: float=0.9997239173348829
        values: [0]: 0.9997239173348829
      LOKI:     ndarray [3, 3] float64
        values: [0.9998, -0.0108, -0.0145, 0.0108, 0.9999]
    'ego2global_translation':
      NuScenes: list len=3, [0]: float=2234.259847599768
        values: [0]: 2234.259847599768
      LOKI:     ndarray [3] float64
        values: [4.6659, 0.1544, -0.0067]
    'gt_lane_bboxes':
      NuScenes: Tensor [23, 4] torch.int64
        values: [56.0000, 20.0000, 58.0000, 47.0000, 79.0000]
      LOKI:     Tensor [0, 4] torch.float32
        values: []
    'map_filename':
      NuScenes: None
        values: 
      LOKI:     MISSING
    'next_idx':
      NuScenes: str='6f0c75b3c00f42768da57f52cb831ef4'
        values: 
      LOKI:     MISSING
    'occ_future_ann_infos':
      NuScenes: list len=7, [0]: dict keys=['gt_bboxes_3d', 'gt_inds', 'gt_labels_3d', 'gt_vis_tokens']
        values: [0]: 
      LOKI:     list len=7, [0]: None
        values: [0]: 
    'prev_idx':
      NuScenes: str='234549a607b046539a1a57ac8718edb7'
        values: 
      LOKI:     MISSING
    'pts_filename':
      NuScenes: str='./data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0'
        values: 
      LOKI:     MISSING
    'sdc_planning':
      NuScenes: ndarray [1, 6, 3] float64
        values: [0.0011, 0.0001, 1.5708, 0.0003, -0.0001]
      LOKI:     ndarray [1, 6, 3] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'sdc_planning_mask':
      NuScenes: ndarray [1, 6, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [1, 6, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'sweeps':
      NuScenes: list len=10, [0]: dict keys=['data_path', 'ego2global_rotation', 'ego2global_translation', 'sample_data_token', 'sensor2ego_rotation', 'sensor2ego_translation', 'sensor2lidar_rotation', 'sensor2lidar_translation', 'timestamp', 'type']
        values: [0]: 
      LOKI:     MISSING

  Critical field value comparison:

  >> can_bus:
    NuScenes: ndarray [18] float64
      values: [2234.2598, 857.4061, 0.0000, 0.9997, 0.0000]
    LOKI:     ndarray [18] float64
      values: [4.6659, 0.1544, -0.0067, 1.0000, 0.0000]

  >> l2g_r_mat:
    NuScenes: ndarray [3, 3] float32
      values: [0.0372, -0.9993, -0.0019, 0.9993, 0.0372]
    LOKI:     ndarray [3, 3] float32
      values: [0.9998, 0.0108, 0.0145, -0.0108, 0.9999]

  >> l2g_t:
    NuScenes: ndarray [3] float32
      values: [2235.1755, 857.4467, 1.8304]
    LOKI:     ndarray [3] float32
      values: [4.6659, 0.1544, -0.0067]

  >> ego2global_rotation:
    NuScenes: list len=4, [0]: float=0.9997239173348829
      values: [0]: 0.9997239173348829
    LOKI:     ndarray [3, 3] float64
      values: [0.9998, -0.0108, -0.0145, 0.0108, 0.9999]

  >> ego2global_translation:
    NuScenes: list len=3, [0]: float=2234.259847599768
      values: [0]: 2234.259847599768
    LOKI:     ndarray [3] float64
      values: [4.6659, 0.1544, -0.0067]

  >> lidar2img:
    NuScenes: list len=6, [0]: ndarray [4, 4] float64
      values: [0]: [1260.3598, 790.4438, 15.5615, -636.4484, 7.8044]
    LOKI:     list len=1, [0]: ndarray [4, 4] float64
      values: [0]: [960.0000, -1662.7688, 0.0000, 0.0000, 604.0000]

  >> img_filename:
    NuScenes: list len=6, [0]: str='./data/nuscenes/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0'
      values: [0]: 
    LOKI:     list len=1, [0]: str='/mnt/storage/loki_data/scenario_001/image_0006.png'
      values: [0]: 

  >> timestamp:
    NuScenes: float=1526915245.547336
      values: 1526915245.547336
    LOKI:     float=0.6
      values: 0.6

================================================================================
  STAGE 3: Single frame after pipeline
================================================================================

  KEY                            NUSCENES                                      LOKI                                          MATCH
  ------------------------------ --------------------------------------------- --------------------------------------------- -----
  command                        int=2                                         int=2                                         OK
  gt_backward_flow               Tensor [7, 2, 200, 200] torch.float32         Tensor [7, 2, 100, 200] torch.float32         OK
  gt_bboxes_3d                   DC(cpu_only=True, stack=False) -> LiDARInsta  DC(cpu_only=True, stack=False) -> LiDARInsta  OK
  gt_centerness                  Tensor [7, 1, 200, 200] torch.float32         Tensor [7, 1, 100, 200] torch.float32         OK
  gt_flow                        Tensor [7, 2, 200, 200] torch.float32         Tensor [7, 2, 100, 200] torch.float32         OK
  gt_fut_traj                    ndarray [39, 12, 2] float64                   ndarray [34, 12, 2] float32                   DIFF
  gt_fut_traj_mask               ndarray [39, 12, 2] float64                   ndarray [34, 12, 2] float32                   DIFF
  gt_future_boxes                list len=7, [0]: LiDARInstance3DBoxes [49, 9  list len=0                                    OK
  gt_future_labels               list len=7, [0]: ndarray [49] int64           list len=0                                    OK
  gt_inds                        ndarray [39] int64                            ndarray [34] int64                            OK
  gt_instance                    Tensor [7, 200, 200] torch.int64              Tensor [7, 100, 200] torch.int64              OK
  gt_labels_3d                   DC(cpu_only=False, stack=False) -> Tensor [3  DC(cpu_only=False, stack=False) -> Tensor [3  OK
  gt_lane_bboxes                 Tensor [23, 4] torch.int64                    Tensor [0, 4] torch.float32                   DIFF
  gt_lane_labels                 Tensor [23] torch.int64                       Tensor [0] torch.int64                        OK
  gt_lane_masks                  Tensor [23, 200, 200] torch.uint8             Tensor [0, 100, 200] torch.uint8              OK
  gt_occ_has_invalid_frame       bool=False                                    bool=False                                    OK
  gt_occ_img_is_valid            ndarray [9] bool                              ndarray [7] bool                              OK
  gt_offset                      Tensor [7, 2, 200, 200] torch.float32         Tensor [7, 2, 100, 200] torch.float32         OK
  gt_past_traj                   ndarray [39, 8, 2] float64                    ndarray [34, 8, 2] float32                    DIFF
  gt_past_traj_mask              ndarray [39, 8, 2] float64                    ndarray [34, 8, 2] float32                    DIFF
  gt_sdc_bbox                    DC(cpu_only=True, stack=False) -> LiDARInsta  DC(cpu_only=True, stack=False) -> LiDARInsta  OK
  gt_sdc_fut_traj                ndarray [1, 12, 2] float64                    ndarray [1, 12, 2] float32                    DIFF
  gt_sdc_fut_traj_mask           ndarray [1, 12, 2] float64                    ndarray [1, 12, 2] float32                    DIFF
  gt_sdc_label                   DC(cpu_only=False, stack=False) -> Tensor [1  DC(cpu_only=False, stack=False) -> Tensor [1  OK
  gt_segmentation                Tensor [7, 200, 200] torch.int64              Tensor [7, 100, 200] torch.int64              OK
  img                            DC(cpu_only=False, stack=True) -> Tensor [6,  DC(cpu_only=False, stack=True) -> Tensor [1,  OK
  img_metas                      DC(cpu_only=True, stack=False) -> dict keys=  DC(cpu_only=True, stack=False) -> dict keys=  OK
  l2g_r_mat                      ndarray [3, 3] float32                        ndarray [3, 3] float32                        OK
  l2g_t                          ndarray [3] float32                           ndarray [3] float32                           OK
  sdc_planning                   ndarray [1, 6, 3] float64                     ndarray [1, 6, 3] float32                     DIFF
  sdc_planning_mask              ndarray [1, 6, 2] float64                     ndarray [1, 6, 2] float32                     DIFF
  timestamp                      float=1526915245.547336                       float=0.6                                     OK

  MISMATCHES (9):
    'gt_fut_traj':
      NuScenes: ndarray [39, 12, 2] float64
        values: [-7.4570, 23.8516, -7.4584, 23.8416, -7.4587]
      LOKI:     ndarray [34, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_fut_traj_mask':
      NuScenes: ndarray [39, 12, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [34, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_lane_bboxes':
      NuScenes: Tensor [23, 4] torch.int64
        values: [56.0000, 20.0000, 58.0000, 47.0000, 79.0000]
      LOKI:     Tensor [0, 4] torch.float32
        values: []
    'gt_past_traj':
      NuScenes: ndarray [39, 8, 2] float64
        values: [-0.0064, 0.0162, -0.0128, 0.0315, -0.0193]
      LOKI:     ndarray [34, 8, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_past_traj_mask':
      NuScenes: ndarray [39, 8, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [34, 8, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_sdc_fut_traj':
      NuScenes: ndarray [1, 12, 2] float64
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
      LOKI:     ndarray [1, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_sdc_fut_traj_mask':
      NuScenes: ndarray [1, 12, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [1, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'sdc_planning':
      NuScenes: ndarray [1, 6, 3] float64
        values: [0.0011, 0.0001, 1.5708, 0.0003, -0.0001]
      LOKI:     ndarray [1, 6, 3] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'sdc_planning_mask':
      NuScenes: ndarray [1, 6, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [1, 6, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

================================================================================
  STAGE 4: prepare_train_data() -> union2one()
================================================================================

  KEY                            NUSCENES                                      LOKI                                          MATCH
  ------------------------------ --------------------------------------------- --------------------------------------------- -----
  command                        int=2                                         int=2                                         OK
  gt_backward_flow               Tensor [7, 2, 200, 200] torch.float32         Tensor [7, 2, 100, 200] torch.float32         OK
  gt_bboxes_3d                   DC(cpu_only=True, stack=False) -> list len=5  DC(cpu_only=True, stack=False) -> list len=3  OK
  gt_centerness                  Tensor [7, 1, 200, 200] torch.float32         Tensor [7, 1, 100, 200] torch.float32         OK
  gt_flow                        Tensor [7, 2, 200, 200] torch.float32         Tensor [7, 2, 100, 200] torch.float32         OK
  gt_fut_traj                    DC(cpu_only=False, stack=False) -> Tensor [3  DC(cpu_only=False, stack=False) -> Tensor [3  DIFF
  gt_fut_traj_mask               DC(cpu_only=False, stack=False) -> Tensor [3  DC(cpu_only=False, stack=False) -> Tensor [3  DIFF
  gt_future_boxes                DC(cpu_only=True, stack=False) -> list len=7  DC(cpu_only=True, stack=False) -> list len=0  OK
  gt_future_labels               DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK
  gt_inds                        DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK
  gt_instance                    Tensor [7, 200, 200] torch.int64              Tensor [7, 100, 200] torch.int64              OK
  gt_labels_3d                   DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK
  gt_lane_bboxes                 Tensor [23, 4] torch.int64                    Tensor [0, 4] torch.float32                   DIFF
  gt_lane_labels                 Tensor [23] torch.int64                       Tensor [0] torch.int64                        OK
  gt_lane_masks                  Tensor [23, 200, 200] torch.uint8             Tensor [0, 100, 200] torch.uint8              OK
  gt_occ_has_invalid_frame       bool=False                                    bool=False                                    OK
  gt_occ_img_is_valid            ndarray [9] bool                              ndarray [7] bool                              OK
  gt_offset                      Tensor [7, 2, 200, 200] torch.float32         Tensor [7, 2, 100, 200] torch.float32         OK
  gt_past_traj                   DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK
  gt_past_traj_mask              DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK
  gt_sdc_bbox                    DC(cpu_only=True, stack=False) -> list len=5  DC(cpu_only=True, stack=False) -> list len=3  OK
  gt_sdc_fut_traj                ndarray [1, 12, 2] float64                    ndarray [1, 12, 2] float32                    DIFF
  gt_sdc_fut_traj_mask           ndarray [1, 12, 2] float64                    ndarray [1, 12, 2] float32                    DIFF
  gt_sdc_label                   DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK
  gt_segmentation                Tensor [7, 200, 200] torch.int64              Tensor [7, 100, 200] torch.int64              OK
  img                            DC(cpu_only=False, stack=True) -> Tensor [5,  DC(cpu_only=False, stack=True) -> Tensor [3,  OK
  img_metas                      DC(cpu_only=True, stack=False) -> dict keys=  DC(cpu_only=True, stack=False) -> dict keys=  OK
  l2g_r_mat                      DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK
  l2g_t                          DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK
  sdc_planning                   ndarray [1, 6, 3] float64                     ndarray [1, 6, 3] float32                     DIFF
  sdc_planning_mask              ndarray [1, 6, 2] float64                     ndarray [1, 6, 2] float32                     DIFF
  timestamp                      DC(cpu_only=False, stack=False) -> list len=  DC(cpu_only=False, stack=False) -> list len=  OK

  MISMATCHES (7):
    'gt_fut_traj':
      NuScenes: DC(cpu_only=False, stack=False) -> Tensor [39, 12, 2] torch.float64
        values: [-7.4570, 23.8516, -7.4584, 23.8416, -7.4587]
      LOKI:     DC(cpu_only=False, stack=False) -> Tensor [34, 12, 2] torch.float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_fut_traj_mask':
      NuScenes: DC(cpu_only=False, stack=False) -> Tensor [39, 12, 2] torch.float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     DC(cpu_only=False, stack=False) -> Tensor [34, 12, 2] torch.float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_lane_bboxes':
      NuScenes: Tensor [23, 4] torch.int64
        values: [56.0000, 20.0000, 58.0000, 47.0000, 79.0000]
      LOKI:     Tensor [0, 4] torch.float32
        values: []
    'gt_sdc_fut_traj':
      NuScenes: ndarray [1, 12, 2] float64
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
      LOKI:     ndarray [1, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'gt_sdc_fut_traj_mask':
      NuScenes: ndarray [1, 12, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [1, 12, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'sdc_planning':
      NuScenes: ndarray [1, 6, 3] float64
        values: [0.0011, 0.0001, 1.5708, 0.0003, -0.0001]
      LOKI:     ndarray [1, 6, 3] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    'sdc_planning_mask':
      NuScenes: ndarray [1, 6, 2] float64
        values: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
      LOKI:     ndarray [1, 6, 2] float32
        values: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

  NuScenes detailed inspection:
    img: [5, 6, 3, 928, 1600] (queue, num_cams, C, H, W)
    img_metas: 5 frames
      frame 0: prev_bev=False, can_bus=(shape=(18,), [:3]=[0. 0. 0.], [-1]=0.0), lidar2img=(list len=6, [0].shape=(4, 4)), img_shape=[(928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3)]
      frame 1: prev_bev=True, can_bus=(shape=(18,), [:3]=[0. 0. 0.], [-1]=1.1647837805384142e-08), lidar2img=(list len=6, [0].shape=(4, 4)), img_shape=[(928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3)]
      frame 2: prev_bev=True, can_bus=(shape=(18,), [:3]=[0. 0. 0.], [-1]=-1.982143782441881e-09), lidar2img=(list len=6, [0].shape=(4, 4)), img_shape=[(928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3)]
      frame 3: prev_bev=True, can_bus=(shape=(18,), [:3]=[4.54747351e-13 0.00000000e+00 0.00000000e+00], [-1]=1.2171459395915463e-09), lidar2img=(list len=6, [0].shape=(4, 4)), img_shape=[(928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3)]
      frame 4: prev_bev=True, can_bus=(shape=(18,), [:3]=[-4.54747351e-13  0.00000000e+00  0.00000000e+00], [-1]=6.005445030154988e-10), lidar2img=(list len=6, [0].shape=(4, 4)), img_shape=[(928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3)]

  LOKI detailed inspection:
    img: [3, 1, 3, 928, 1600] (queue, num_cams, C, H, W)
    img_metas: 3 frames
      frame 0: prev_bev=False, canbus=(shape=(18,)_, [:3]=[0. 0. 0.], [-1]=0.0), lidar2img=(list len=1, [0].shape=(4, 4)), img_shape=[(928, 1600, 3)]
      frame 1: prev_bev=True, can_bus=(shape=(18,), [:3]=[1.52207458 0.05081277 0.00549993], [-1]=0.20459217081764702), lidar2img=(list len=1, [0].shape=(4, 4)), img_shape=[(928, 1600, 3)]
      frame 2: prev_bev=True, can_bus=(shape=(18,), [:3]=[1.44238734 0.04162998 0.01825937], [-1]=0.13724902913422787), lidar2img=(list len=1, [0].shape=(4, 4)), img_shape=[(928, 1600, 3)]

================================================================================
  COMPARISON COMPLETE
================================================================================
Key differences to expect (by design):
  - img: NuScenes=[Q,6,3,H,W] vs LOKI=[Q,1,3,H,W] (single camera)
  - lidar2img: NuScenes=list of 6 vs LOKI=list of 1
  - ego2global_rotation: NuScenes=quaternion[4] vs LOKI=matrix[3,3]
  - NuScenes has pts_filename, sweeps, prev_idx, next_idx (LOKI doesn't)
  - LOKI trajectories are zero-filled (no traj_api)
  - LOKI map labels are empty (no HD map)