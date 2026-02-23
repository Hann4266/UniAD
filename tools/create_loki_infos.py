#!/usr/bin/env python
"""
Create LOKI dataset info file (loki_infos.pkl) for UniAD.

Scans the LOKI dataset directory, parses all annotations, computes
transforms, velocities, and trajectory labels, and saves a pickle
file compatible with the LOKI UniAD dataloader.

Usage:
    python tools/create_loki_infos.py \
        --data-root /mnt/storage/loki_data \
        --out-dir data/infos \
        --split-ratio 0.8 0.1 0.1
"""

import argparse
import os
import json
import csv
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import io
from plyfile import PlyData


# --------------------------------------------------------------------- #
#  LOKI → UniAD class mapping
# --------------------------------------------------------------------- #
LOKI_CLASS_MAP = {
    'Car': 'car',
    'Truck': 'truck',
    'Van': 'car',        # map Van → car
    'Bus': 'bus',
    'Pedestrian': 'pedestrian',
    'Motorcyclist': 'motorcycle',
    'Bicyclist': 'bicycle',
}
LOKI_CLASSES = list(dict.fromkeys(LOKI_CLASS_MAP.values()))
# ['car', 'truck', 'bus', 'pedestrian', 'motorcycle', 'bicycle']

# Camera intrinsics estimation (FOV=60°, resolution 1920×1208)
ORIG_W, ORIG_H = 1920, 1208
FOV_DEG = 60.0
FX = (ORIG_W / 2.0) / np.tan(np.deg2rad(FOV_DEG / 2.0))
FY = FX  # square pixels
CX = ORIG_W / 2.0
CY = ORIG_H / 2.0

CAM_INTRINSIC = np.array([
    [FX,  0.0, CX, 0.0],
    [0.0, FY,  CY, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# LiDAR-to-camera extrinsic: assume camera is co-located with LiDAR,
# looking forward along +X in LiDAR frame.
# Standard convention: camera Z = LiDAR X, camera X = -LiDAR Y, camera Y = -LiDAR Z
LIDAR2CAM = np.array([
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [1.0,  0.0,  0.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=np.float64)

LIDAR2IMG = CAM_INTRINSIC @ LIDAR2CAM


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #
def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to a 3×3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def parse_odom(filepath):
    """Parse a single odometry file → (x, y, z, roll, pitch, yaw).

    Returns None if the file is corrupted or unreadable.
    """
    try:
        with open(filepath, 'r') as f:
            parts = [v for v in f.readline().strip().split(',') if v]
        if len(parts) != 6:
            print(f"  WARNING: skipping bad odom {filepath}: got {len(parts)} values")
            return None
        vals = [float(v) for v in parts]
        return np.array(vals, dtype=np.float64)
    except Exception as e:
        print(f"  WARNING: skipping corrupted odom {filepath}: {e}")
        return None


def parse_label3d(filepath):
    """Parse a 3D label file → list of annotation dicts."""
    annotations = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  WARNING: skipping corrupted label3d {filepath}: {e}")
        return annotations
    if len(lines) <= 1:
        return annotations
    # skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        # Use csv reader to handle commas properly
        reader = csv.reader(io.StringIO(line))
        parts = next(reader)
        if len(parts) < 14:
            continue
        label = parts[0].strip()
        if label not in LOKI_CLASS_MAP:
            continue
        track_id = parts[1].strip()
        stationary = parts[2].strip()
        pos_x = float(parts[3])
        pos_y = float(parts[4])
        pos_z = float(parts[5])
        dim_x = float(parts[6])
        dim_y = float(parts[7])
        dim_z = float(parts[8])
        yaw = float(parts[9])
        vehicle_state = parts[10].strip()
        intended_actions = parts[11].strip()

        annotations.append(dict(
            label=label,
            mapped_class=LOKI_CLASS_MAP[label],
            track_id=track_id,
            stationary=(stationary == 'stationary'),
            pos=np.array([pos_x, pos_y, pos_z], dtype=np.float64),
            dim=np.array([dim_x, dim_y, dim_z], dtype=np.float64),
            yaw=yaw,
            vehicle_state=vehicle_state,
            intended_actions=intended_actions,
        ))
    return annotations


def get_sorted_frame_indices(scenario_dir):
    """Get sorted frame indices from image files in the scenario dir."""
    frame_indices = set()
    for fname in os.listdir(scenario_dir):
        m = re.match(r'image_(\d+)\.png', fname)
        if m:
            frame_indices.add(int(m.group(1)))
    return sorted(frame_indices)


def get_frame_data_availability(scenario_dir, frame_idx):
    """Check which data files exist for a given frame index."""
    fid = f"{frame_idx:04d}"
    return {
        'image': os.path.exists(os.path.join(scenario_dir, f"image_{fid}.png")),
        'label3d': os.path.exists(os.path.join(scenario_dir, f"label3d_{fid}.txt")),
        'label2d': os.path.exists(os.path.join(scenario_dir, f"label2d_{fid}.json")),
        'odom': os.path.exists(os.path.join(scenario_dir, f"odom_{fid}.txt")),
        'pc': os.path.exists(os.path.join(scenario_dir, f"pc_{fid}.ply")),
    }


def load_lidar_points(ply_path):
    """Load LiDAR point cloud from PLY file → (N, 3) xyz array.

    Returns None if the file is corrupted or unreadable.
    """
    try:
        ply = PlyData.read(ply_path)
        v = ply['vertex']
        return np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64)
    except Exception as e:
        print(f"  WARNING: skipping corrupted PLY {ply_path}: {e}")
        return None


def count_points_in_boxes(pts, boxes):
    """Count LiDAR points inside each 3D bounding box.

    Args:
        pts: (N, 3) LiDAR points [x, y, z].
        boxes: list of dicts with 'pos' (3,), 'dim' (3,), 'yaw' (float).

    Returns:
        np.ndarray: (M,) point count per box.
    """
    counts = np.zeros(len(boxes), dtype=np.int64)
    for i, box in enumerate(boxes):
        cx, cy, cz = box['pos']
        dx, dy, dz = box['dim']
        yaw = box['yaw']

        # Translate to box center
        local = pts - np.array([cx, cy, cz])
        # Rotate to box-local frame
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        local_x = cos_y * local[:, 0] - sin_y * local[:, 1]
        local_y = sin_y * local[:, 0] + cos_y * local[:, 1]
        local_z = local[:, 2]
        # Check inside box
        inside = ((np.abs(local_x) <= dx / 2) &
                  (np.abs(local_y) <= dy / 2) &
                  (np.abs(local_z) <= dz / 2))
        counts[i] = inside.sum()
    return counts


def process_scenario(scenario_dir, scenario_name, global_track_id_map):
    """Process a single scenario and return a list of frame info dicts."""
    frame_indices = get_sorted_frame_indices(scenario_dir)
    if len(frame_indices) == 0:
        return []

    # ------------------------------------------------------------------ #
    # 1. Parse all odometry and accumulate ego poses
    # ------------------------------------------------------------------ #
    odom_data = {}
    for fidx in frame_indices:
        odom_path = os.path.join(scenario_dir, f"odom_{fidx:04d}.txt")
        if os.path.exists(odom_path):
            parsed = parse_odom(odom_path)
            if parsed is not None:
                odom_data[fidx] = parsed

    # Accumulate ego poses: first frame as origin
    # Odometry values appear to already be cumulative poses (positions grow)
    ego_poses = {}  # fidx → (translation_3, rotation_matrix_3x3)
    for fidx in frame_indices:
        if fidx in odom_data:
            odom = odom_data[fidx]
            t = odom[:3]  # x, y, z
            R = euler_to_rotation_matrix(odom[3], odom[4], odom[5])
            ego_poses[fidx] = (t.copy(), R.copy())
        else:
            ego_poses[fidx] = (np.zeros(3), np.eye(3))

    # ------------------------------------------------------------------ #
    # 2. Parse all 3D labels and build per-frame annotations
    # ------------------------------------------------------------------ #
    all_label3d = {}
    for fidx in frame_indices:
        label_path = os.path.join(scenario_dir, f"label3d_{fidx:04d}.txt")
        if os.path.exists(label_path):
            all_label3d[fidx] = parse_label3d(label_path)
        else:
            all_label3d[fidx] = []

    # Build global integer track_id map (UUIDs → ints, starting from 1)
    for fidx in frame_indices:
        for ann in all_label3d.get(fidx, []):
            uuid = ann['track_id']
            if uuid not in global_track_id_map:
                global_track_id_map[uuid] = len(global_track_id_map) + 1

    # ------------------------------------------------------------------ #
    # 3. Compute per-object velocities across frames
    # ------------------------------------------------------------------ #
    # Group positions by track_id across frames
    track_positions = defaultdict(dict)  # uuid → {fidx: pos}
    for fidx in frame_indices:
        for ann in all_label3d.get(fidx, []):
            track_positions[ann['track_id']][fidx] = ann['pos'].copy()

    # Compute velocity for each (track_id, frame) pair
    track_velocities = defaultdict(dict)  # uuid → {fidx: vel}
    fps = 5.0  # LOKI is 5 FPS (frames every 0.2s based on even indexing)
    # The frame step is 2 (0000, 0002, 0004...) so dt between consecutive = 1/fps
    for uuid, positions in track_positions.items():
        sorted_fidxs = sorted(positions.keys())
        for i, fidx in enumerate(sorted_fidxs):
            if i + 1 < len(sorted_fidxs):
                next_fidx = sorted_fidxs[i + 1]
                dt = (next_fidx - fidx) / 2.0 / fps  # frame step=2, at 5fps
                if dt > 0:
                    vel = (positions[next_fidx] - positions[fidx]) / dt
                    track_velocities[uuid][fidx] = vel[:2]  # vx, vy only
                else:
                    track_velocities[uuid][fidx] = np.zeros(2)
            else:
                # last frame: copy previous velocity or zero
                if i > 0:
                    prev_fidx = sorted_fidxs[i - 1]
                    track_velocities[uuid][fidx] = track_velocities[uuid].get(
                        prev_fidx, np.zeros(2))
                else:
                    track_velocities[uuid][fidx] = np.zeros(2)

    # ------------------------------------------------------------------ #
    # 4. Build frame info dicts
    # ------------------------------------------------------------------ #
    frame_infos = []
    for fidx in frame_indices:
        avail = get_frame_data_availability(scenario_dir, fidx)
        if not avail['image']:
            continue

        fid_str = f"{fidx:04d}"
        img_path = os.path.join(scenario_dir, f"image_{fid_str}.png")

        # Ego pose
        ego_t, ego_R = ego_poses.get(fidx, (np.zeros(3), np.eye(3)))

        # Since we treat lidar frame = ego frame, lidar2ego is identity
        l2e_r = np.eye(3, dtype=np.float64)
        l2e_t = np.zeros(3, dtype=np.float64)
        e2g_r = ego_R.astype(np.float64)
        e2g_t = ego_t.astype(np.float64)

        # lidar to global
        l2g_r_mat = (l2e_r.T @ e2g_r.T).astype(np.float32)
        l2g_t_vec = (l2e_t @ e2g_r.T + e2g_t).astype(np.float32)

        # can_bus: [x, y, z, qw, qx, qy, qz, 0..0, patch_angle_rad, patch_angle_deg]
        # We fill 18 entries to match nuScenes can_bus format
        yaw = np.arctan2(ego_R[1, 0], ego_R[0, 0])
        patch_angle = np.rad2deg(yaw)
        if patch_angle < 0:
            patch_angle += 360.0

        can_bus = np.zeros(18, dtype=np.float64)
        can_bus[0:3] = ego_t  # global translation
        # quaternion from rotation matrix (simplified: just store identity-like)
        # We don't need exact quaternion, just yaw for BEV rotation
        can_bus[3] = 1.0  # qw
        can_bus[4:7] = 0.0  # qx, qy, qz
        can_bus[-2] = np.deg2rad(patch_angle)  # rad
        can_bus[-1] = patch_angle  # deg

        # 3D annotations
        anns = all_label3d.get(fidx, [])
        n_obj = len(anns)

        # Camera visibility: check if each 3D agent has a 2D bbox in label2d
        label2d_path = os.path.join(scenario_dir, f"label2d_{fid_str}.json")
        visible_uuids = set()
        if os.path.exists(label2d_path):
            with open(label2d_path, 'r') as f2d:
                data2d = json.load(f2d)
            for cls_entries in data2d.values():
                if isinstance(cls_entries, dict):
                    visible_uuids.update(cls_entries.keys())
        gt_camera_visible = np.array(
            [ann['track_id'] in visible_uuids for ann in anns],
            dtype=bool) if n_obj > 0 else np.array([], dtype=bool)

        if n_obj > 0:
            gt_boxes = np.zeros((n_obj, 9), dtype=np.float64)
            gt_names = []
            gt_labels = np.zeros(n_obj, dtype=np.int64)
            gt_inds = np.zeros(n_obj, dtype=np.int64)
            gt_velocity = np.zeros((n_obj, 2), dtype=np.float64)

            for i, ann in enumerate(anns):
                cls_name = ann['mapped_class']
                gt_names.append(cls_name)
                gt_labels[i] = LOKI_CLASSES.index(cls_name)
                gt_inds[i] = global_track_id_map[ann['track_id']]

                # Box: [x, y, z, dx, dy, dz, yaw]
                gt_boxes[i, 0] = ann['pos'][0]
                gt_boxes[i, 1] = ann['pos'][1]
                gt_boxes[i, 2] = ann['pos'][2]
                gt_boxes[i, 3] = ann['dim'][0]  # dx (width in LOKI = dim_x)
                gt_boxes[i, 4] = ann['dim'][1]  # dy (length in LOKI = dim_y)
                gt_boxes[i, 5] = ann['dim'][2]  # dz (height in LOKI = dim_z)
                gt_boxes[i, 6] = ann['yaw']

                vel = track_velocities[ann['track_id']].get(fidx, np.zeros(2))
                gt_velocity[i] = vel
                gt_boxes[i, 7] = vel[0]  # vx
                gt_boxes[i, 8] = vel[1]  # vy

            gt_names = np.array(gt_names)
        else:
            gt_boxes = np.zeros((0, 9), dtype=np.float64)
            gt_names = np.array([], dtype='<U20')
            gt_labels = np.array([], dtype=np.int64)
            gt_inds = np.array([], dtype=np.int64)
            gt_velocity = np.zeros((0, 2), dtype=np.float64)

        # Timestamp: use frame index / fps as seconds
        timestamp = float(fidx) / 2.0 / fps  # convert frame step to seconds

        # LiDAR point cloud path
        pc_path = os.path.join(scenario_dir, f"pc_{fid_str}.ply")

        # Compute real num_lidar_pts per GT box
        if avail['pc'] and n_obj > 0:
            lidar_pts = load_lidar_points(pc_path)
            if lidar_pts is not None:
                box_dicts = [
                    dict(pos=anns[j]['pos'], dim=anns[j]['dim'], yaw=anns[j]['yaw'])
                    for j in range(n_obj)
                ]
                num_lidar_pts = count_points_in_boxes(lidar_pts, box_dicts)
            else:
                num_lidar_pts = np.zeros(n_obj, dtype=np.int64)
        else:
            num_lidar_pts = np.zeros(n_obj, dtype=np.int64)

        info = dict(
            # Identity
            scenario=scenario_name,
            scene_token=scenario_name,
            token=f"{scenario_name}_frame_{fid_str}",
            frame_idx=fidx,
            timestamp=timestamp,

            # Paths
            img_filename=img_path,
            pts_filename=pc_path if avail['pc'] else '',

            # Ego transforms
            lidar2ego_rotation=l2e_r,
            lidar2ego_translation=l2e_t,
            ego2global_rotation=e2g_r,
            ego2global_translation=e2g_t,
            l2g_r_mat=l2g_r_mat,
            l2g_t=l2g_t_vec,
            can_bus=can_bus,

            # Camera
            lidar2img=LIDAR2IMG.astype(np.float64),
            cam_intrinsic=CAM_INTRINSIC.astype(np.float64),
            lidar2cam=LIDAR2CAM.astype(np.float64),

            # Annotations
            gt_boxes=gt_boxes.astype(np.float32),
            gt_names=gt_names,
            gt_labels=gt_labels,
            gt_inds=gt_inds,
            gt_velocity=gt_velocity.astype(np.float32),

            # Validity based on real LiDAR point counts
            valid_flag=num_lidar_pts > 0 if n_obj > 0 else np.ones(0, dtype=bool),
            num_lidar_pts=num_lidar_pts,

            # Camera visibility: True if the 3D agent has a 2D bbox in label2d
            gt_camera_visible=gt_camera_visible,
        )
        frame_infos.append(info)

    return frame_infos


def split_scenarios(scenario_names, ratios):
    """Split scenario names into train/val/test."""
    np.random.seed(42)
    perm = np.random.permutation(len(scenario_names))
    n = len(scenario_names)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_names = [scenario_names[i] for i in perm[:n_train]]
    val_names = [scenario_names[i] for i in perm[n_train:n_train + n_val]]
    test_names = [scenario_names[i] for i in perm[n_train + n_val:]]
    return train_names, val_names, test_names


def main():
    parser = argparse.ArgumentParser(description='Create LOKI infos for UniAD')
    parser.add_argument('--data-root', type=str,
                        default='/mnt/storage/loki_data',
                        help='Root path of the LOKI dataset')
    parser.add_argument('--out-dir', type=str,
                        default='data/infos',
                        help='Output directory for info files')
    parser.add_argument('--split-ratio', type=float, nargs=3,
                        default=[0.8, 0.1, 0.1],
                        help='Train/val/test split ratio')
    args = parser.parse_args()

    data_root = args.data_root
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Discover scenarios
    scenario_names = sorted([
        d for d in os.listdir(data_root)
        if d.startswith('scenario_') and os.path.isdir(os.path.join(data_root, d))
    ])
    print(f"Found {len(scenario_names)} scenarios")

    # Split
    train_scenarios, val_scenarios, test_scenarios = split_scenarios(
        scenario_names, args.split_ratio)
    print(f"Split: train={len(train_scenarios)}, val={len(val_scenarios)}, test={len(test_scenarios)}")

    # Process all scenarios
    global_track_id_map = {}
    all_infos = {}
    for i, scenario_name in enumerate(scenario_names):
        scenario_dir = os.path.join(data_root, scenario_name)
        frame_infos = process_scenario(scenario_dir, scenario_name, global_track_id_map)
        all_infos[scenario_name] = frame_infos
        if (i + 1) % 10 == 0 or (i + 1) == len(scenario_names):
            print(f"  Processed {i + 1}/{len(scenario_names)} scenarios "
                  f"({sum(len(v) for v in all_infos.values())} frames total)",
                  flush=True)

    # Build split info lists
    def build_split_infos(split_scenarios):
        infos = []
        for name in sorted(split_scenarios):
            infos.extend(all_infos.get(name, []))
        return infos

    train_infos = build_split_infos(train_scenarios)
    val_infos = build_split_infos(val_scenarios)
    test_infos = build_split_infos(test_scenarios)

    print(f"\nFinal counts: train={len(train_infos)}, val={len(val_infos)}, test={len(test_infos)}")

    # Save metadata
    metadata = dict(
        version='loki_v1.0',
        classes=LOKI_CLASSES,
        class_map=LOKI_CLASS_MAP,
        num_scenarios=len(scenario_names),
        cam_intrinsic=CAM_INTRINSIC,
        lidar2img=LIDAR2IMG,
        lidar2cam=LIDAR2CAM,
    )

    # Save pickle files
    for split_name, infos in [('train', train_infos), ('val', val_infos), ('test', test_infos)]:
        data = dict(
            infos=infos,
            metadata=metadata,
        )
        out_path = os.path.join(out_dir, f'loki_infos_{split_name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {out_path} ({len(infos)} frames)")

    # Save split info for reproducibility
    split_info = dict(
        train=sorted(train_scenarios),
        val=sorted(val_scenarios),
        test=sorted(test_scenarios),
    )
    split_path = os.path.join(out_dir, 'loki_split.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump(split_info, f)
    print(f"Saved split info to {split_path}")

    # Save track_id map
    track_map_path = os.path.join(out_dir, 'loki_track_id_map.pkl')
    with open(track_map_path, 'wb') as f:
        pickle.dump(global_track_id_map, f)
    print(f"Saved track ID map ({len(global_track_id_map)} unique tracks)")

    print("\nDone!")


if __name__ == '__main__':
    main()
