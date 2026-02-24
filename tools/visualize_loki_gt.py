#!/usr/bin/env python3
"""
Visualize LOKI ground truth data to verify dataset loader correctness.

Produces for each sampled frame:
  1. Camera view  – front image with 2D bounding boxes from label2d files
                    showing relation to 3D filtering stages
  2. BEV view     – bird's-eye-view with oriented 3D boxes in the rotated
                    lidar frame (x=right, y=forward), range boundary, and
                    60° FOV cone overlay

Filtering stages shown:
  - Range filter (point cloud range)
  - FOV filter (front-camera cone)
  - Camera-visibility filter (`gt_camera_visible`, i.e. 3D box has matching 2D ID)

Usage:
    python tools/visualize_loki_gt.py \
        --pkl data/infos/loki_infos_val.pkl \
        --data-root /mnt/storage/loki_data \
        --out-dir viz_loki_gt \
        --num-samples 20 \
        --seed 42
"""

import argparse
import os
import pickle
import csv
import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Wedge
from PIL import Image
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
# Rotated frame range (x=right, y=forward) — matches config
POINT_CLOUD_RANGE = [-51.2, 0, -5.0, 51.2, 51.2, 3.0]
ORIG_IMG_SIZE = (1920, 1208)  # W, H

# LOKI raw class → mapped class (same as create_loki_infos.py)
LOKI_CLASS_MAP = {
    'Car': 'car',
    'Truck': 'truck',
    'Van': 'car',
    'Bus': 'bus',
    'Pedestrian': 'pedestrian',
    'Motorcyclist': 'motorcycle',
    'Bicyclist': 'bicycle',
}

# Class colours (for 2D boxes on image)
CLASS_COLORS = {
    "car":        (0.12, 0.56, 1.0),   # dodger blue
    "truck":      (1.0,  0.55, 0.0),   # orange
    "bus":        (0.58, 0.0,  0.83),   # purple
    "pedestrian": (1.0,  0.27, 0.0),   # red-orange
    "motorcycle": (0.0,  0.8,  0.4),   # green
    "bicycle":    (0.0,  0.75, 0.75),   # teal
    "unknown":    (0.5,  0.5,  0.5),   # grey
}

# BEV class colours (brighter for dark background)
BEV_CLASS_COLORS = {
    "car":        "#1E90FF",
    "truck":      "#FF8C00",
    "bus":        "#9400D3",
    "pedestrian": "#FF4500",
    "motorcycle": "#00CC66",
    "bicycle":    "#00BFBF",
    "unknown":    "#888888",
}


# ------------------------------------------------------------------ #
#  Rotation helper (matches dataset loader get_ann_info)
# ------------------------------------------------------------------ #
def rotate_boxes_90ccw(gt_boxes):
    """Apply 90° CCW rotation to match nuScenes convention.

    LOKI original: x=forward, y=lateral
    After rotation: x=right (-old_y), y=forward (old_x)
    """
    boxes = gt_boxes.copy()
    if len(boxes) == 0:
        return boxes
    old_x = boxes[:, 0].copy()
    old_y = boxes[:, 1].copy()
    boxes[:, 0] = -old_y   # new x = -old y (right)
    boxes[:, 1] = old_x    # new y = old x (forward)
    boxes[:, 6] += 0.5 * np.pi  # rotate yaw
    if boxes.shape[-1] >= 9:
        old_vx = boxes[:, 7].copy()
        old_vy = boxes[:, 8].copy()
        boxes[:, 7] = -old_vy
        boxes[:, 8] = old_vx
    return boxes


# ------------------------------------------------------------------ #
#  FOV filter (same logic as ObjectFOVFilterTrack / _in_fov)
# ------------------------------------------------------------------ #
def fov_mask(boxes_rotated, fov_deg=60.0):
    """Return boolean mask for boxes within the front camera FOV.

    In the rotated frame: +y = forward, +x = right.
    Keeps objects where y > 0 and |atan2(x, y)| <= fov_deg/2.
    """
    if len(boxes_rotated) == 0:
        return np.array([], dtype=bool)
    half_fov = np.deg2rad(fov_deg / 2.0)
    x = boxes_rotated[:, 0]
    y = boxes_rotated[:, 1]
    angles = np.abs(np.arctan2(x, y))
    return (y > 0) & (angles <= half_fov)


# ------------------------------------------------------------------ #
#  2D label parsing
# ------------------------------------------------------------------ #
def parse_label2d_raw(filepath):
    """Parse a LOKI label2d_XXXX.json file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    annotations = []
    for raw_class, objects in data.items():
        mapped = LOKI_CLASS_MAP.get(raw_class, None)
        if mapped is None:
            continue
        if not isinstance(objects, dict):
            continue
        for track_id, obj in objects.items():
            box = obj.get('box', None)
            if box is None:
                continue
            annotations.append(dict(
                label=raw_class,
                mapped=mapped,
                track_id=track_id,
                box=box,
                not_in_lidar=obj.get('not_in_lidar', False),
            ))
    return annotations


# ------------------------------------------------------------------ #
#  3D / BEV helpers
# ------------------------------------------------------------------ #
def corners_3d_bottom(box):
    """Compute the 4 bottom-face corners of a 3D box for BEV drawing."""
    x, y, z, dx, dy, dz, yaw = box[:7]
    hdx, hdy = dx / 2.0, dy / 2.0
    local = np.array([[-hdx, -hdy], [-hdx, hdy],
                       [hdx, hdy], [hdx, -hdy]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated = (R @ local.T).T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated


def box_corners_3d(box):
    """8 corners of a 3D box. box: [x,y,z,dx,dy,dz,yaw,...]. Returns (8,3)."""
    x, y, z, dx, dy, dz, yaw = box[:7]
    hdx, hdy, hdz = dx / 2.0, dy / 2.0, dz / 2.0
    local = np.array([
        [-hdx, -hdy, -hdz], [-hdx, -hdy,  hdz],
        [-hdx,  hdy, -hdz], [-hdx,  hdy,  hdz],
        [ hdx, -hdy, -hdz], [ hdx, -hdy,  hdz],
        [ hdx,  hdy, -hdz], [ hdx,  hdy,  hdz],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (R @ local.T).T + np.array([x, y, z], dtype=np.float64)


def project_box_to_2d(box, lidar2img, img_w, img_h):
    """Project 3D box to axis-aligned 2D bbox. Returns (x1,y1,x2,y2) or None."""
    corners = box_corners_3d(box)
    pts_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
    proj = (lidar2img @ pts_h.T).T
    depth = proj[:, 2]
    valid = depth > 0.1
    if not np.any(valid):
        return None
    u = proj[valid, 0] / depth[valid]
    v = proj[valid, 1] / depth[valid]
    x1 = float(np.clip(u.min(), 0, img_w - 1))
    y1 = float(np.clip(v.min(), 0, img_h - 1))
    x2 = float(np.clip(u.max(), 0, img_w - 1))
    y2 = float(np.clip(v.max(), 0, img_h - 1))
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return None
    return (x1, y1, x2, y2)


def parse_label3d_raw(filepath):
    """Parse a raw label3d file for cross-checking track IDs."""
    annotations = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return annotations
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        reader = csv.reader(io.StringIO(line))
        parts = next(reader)
        if len(parts) < 14:
            continue
        raw_class = parts[0].strip()
        if raw_class not in LOKI_CLASS_MAP:
            continue
        annotations.append(dict(
            label=raw_class,
            mapped=LOKI_CLASS_MAP[raw_class],
            track_id=parts[1].strip(),
        ))
    return annotations


# ------------------------------------------------------------------ #
#  Drawing helpers
# ------------------------------------------------------------------ #
def draw_2d_box_on_image(ax, box, color, label=None, linewidth=2.0):
    """Draw a 2D bounding box rectangle on a matplotlib axes."""
    top = float(box['top'])
    left = float(box['left'])
    h = float(box['height'])
    w = float(box['width'])
    rect = Rectangle((left, top), w, h,
                      linewidth=linewidth, edgecolor=color,
                      facecolor='none', alpha=0.85)
    ax.add_patch(rect)
    if label is not None:
        ax.text(left, top - 3, label, fontsize=6, color=color,
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='black', alpha=0.6),
                ha='left', va='bottom')


def draw_2d_rect_on_image(ax, x1, y1, x2, y2, color,
                          label=None, linewidth=2.0, linestyle='-', alpha=0.85):
    """Draw a 2D rectangle from projected corners."""
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                     linewidth=linewidth, edgecolor=color,
                     facecolor='none', linestyle=linestyle, alpha=alpha)
    ax.add_patch(rect)
    if label is not None:
        ax.text(x1, y1 - 3, label, fontsize=6, color=color,
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='black', alpha=0.6),
                ha='left', va='bottom')


def draw_bev_box(ax, box, color, label=None, linewidth=1.2):
    """Draw an oriented rectangle in BEV (top-down XY plane)."""
    bottom = corners_3d_bottom(box)
    poly = plt.Polygon(bottom, closed=True, fill=False,
                       edgecolor=color, linewidth=linewidth, alpha=0.9)
    ax.add_patch(poly)
    if label:
        cx, cy = box[0], box[1]
        ax.text(cx, cy - 1.0, label, fontsize=5, color=color,
                ha='center', va='top', alpha=0.7)


def draw_fov_cone(ax, fov_deg, max_range=55.0):
    """Draw the camera FOV cone on the BEV axes.

    In the rotated frame: +y = forward (up), +x = right.
    """
    half_fov = np.deg2rad(fov_deg / 2.0)
    # FOV boundary lines from origin
    # Right edge: +half_fov from +y
    rx = max_range * np.sin(half_fov)
    ry = max_range * np.cos(half_fov)
    # Left edge: -half_fov from +y
    lx = -rx

    # Draw the two boundary lines
    ax.plot([0, rx], [0, ry], '--', color='#00ff88', linewidth=1.5,
            alpha=0.8, zorder=2)
    ax.plot([0, lx], [0, ry], '--', color='#00ff88', linewidth=1.5,
            alpha=0.8, zorder=2)

    # Fill the FOV wedge with transparent green
    # Wedge angle: matplotlib Wedge uses degrees from +x axis CCW
    # +y direction = 90° from +x, so FOV spans [90-half_fov_deg, 90+half_fov_deg]
    half_deg = fov_deg / 2.0
    wedge = Wedge((0, 0), max_range, 90 - half_deg, 90 + half_deg,
                  facecolor='#00ff88', alpha=0.06, edgecolor='none', zorder=1)
    ax.add_patch(wedge)

    # Label
    ax.text(0, max_range + 2, f'{fov_deg}° FOV', fontsize=8,
            color='#00ff88', ha='center', va='bottom', alpha=0.9)


# ------------------------------------------------------------------ #
#  Main per-frame visualization
# ------------------------------------------------------------------ #
def visualize_single_frame(info, data_root, out_dir, frame_num,
                           pc_range=POINT_CLOUD_RANGE, fov_deg=60.0):
    """Visualize a single frame with range + FOV + camera-visible filtering.

    BEV is shown in the rotated lidar frame (x=right, y=forward)
    to match the training coordinate system.
    """
    # 1. Setup
    img_path = info['img_filename'].replace(
        '/mnt/storage/loki_data', data_root.rstrip('/'))
    if not os.path.exists(img_path):
        return False, 0, 0, 0, 0, 0

    img = np.array(Image.open(img_path))
    gt_boxes_all = info['gt_boxes']
    gt_names_all = np.asarray(info['gt_names'])
    gt_visible_all = np.asarray(
        info.get('gt_camera_visible', np.ones(len(gt_boxes_all), dtype=bool))
    ).astype(bool)
    valid_flag_all = np.asarray(
        info.get('valid_flag', np.ones(len(gt_boxes_all), dtype=bool))
    ).astype(bool)
    if len(valid_flag_all) != len(gt_boxes_all):
        valid_flag_all = np.ones(len(gt_boxes_all), dtype=bool)
    if len(gt_visible_all) != len(gt_boxes_all):
        gt_visible_all = np.ones(len(gt_boxes_all), dtype=bool)

    gt_boxes_orig = gt_boxes_all[valid_flag_all]
    gt_names = gt_names_all[valid_flag_all]
    gt_visible = gt_visible_all[valid_flag_all]

    scenario = info.get('scenario', '?')
    frame_idx = info.get('frame_idx', '?')
    fid_str = f"{frame_idx:04d}" if isinstance(frame_idx, int) else str(frame_idx)

    # Track IDs from raw label3d for 3D<->2D correspondence checks
    raw_label3d_path = os.path.join(data_root, scenario, f"label3d_{fid_str}.txt")
    track_ids = np.array([''] * len(gt_boxes_orig), dtype=object)
    if os.path.exists(raw_label3d_path):
        raw_anns_3d = parse_label3d_raw(raw_label3d_path)
        raw_track_ids = np.array([ann['track_id'] for ann in raw_anns_3d], dtype=object)
        if len(raw_track_ids) == len(gt_boxes_all):
            track_ids = raw_track_ids[valid_flag_all]
        else:
            n_align = min(len(raw_track_ids), len(track_ids))
            track_ids[:n_align] = raw_track_ids[:n_align]

    # 2. Apply 90° CCW rotation (same as dataset loader)
    gt_boxes_rot = rotate_boxes_90ccw(gt_boxes_orig)

    # 3. Range filter in rotated frame
    bev_range = [pc_range[0], pc_range[1], pc_range[3], pc_range[4]]
    n_total = len(gt_boxes_rot)

    if n_total > 0:
        gt_boxes_obj = LiDARInstance3DBoxes(
            gt_boxes_rot, box_dim=gt_boxes_rot.shape[-1])
        mask_range = gt_boxes_obj.in_range_bev(bev_range).numpy().astype(bool)
    else:
        mask_range = np.array([], dtype=bool)

    # 4. FOV filter in rotated frame
    mask_fov = fov_mask(gt_boxes_rot, fov_deg=fov_deg)

    # 5. Camera-visibility filter (from pkl)
    if n_total > 0:
        if len(gt_visible) != n_total:
            gt_visible = np.ones(n_total, dtype=bool)
        mask_visible = gt_visible.astype(bool)
    else:
        mask_visible = np.array([], dtype=bool)

    # 6. Combined masks by stage
    mask_after_fov = mask_range & mask_fov if n_total > 0 else np.array([], dtype=bool)
    mask_visibility_filtered = (
        mask_after_fov & ~mask_visible if n_total > 0 else np.array([], dtype=bool))
    mask_kept = (
        mask_after_fov & mask_visible if n_total > 0 else np.array([], dtype=bool))

    n_kept = int(mask_kept.sum()) if n_total > 0 else 0
    n_range_only = int((~mask_range).sum()) if n_total > 0 else 0
    n_fov_only = int((mask_range & ~mask_fov).sum()) if n_total > 0 else 0
    n_visibility_only = int(mask_visibility_filtered.sum()) if n_total > 0 else 0

    # 7. Load 2D labels and cross-check camera visibility flags by track ID
    label2d_path = os.path.join(data_root, scenario, f"label2d_{fid_str}.json")
    anns_2d = parse_label2d_raw(label2d_path) if os.path.exists(label2d_path) else []
    ids_2d = set(ann['track_id'] for ann in anns_2d)
    mask_visible_from_2d = np.array(
        [(tid in ids_2d) if tid else False for tid in track_ids], dtype=bool
    ) if n_total > 0 else np.array([], dtype=bool)
    mask_visibility_mismatch = (
        mask_visible_from_2d ^ mask_visible if n_total > 0 else np.array([], dtype=bool))
    n_visibility_mismatch = int(mask_visibility_mismatch.sum()) if n_total > 0 else 0

    # 8. Track-ID groups for camera panel statuses
    ids_kept = set(track_ids[mask_kept].tolist()) if n_total > 0 else set()
    ids_fov_filtered = set(track_ids[mask_range & ~mask_fov].tolist()) if n_total > 0 else set()
    ids_range_filtered = set(track_ids[~mask_range].tolist()) if n_total > 0 else set()
    ids_visibility_filtered = (
        set(track_ids[mask_visibility_filtered].tolist()) if n_total > 0 else set())
    ids_mismatch = set(track_ids[mask_visibility_mismatch].tolist()) if n_total > 0 else set()
    ids_all_3d = set([tid for tid in track_ids.tolist() if tid])
    ids_no2d_filtered = [tid for tid in track_ids[mask_visibility_filtered].tolist() if tid]

    # 9. Figure layout
    fig = plt.figure(figsize=(24, 10), facecolor='#1a1a2e')
    fig.suptitle(
        f"LOKI GT: {scenario} | Frame {frame_idx}  "
        f"({n_kept}/{n_total} final-kept, {n_visibility_only} no2D-filtered, "
        f"{n_fov_only} FOV-filtered, {n_range_only} range-filtered)",
        fontsize=14, color='white', y=0.96)

    # --- Panel 1: Camera View ---
    ax_cam = fig.add_axes([0.02, 0.05, 0.47, 0.85])
    ax_cam.imshow(img)
    ax_cam.set_title("Camera View (2D labels + projected no2D-filtered 3D boxes)",
                     color='white')
    ax_cam.axis('off')

    for ann in anns_2d:
        cls, tid = ann['mapped'], ann['track_id']
        color_cls = CLASS_COLORS.get(cls, CLASS_COLORS['unknown'])
        if tid not in ids_all_3d:
            # Skip 2D-only annotations (no corresponding 3D GT in this frame).
            continue

        if tid in ids_mismatch:
            draw_2d_box_on_image(ax_cam, ann['box'], (0.2, 1.0, 1.0),
                                 label=f"{cls} [VIS-MISMATCH]", linewidth=2.0)
        elif tid in ids_kept:
            draw_2d_box_on_image(ax_cam, ann['box'], color_cls,
                                 label=f"{cls} [3D KEPT]", linewidth=2.5)
        elif tid in ids_fov_filtered:
            draw_2d_box_on_image(ax_cam, ann['box'], (1.0, 0.5, 0.0),
                                 label=f"{cls} [3D FOV-FILTERED]", linewidth=1.6)
        elif tid in ids_range_filtered:
            draw_2d_box_on_image(ax_cam, ann['box'], (0.65, 0.35, 0.2),
                                 label=f"{cls} [3D RANGE-FILTERED]", linewidth=1.2)
        elif tid in ids_visibility_filtered:
            draw_2d_box_on_image(ax_cam, ann['box'], (1.0, 0.9, 0.0),
                                 label=f"{cls} [UNEXPECTED no2D]", linewidth=1.8)
        elif tid in ids_all_3d:
            draw_2d_box_on_image(ax_cam, ann['box'], (0.85, 0.25, 0.25),
                                 label=f"{cls} [3D FILTERED]", linewidth=1.0)

    # Overlay camera-visible filtered 3D boxes (no matching 2D ID) in yellow.
    img_h, img_w = img.shape[:2]
    lidar2img_old = np.asarray(info.get('lidar2img', np.eye(4)), dtype=np.float64)
    if lidar2img_old.ndim == 3:
        lidar2img_old = lidar2img_old[0]
    R_inv_4x4 = np.eye(4, dtype=np.float64)
    R_inv_4x4[:3, :3] = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    lidar2img_rot = lidar2img_old @ R_inv_4x4

    n_no2d_projected = 0
    for i in np.where(mask_visibility_filtered)[0]:
        box2d = project_box_to_2d(gt_boxes_rot[i], lidar2img_rot, img_w, img_h)
        if box2d is None:
            continue
        x1, y1, x2, y2 = box2d
        tid_suffix = str(track_ids[i])[-4:] if i < len(track_ids) and track_ids[i] else ''
        label = f"{str(gt_names[i])} [no2D]"
        if tid_suffix:
            label += f" {tid_suffix}"
        draw_2d_rect_on_image(
            ax_cam, x1, y1, x2, y2, color=(1.0, 0.9, 0.0),
            label=label, linewidth=1.8, linestyle='--', alpha=0.95)
        n_no2d_projected += 1

    no2d_preview = ', '.join(ids_no2d_filtered[:8]) if ids_no2d_filtered else '-'
    ax_cam.text(
        0.01, 0.99,
        f"3D no-2D filtered (in range+FOV): {n_visibility_only}\n"
        f"Projected in camera: {n_no2d_projected}\n"
        f"IDs: {no2d_preview}\n"
        f"vis-flag mismatch count: {n_visibility_mismatch}",
        transform=ax_cam.transAxes, ha='left', va='top', fontsize=8,
        color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.45,
                  edgecolor='#666666'))

    # --- Panel 2: BEV View (rotated frame: x=right, y=forward) ---
    ax_bev = fig.add_axes([0.52, 0.05, 0.45, 0.85])
    ax_bev.set_facecolor('#0d1117')
    ax_bev.set_title(
        "BEV (rotated frame: x=right, y=forward) + range/FOV/visibility filters",
        color='white')

    # Draw range boundary
    range_w = pc_range[3] - pc_range[0]
    range_h = pc_range[4] - pc_range[1]
    ax_bev.add_patch(plt.Rectangle(
        (pc_range[0], pc_range[1]), range_w, range_h,
        edgecolor='#00ff88', facecolor='#00ff8808', linewidth=1.0,
        linestyle=':', zorder=1))

    # Draw FOV cone
    draw_fov_cone(ax_bev, fov_deg, max_range=pc_range[4] + 5)

    # Draw ego vehicle marker at origin
    ax_bev.plot(0, 0, 's', color='white', markersize=6, zorder=5)
    ax_bev.text(0, -3, 'EGO', fontsize=7, color='white',
                ha='center', va='top', zorder=5)

    # Plot 3D boxes (in rotated frame)
    for i in range(n_total):
        box = gt_boxes_rot[i]
        name = str(gt_names[i])
        tid = track_ids[i] if i < len(track_ids) else ''

        if mask_kept[i]:
            color = BEV_CLASS_COLORS.get(name, BEV_CLASS_COLORS['unknown'])
            draw_bev_box(ax_bev, box, color, linewidth=2.0)
        elif mask_visibility_filtered[i]:
            # In range + FOV, but no matching 2D bbox => camera-visible filtered
            draw_bev_box(ax_bev, box, '#ffd700', linewidth=1.8)
            ax_bev.plot(box[0], box[1], 'x', color='#ffd700',
                        markersize=8, alpha=0.9)
            if tid:
                ax_bev.text(box[0], box[1] + 0.8, f"no2d:{tid[-4:]}",
                            fontsize=5, color='#ffd700',
                            ha='center', va='bottom', alpha=0.9)
        elif not mask_range[i]:
            # Outside range — dim red
            draw_bev_box(ax_bev, box, '#663333', linewidth=0.6)
            ax_bev.plot(box[0], box[1], 'x', color='#663333',
                        markersize=5, alpha=0.4)
        else:
            # In range but outside FOV — bright red with X
            draw_bev_box(ax_bev, box, '#ff4444', linewidth=1.0)
            ax_bev.plot(box[0], box[1], 'x', color='#ff4444',
                        markersize=8, alpha=0.7)

    # Axis labels and limits
    ax_bev.set_xlabel('x (right) [m]', color='#aaaaaa', fontsize=9)
    ax_bev.set_ylabel('y (forward) [m]', color='#aaaaaa', fontsize=9)
    ax_bev.set_xlim(pc_range[0] - 10, pc_range[3] + 10)
    ax_bev.set_ylim(-10, pc_range[4] + 10)
    ax_bev.set_aspect('equal')
    ax_bev.tick_params(colors='#666666')

    ax_bev.text(
        0.02, 0.02,
        f"3D(valid): {n_total}\n"
        f"in range: {int(mask_range.sum()) if n_total > 0 else 0}\n"
        f"in range+FOV: {int(mask_after_fov.sum()) if n_total > 0 else 0}\n"
        f"filtered by no2D: {n_visibility_only}\n"
        f"final kept: {n_kept}\n"
        f"vis-flag mismatch: {n_visibility_mismatch}",
        transform=ax_bev.transAxes, ha='left', va='bottom',
        fontsize=8, color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.45,
                  edgecolor='#666666'))

    # Legend
    legend_items = [
        mpatches.Patch(facecolor='none', edgecolor='#1E90FF',
                       linewidth=2, label='Final KEPT'),
        mpatches.Patch(facecolor='none', edgecolor='#ffd700',
                       linewidth=1.8, label='Camera-visible filtered (no 2D ID)'),
        mpatches.Patch(facecolor='none', edgecolor='#ff4444',
                       linewidth=1, label='FOV-filtered (in range, outside FOV)'),
        mpatches.Patch(facecolor='none', edgecolor='#663333',
                       linewidth=0.6, label='Range-filtered (outside BEV range)'),
    ]
    ax_bev.legend(handles=legend_items, loc='upper right', fontsize=7,
                  facecolor='#1a1a2e', edgecolor='#444444',
                  labelcolor='white')

    # Save
    out_path = os.path.join(
        out_dir, f"frame_{frame_num:04d}_{scenario}_f{frame_idx}.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True, n_total, n_kept, n_fov_only, n_visibility_only, n_visibility_mismatch


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description='Visualize LOKI ground truth with range + FOV + camera-visible filtering')
    parser.add_argument('--pkl', type=str,
                        default='data/infos/loki_infos_val.pkl',
                        help='Path to loki_infos pkl file')
    parser.add_argument('--data-root', type=str,
                        default='/mnt/storage/loki_data',
                        help='Root path to LOKI images')
    parser.add_argument('--out-dir', type=str,
                        default='viz_loki_gt_fov',
                        help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of frames to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for frame sampling')
    parser.add_argument('--fov-deg', type=float, default=60.0,
                        help='Camera horizontal FOV in degrees (default: 60)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"FOV filter: {args.fov_deg}°  (half = ±{args.fov_deg/2:.1f}°)")
    print(f"Range filter: {POINT_CLOUD_RANGE}")
    print("Camera-visible filter: keep 3D boxes with matching 2D track ID")

    # Load pkl
    print(f"Loading {args.pkl} ...")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    print(f"  Loaded {len(infos)} frame infos")

    # Sample frames
    print(f"\n=== Visualizing {args.num_samples} frames ===")
    np.random.seed(args.seed)
    n_obj_counts = [len(info['gt_boxes']) for info in infos]
    num_samples = min(args.num_samples, len(infos))
    num_top = min(num_samples // 2, len(infos))
    top_indices = np.argsort(n_obj_counts)[-num_top:] if num_top > 0 else np.array([], dtype=np.int64)

    num_rand = num_samples - len(top_indices)
    rand_pool = np.setdiff1d(np.arange(len(infos)), top_indices, assume_unique=False)
    if num_rand > 0 and len(rand_pool) > 0:
        num_rand = min(num_rand, len(rand_pool))
        rand_indices = np.random.choice(rand_pool, size=num_rand, replace=False)
    else:
        rand_indices = np.array([], dtype=np.int64)

    sample_indices = sorted(set(top_indices.tolist() + rand_indices.tolist()))[:num_samples]

    total_all, total_kept = 0, 0
    total_fov_filtered, total_visibility_filtered = 0, 0
    total_visibility_mismatch = 0
    for frame_num, idx in enumerate(sample_indices):
        info = infos[idx]
        success, n_total, n_kept, n_fov, n_vis, n_mismatch = visualize_single_frame(
            info, args.data_root, args.out_dir, frame_num,
            fov_deg=args.fov_deg)
        total_all += n_total
        total_kept += n_kept
        total_fov_filtered += n_fov
        total_visibility_filtered += n_vis
        total_visibility_mismatch += n_mismatch
        status = "OK" if success else "SKIP"
        print(f"  [{status}] {frame_num+1}/{len(sample_indices)}: "
              f"{info.get('scenario','?')} frame {info.get('frame_idx','?')} "
              f"— {n_kept}/{n_total} kept, {n_vis} no2D-filtered, "
              f"{n_fov} FOV-filtered, {n_mismatch} vis-mismatch")

    print(f"\n=== Summary ===")
    print(f"  Total objects:      {total_all}")
    print(f"  Final kept:         {total_kept}")
    print(f"  no2D-filtered:      {total_visibility_filtered}")
    print(f"  FOV-filtered:       {total_fov_filtered}")
    print(f"  Range-filtered:     {total_all - total_kept - total_fov_filtered - total_visibility_filtered}")
    print(f"  vis-flag mismatch:  {total_visibility_mismatch}")
    if total_all > 0:
        print(f"  no2D removed:       {total_visibility_filtered/total_all*100:.1f}% of all objects")
        print(f"  FOV removed:        {total_fov_filtered/total_all*100:.1f}% of all objects")
    print(f"\nDone! Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
