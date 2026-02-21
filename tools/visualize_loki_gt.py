#!/usr/bin/env python3
"""
Visualize LOKI ground truth data to verify dataset loader correctness.

Produces for each sampled frame:
  1. Camera view  – front image with 2D bounding boxes from label2d files
                    showing KEPT vs FILTERED status
  2. BEV view     – bird's-eye-view with oriented 3D boxes in the rotated
                    lidar frame (x=right, y=forward), range boundary, and
                    60° FOV cone overlay

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
        if len(parts) < 10:
            continue
        annotations.append(dict(
            label=parts[0].strip(),
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


def draw_bev_box(ax, box, color, label=None, linewidth=1.2):
    """Draw an oriented rectangle in BEV (top-down XY plane)."""
    bottom = corners_3d_bottom(box)
    poly = plt.Polygon(bottom, closed=True, fill=False,
                       edgecolor=color, linewidth=linewidth, alpha=0.9)
    ax.add_patch(poly)
    cx, cy = box[0], box[1]
    front_mid = (bottom[2] + bottom[3]) / 2
    ax.annotate('', xy=front_mid, xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0))
    if label:
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
    """Visualize a single frame with range + FOV filtering.

    BEV is shown in the rotated lidar frame (x=right, y=forward)
    to match the training coordinate system.
    """
    # 1. Setup
    img_path = info['img_filename'].replace(
        '/mnt/storage/loki_data', data_root.rstrip('/'))
    if not os.path.exists(img_path):
        return False, 0, 0, 0

    img = np.array(Image.open(img_path))
    gt_boxes_orig = info['gt_boxes']
    gt_names = info['gt_names']
    scenario = info.get('scenario', '?')
    frame_idx = info.get('frame_idx', '?')
    fid_str = f"{frame_idx:04d}" if isinstance(frame_idx, int) else str(frame_idx)

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

    # 5. Combined mask
    mask_kept = mask_range & mask_fov if n_total > 0 else np.array([], dtype=bool)

    n_kept = int(mask_kept.sum()) if n_total > 0 else 0
    n_range_only = int((~mask_range).sum()) if n_total > 0 else 0
    n_fov_only = int((mask_range & ~mask_fov).sum()) if n_total > 0 else 0

    # 6. Match track IDs (for camera view 2D→3D correspondence)
    kept_ids = set()
    raw_label3d_path = os.path.join(data_root, scenario, f"label3d_{fid_str}.txt")
    if os.path.exists(raw_label3d_path) and n_total > 0:
        raw_anns_3d = parse_label3d_raw(raw_label3d_path)
        for i in range(min(len(mask_kept), len(raw_anns_3d))):
            if mask_kept[i]:
                kept_ids.add(raw_anns_3d[i]['track_id'])

    # 7. Figure layout
    fig = plt.figure(figsize=(24, 10), facecolor='#1a1a2e')
    fig.suptitle(
        f"LOKI GT: {scenario} | Frame {frame_idx}  "
        f"({n_kept}/{n_total} kept, {n_fov_only} FOV-filtered, "
        f"{n_range_only} range-filtered)",
        fontsize=14, color='white', y=0.96)

    # --- Panel 1: Camera View ---
    ax_cam = fig.add_axes([0.02, 0.05, 0.47, 0.85])
    ax_cam.imshow(img)
    ax_cam.set_title("Camera View (2D labels + filter status)", color='white')
    ax_cam.axis('off')

    label2d_path = os.path.join(data_root, scenario, f"label2d_{fid_str}.json")
    if os.path.exists(label2d_path):
        anns_2d = parse_label2d_raw(label2d_path)
        for ann in anns_2d:
            cls, tid = ann['mapped'], ann['track_id']
            is_kept = tid in kept_ids
            color = CLASS_COLORS.get(cls, CLASS_COLORS['unknown'])

            if is_kept:
                draw_2d_box_on_image(ax_cam, ann['box'], color,
                                     label=f"{cls} [KEPT]", linewidth=2.5)
            else:
                draw_2d_box_on_image(ax_cam, ann['box'], (0.8, 0.2, 0.2),
                                     label=f"{cls} [FILTERED]", linewidth=1.0)
                b = ann['box']
                cx = float(b['left']) + float(b['width']) / 2
                cy = float(b['top']) + float(b['height']) / 2
                ax_cam.plot(cx, cy, 'x', color='red', markersize=10, alpha=0.6)

    # --- Panel 2: BEV View (rotated frame: x=right, y=forward) ---
    ax_bev = fig.add_axes([0.52, 0.05, 0.45, 0.85])
    ax_bev.set_facecolor('#0d1117')
    ax_bev.set_title("BEV (rotated frame: x=right, y=forward) + FOV cone",
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

        if mask_kept[i]:
            color = BEV_CLASS_COLORS.get(name, BEV_CLASS_COLORS['unknown'])
            draw_bev_box(ax_bev, box, color, linewidth=2.0)
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

    # Legend
    legend_items = [
        mpatches.Patch(facecolor='none', edgecolor='#1E90FF',
                       linewidth=2, label='KEPT (in range + FOV)'),
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
    return True, n_total, n_kept, n_fov_only


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description='Visualize LOKI ground truth with range + FOV filtering')
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
    top_indices = np.argsort(n_obj_counts)[-args.num_samples // 2:]
    rand_indices = np.random.choice(
        len(infos), size=args.num_samples - len(top_indices), replace=False)
    sample_indices = sorted(
        set(top_indices.tolist() + rand_indices.tolist()))[:args.num_samples]

    total_all, total_kept, total_fov_filtered = 0, 0, 0
    for frame_num, idx in enumerate(sample_indices):
        info = infos[idx]
        success, n_total, n_kept, n_fov = visualize_single_frame(
            info, args.data_root, args.out_dir, frame_num,
            fov_deg=args.fov_deg)
        total_all += n_total
        total_kept += n_kept
        total_fov_filtered += n_fov
        status = "OK" if success else "SKIP"
        print(f"  [{status}] {frame_num+1}/{len(sample_indices)}: "
              f"{info.get('scenario','?')} frame {info.get('frame_idx','?')} "
              f"— {n_kept}/{n_total} kept, {n_fov} FOV-filtered")

    print(f"\n=== Summary ===")
    print(f"  Total objects:      {total_all}")
    print(f"  Kept (in FOV):      {total_kept}")
    print(f"  FOV-filtered:       {total_fov_filtered}")
    print(f"  Range-filtered:     {total_all - total_kept - total_fov_filtered}")
    if total_all > 0:
        print(f"  FOV filter removed: {total_fov_filtered/total_all*100:.1f}% "
              f"of all objects")
    print(f"\nDone! Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
