#!/usr/bin/env python3
"""
Visualize LOKI ground truth data to verify dataset loader correctness.

Produces for each sampled frame:
  1. Camera view  – front image with 2D bounding boxes from label2d files
                    + projected 3D box centers as dots (to diagnose z offset)
  2. BEV view     – bird's-eye-view with oriented 3D boxes
  3. Cross-check  – compares pkl info against raw label files

Also produces aggregate statistics:
  - Class distribution, position/dimension histograms
  - Fraction of objects inside vs outside the BEV range
  - 2D vs projected-3D alignment check

Usage:
    python tools/visualize_loki_gt.py \
        --pkl data/infos/loki_infos_train.pkl \
        --data-root /mnt/zihan/loki_data \
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
from matplotlib.patches import Rectangle
from PIL import Image
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
import torch # mmdet3d usually expects tensors or handles the conversion

# ------------------------------------------------------------------ #
#  Constants (must match create_loki_infos.py)
# ------------------------------------------------------------------ #
POINT_CLOUD_RANGE_FULL = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# Config-filtered range: y_min=0 to keep only forward-facing objects
POINT_CLOUD_RANGE_FILTERED = [-51.2, 0, -5.0, 51.2, 51.2, 3.0]
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
#  2D label parsing  (the ground-truth we actually trust)
# ------------------------------------------------------------------ #
def parse_label2d_raw(filepath):
    """
    Parse a LOKI label2d_XXXX.json file.

    Returns:
        list of dicts, each with:
            label:     str  (raw LOKI class, e.g. 'Car')
            mapped:    str  (mapped class, e.g. 'car')
            track_id:  str  (UUID)
            box:       dict {top, left, height, width}  in original image pixels
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    annotations = []
    for raw_class, objects in data.items():
        mapped = LOKI_CLASS_MAP.get(raw_class, None)
        if mapped is None:
            continue  # skip Traffic_Sign, Traffic_Light, etc.
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
                box=box,  # {top, left, height, width}
                not_in_lidar=obj.get('not_in_lidar', False),
            ))
    return annotations


# ------------------------------------------------------------------ #
#  3D helpers (for BEV and optional projection overlay)
# ------------------------------------------------------------------ #
def corners_3d_bottom(box):
    """
    Compute the 4 bottom-face corners of a 3D box (for BEV drawing).

    Args:
        box: [x, y, z, dx, dy, dz, yaw, ...]

    Returns:
        (4, 2) array of XY corner positions in LiDAR frame.
    """
    x, y, z, dx, dy, dz, yaw = box[:7]
    hdx, hdy = dx / 2.0, dy / 2.0

    # Bottom face corners in local frame
    local = np.array([
        [-hdx, -hdy],
        [-hdx,  hdy],
        [ hdx,  hdy],
        [ hdx, -hdy],
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    rotated = (R @ local.T).T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated


def project_center_to_image(center_3d, lidar2img):
    """
    Project a single 3D point to image pixel coordinates.

    Args:
        center_3d: (3,) point in LiDAR frame
        lidar2img: (4, 4) projection matrix

    Returns:
        (u, v): pixel coordinates, or None if behind camera
    """
    pt = np.array([center_3d[0], center_3d[1], center_3d[2], 1.0])
    proj = lidar2img @ pt
    if proj[2] <= 0.5:
        return None  # behind camera
    u = proj[0] / proj[2]
    v = proj[1] / proj[2]
    return (u, v)


# ------------------------------------------------------------------ #
#  Drawing helpers
# ------------------------------------------------------------------ #
def draw_2d_box_on_image(ax, box, color, label=None, linewidth=2.0):
    """
    Draw a 2D bounding box rectangle on a matplotlib axes.

    Args:
        box: dict with {top, left, height, width}
    """
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

    # Draw heading arrow (from center to front midpoint)
    cx, cy = box[0], box[1]
    front_mid = (bottom[2] + bottom[3]) / 2  # midpoint of front edge
    ax.annotate('', xy=front_mid, xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

    if label:
        ax.text(cx, cy - 1.0, label, fontsize=5, color=color,
                ha='center', va='top', alpha=0.7)


# ------------------------------------------------------------------ #
#  Raw data cross-check helpers
# ------------------------------------------------------------------ #
def parse_label3d_raw(filepath):
    """Parse a raw label3d file for cross-checking."""
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
            pos=np.array([float(parts[3]), float(parts[4]), float(parts[5])]),
            dim=np.array([float(parts[6]), float(parts[7]), float(parts[8])]),
            yaw=float(parts[9]),
        ))
    return annotations


def parse_odom_raw(filepath):
    """Parse a raw odometry file."""
    with open(filepath, 'r') as f:
        vals = [float(v) for v in f.readline().strip().split(',')]
    return np.array(vals, dtype=np.float64)


def visualize_single_frame(info, data_root, out_dir, frame_num,
                           pc_range=POINT_CLOUD_RANGE_FULL,
                           filter_range=None):
    """
    Restores 2D bounding boxes while using mmdet3d for filtering logic.
    """
    # 1. Setup paths and load data
    img_path = info['img_filename'].replace('/mnt/storage/loki_data', data_root.rstrip('/'))
    if not os.path.exists(img_path):
        return False

    img = np.array(Image.open(img_path))
    gt_boxes = info['gt_boxes'] 
    gt_names = info['gt_names']
    lidar2img = info['lidar2img']
    
    scenario = info.get('scenario', '?')
    frame_idx = info.get('frame_idx', '?')
    fid_str = f"{frame_idx:04d}" if isinstance(frame_idx, int) else str(frame_idx)

    # 2. Vectorized Range Filtering via mmdet3d (The Ground Truth)
    check_range = filter_range if filter_range is not None else pc_range
    bev_range = [check_range[0], check_range[1], check_range[3], check_range[4]]
    
    kept_ids = set()
    mask_bev = np.array([], dtype=bool)
    
    if len(gt_boxes) > 0:
        gt_boxes_obj = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1])
        mask_bev = gt_boxes_obj.in_range_bev(bev_range).numpy()
        
        # Build a lookup of IDs that are "KEPT"
        # We assume the info['gt_ids'] (if available) or the raw label3d track IDs
        # are what we match against label2d.
        raw_label3d_path = os.path.join(data_root, scenario, f"label3d_{fid_str}.txt")
        if os.path.exists(raw_label3d_path):
            raw_anns_3d = parse_label3d_raw(raw_label3d_path)
            # Match index-to-index since pkl boxes usually follow raw file order
            for i in range(min(len(mask_bev), len(raw_anns_3d))):
                if mask_bev[i]:
                    kept_ids.add(raw_anns_3d[i]['track_id'])

    # 3. Figure Layout
    fig = plt.figure(figsize=(24, 10), facecolor='#1a1a2e')
    fig.suptitle(f"LOKI GT: {scenario} | Frame {frame_idx}", fontsize=16, color='white', y=0.95)

    # --- Panel 1: Camera View ---
    ax_cam = fig.add_axes([0.02, 0.05, 0.47, 0.85]) 
    ax_cam.imshow(img)
    ax_cam.set_title("Camera View (2D Rectangles + 3D Filter Status)", color='white')
    ax_cam.axis('off')

    label2d_path = os.path.join(data_root, scenario, f"label2d_{fid_str}.json")
    if os.path.exists(label2d_path):
        anns_2d = parse_label2d_raw(label2d_path)
        for ann in anns_2d:
            cls, tid = ann['mapped'], ann['track_id']
            
            # DETERMINING STATUS:
            # If the track ID was found in our "Kept" set from the 3D mask...
            is_kept = tid in kept_ids
            color = CLASS_COLORS.get(cls, CLASS_COLORS['unknown'])
            
            if is_kept:
                draw_2d_box_on_image(ax_cam, ann['box'], color, label=f"{cls} [KEPT]", linewidth=2.5)
            else:
                # Still draw the box, but make it red/faded to show it's filtered
                draw_2d_box_on_image(ax_cam, ann['box'], (0.8, 0.2, 0.2), label=f"{cls} [FILTERED]", linewidth=1.0)
                # Add a red X in the middle for clarity
                b = ann['box']
                cx = float(b['left']) + float(b['width']) / 2
                cy = float(b['top']) + float(b['height']) / 2
                ax_cam.plot(cx, cy, 'x', color='red', markersize=10, alpha=0.6)

    # --- Panel 2: BEV View ---
    ax_bev = fig.add_axes([0.52, 0.05, 0.45, 0.85])
    ax_bev.set_facecolor('#0d1117')
    ax_bev.set_title("BEV Filter (mmdet3d)", color='white')

    # Draw Range Boundary
    ax_bev.add_patch(plt.Rectangle((pc_range[0], pc_range[1]), pc_range[3]-pc_range[0], 
                                   pc_range[4]-pc_range[1], edgecolor='#444444', fill=False, ls=':'))
    if filter_range is not None:
        ax_bev.add_patch(plt.Rectangle((filter_range[0], filter_range[1]), filter_range[3]-filter_range[0], 
                                       filter_range[4]-filter_range[1], edgecolor='#00ff88', facecolor='#00ff8808', zorder=1))

    # Plot 3D Boxes
    for i in range(len(gt_boxes)):
        box, name = gt_boxes[i], str(gt_names[i])
        is_kept = mask_bev[i]
        color = BEV_CLASS_COLORS.get(name, BEV_CLASS_COLORS['unknown']) if is_kept else '#ff4444'
        draw_bev_box(ax_bev, box, color, linewidth=2.0 if is_kept else 0.8)
        if not is_kept:
            ax_bev.plot(box[0], box[1], 'x', color='red', alpha=0.5)

    ax_bev.set_xlim(check_range[0]-10, check_range[3]+10)
    ax_bev.set_ylim(check_range[1]-10, check_range[4]+10)
    ax_bev.set_aspect('equal')

    # Save
    out_path = os.path.join(out_dir, f"frame_{frame_num:04d}_{scenario}_f{frame_idx}.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ------------------------------------------------------------------ #
#  Aggregate statistics
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Projection sanity check: 2D GT boxes vs projected 3D centers
# ------------------------------------------------------------------ #
def plot_projection_sanity(infos, data_root, out_dir):
    """
    For a handful of frames, show the image with:
      - 2D ground truth boxes (rectangles from label2d)
      - Projected 3D box centers (dots)
    This makes it easy to see if the 3D→2D projection is off.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 10), facecolor='#1a1a2e')
    fig.suptitle("2D GT Boxes vs Projected 3D Centers — Alignment Check",
                 fontsize=14, color='white')

    # Pick 6 frames with many objects
    scored = [(i, len(info['gt_boxes'])) for i, info in enumerate(infos)]
    scored.sort(key=lambda x: -x[1])
    selected = [scored[j][0] for j in range(min(6, len(scored)))]

    for ax_idx, info_idx in enumerate(selected):
        ax = axes.flat[ax_idx]
        info = infos[info_idx]
        img_path = info['img_filename'].replace(
            '/mnt/storage/loki_data', data_root.rstrip('/'))
        if not os.path.exists(img_path):
            ax.text(0.5, 0.5, "Image not found", transform=ax.transAxes,
                    ha='center', color='red')
            continue

        img = np.array(Image.open(img_path))
        ax.imshow(img)

        scenario = info.get('scenario', '?')
        frame_idx = info.get('frame_idx', '?')
        fid_str = f"{frame_idx:04d}" if isinstance(frame_idx, int) else str(frame_idx)

        # Load 2D labels
        label2d_path = os.path.join(data_root, scenario, f"label2d_{fid_str}.json")
        if os.path.exists(label2d_path):
            anns_2d = parse_label2d_raw(label2d_path)
            for ann in anns_2d:
                cls = ann['mapped']
                color = CLASS_COLORS.get(cls, CLASS_COLORS['unknown'])
                draw_2d_box_on_image(ax, ann['box'], color, linewidth=1.5)

        # Overlay projected 3D centers
        gt_boxes = info['gt_boxes']
        gt_names = info['gt_names']
        lidar2img = info['lidar2img']

        for i in range(len(gt_boxes)):
            center_3d = gt_boxes[i, :3]
            uv = project_center_to_image(center_3d, lidar2img)
            if uv is not None:
                name = str(gt_names[i])
                color = CLASS_COLORS.get(name, CLASS_COLORS['unknown'])
                dist = np.sqrt(gt_boxes[i, 0]**2 + gt_boxes[i, 1]**2)
                ax.plot(uv[0], uv[1], 'x', color='lime',
                        markersize=max(3, 8 - dist/15),
                        markeredgewidth=1.5, alpha=0.8)

        ax.set_title(f"{scenario} f{frame_idx} "
                     f"({len(gt_boxes)} 3D obj)", color='white', fontsize=9)
        ax.axis('off')

    # Hide unused axes
    for ax_idx in range(len(selected), 6):
        axes.flat[ax_idx].axis('off')

    # Add legend
    fig.text(0.5, 0.02,
             "Colored rectangles = 2D GT boxes (label2d)  |  "
             "Lime × = Projected 3D box centers (using estimated lidar2img)",
             ha='center', fontsize=10, color='#cccccc')

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = os.path.join(out_dir, "projection_sanity.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved projection sanity check to {out_path}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description='Visualize LOKI ground truth for UniAD')
    parser.add_argument('--pkl', type=str,
                        default='data/infos/loki_infos_train.pkl',
                        help='Path to loki_infos pkl file')
    parser.add_argument('--data-root', type=str,
                        default='/mnt/zihan/loki_data',
                        help='Root path to LOKI images (read-only)')
    parser.add_argument('--out-dir', type=str,
                        default='viz_loki_gt',
                        help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of frames to visualize individually')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for frame sampling')
    parser.add_argument('--skip-frames', action='store_true',
                        help='Skip individual frame visualizations')
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=None,
                        metavar=('X_MIN', 'Y_MIN', 'Z_MIN',
                                 'X_MAX', 'Y_MAX', 'Z_MAX'),
                        help='Point cloud range filter [x_min y_min z_min '
                             'x_max y_max z_max]. Objects outside this range '
                             'are shown as filtered. '
                             'Default: [-51.2, 0, -5.0, 51.2, 51.2, 3.0]')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Parse filter range
    filter_range = args.pc_range
    if filter_range is None:
        # Default to config-based filter range
        filter_range = POINT_CLOUD_RANGE_FILTERED
    print(f"Filter range: {filter_range}")

    # Load pkl
    print(f"Loading {args.pkl} ...")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    metadata = data['metadata']
    print(f"  Loaded {len(infos)} frame infos")
    print(f"  Metadata classes: {metadata.get('classes', '?')}")


    # ---- Projection sanity check ---- #
    print("\n=== Generating projection sanity check (2D GT vs projected 3D) ===")
    plot_projection_sanity(infos, args.data_root, args.out_dir)

    # ---- Individual frames ---- #
    if not args.skip_frames:
        print(f"\n=== Visualizing {args.num_samples} individual frames ===")
        np.random.seed(args.seed)

        # Sample a mix: some with many objects, some random
        n_obj_counts = [len(info['gt_boxes']) for info in infos]
        top_indices = np.argsort(n_obj_counts)[-args.num_samples // 2:]
        rand_indices = np.random.choice(len(infos),
                                        size=args.num_samples - len(top_indices),
                                        replace=False)
        sample_indices = list(set(top_indices.tolist() + rand_indices.tolist()))
        sample_indices = sorted(sample_indices)[:args.num_samples]

        for frame_num, idx in enumerate(sample_indices):
            info = infos[idx]
            success = visualize_single_frame(
                info, args.data_root, args.out_dir, frame_num,
                filter_range=filter_range)
            status = "OK" if success else "SKIP"
            print(f"  [{status}] {frame_num+1}/{len(sample_indices)}: "
                  f"{info['scenario']} frame {info['frame_idx']} "
                  f"({len(info['gt_boxes'])} 3D objects)")

    print(f"\nDone! Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
