#!/usr/bin/env python3
"""
Visualize the LiDAR-points GT filter.

Renders, for each chosen frame, two panels:
  1. Camera image with annotated 2D bboxes from label2d_*.json, color-coded
     by the per-box num_lidar_pts tier (red = filtered, yellow = borderline,
     green = healthy). Each box is annotated with its count.
  2. BEV with the actual LiDAR sweep (pc_*.ply) scattered as points, plus
     3D box outlines in the same color tiers, the 60° FOV cone, and the ego
     marker. BEV is in the rotated-lidar frame (x=right, y=forward) to match
     the model's operating frame.

The tool prefers frames that contain at least one in-FOV agent that would be
filtered (num_lidar_pts < min_lidar_pts) so that the visual difference is
clear. Override with --tokens to inspect specific frames.

Usage:
    python tools/visualize_lidar_filter.py \
        --pkl data/infos/loki_infos_val.pkl \
        --data-root /mnt/storage/loki_data \
        --out-dir viz_lidar_filter \
        --num-samples 12
"""

import argparse
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from PIL import Image
from plyfile import PlyData


POINT_CLOUD_RANGE = [-51.2, 0, -5.0, 51.2, 51.2, 3.0]
ORIG_IMG_SIZE = (1920, 1208)
HALF_FOV_RAD = np.deg2rad(30.0)

LOKI_CLASS_MAP = {
    "Car": "car",
    "Truck": "truck",
    "Van": "car",
    "Bus": "bus",
    "Pedestrian": "pedestrian",
    "Motorcyclist": "motorcycle",
    "Bicyclist": "bicycle",
}

# Filter-tier colors: shared between camera and BEV.
TIER_FILTERED = "#ff3b3b"   # would be dropped (num_lidar_pts < min)
TIER_BORDER = "#ffd84a"     # kept but on the edge
TIER_HEALTHY = "#5cff7d"    # plenty of LiDAR support


# --------------------------------------------------------------------- #
#  IO helpers
# --------------------------------------------------------------------- #
def load_pc_ply(path):
    """Load LOKI pc_*.ply → (N, 3) float32, lidar frame (x=fwd, y=lat)."""
    ply = PlyData.read(path)
    v = ply["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def load_label2d(path):
    """Read label2d_*.json → {track_id: dict(box, mapped_class, raw_class)}."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for raw_cls, objs in data.items():
        if not isinstance(objs, dict):
            continue
        mapped = LOKI_CLASS_MAP.get(raw_cls, None)
        for tid, obj in objs.items():
            box = obj.get("box", None)
            if box is None:
                continue
            out[tid] = dict(
                raw_class=raw_cls,
                mapped_class=mapped,
                box=box,
            )
    return out


# --------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------- #
def rotate_xy_90ccw(xy):
    """Apply 90° CCW so that (x_fwd, y_lat) → (x_right, y_fwd).

    Same convention as get_ann_info / rotate_boxes_90ccw in
    visualize_loki_gt.py: new_x = -old_y, new_y = old_x.
    """
    out = xy.copy()
    out[..., 0] = -xy[..., 1]
    out[..., 1] = xy[..., 0]
    return out


def rotate_box_inplace(box):
    """Rotate a single 9-DOF box [x,y,z,dx,dy,dz,yaw,vx,vy] 90° CCW in-place."""
    b = box.copy()
    ox, oy = b[0], b[1]
    b[0] = -oy
    b[1] = ox
    b[6] = b[6] + 0.5 * np.pi
    if b.shape[0] >= 9:
        ovx, ovy = b[7], b[8]
        b[7] = -ovy
        b[8] = ovx
    return b


def bottom_corners_xy(rotated_box):
    """4 bottom-face corners of an oriented BEV box in rotated frame."""
    x, y, z, dx, dy, dz, yaw = rotated_box[:7]
    hdx, hdy = dx * 0.5, dy * 0.5
    local = np.array([[-hdx, -hdy], [-hdx, hdy], [hdx, hdy], [hdx, -hdy]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rot = (R @ local.T).T
    rot[:, 0] += x
    rot[:, 1] += y
    return rot


def in_fov_rotated(xy):
    """In rotated frame: y > 0 and |atan2(x, y)| <= 30°."""
    x, y = float(xy[0]), float(xy[1])
    return y > 0 and abs(np.arctan2(x, y)) <= HALF_FOV_RAD


def in_pc_range_rotated(xy, rng=POINT_CLOUD_RANGE):
    x, y = float(xy[0]), float(xy[1])
    return rng[0] <= x <= rng[3] and rng[1] <= y <= rng[4]


# --------------------------------------------------------------------- #
#  Tier classification
# --------------------------------------------------------------------- #
def tier_of(count, min_pts):
    """Map a per-box LiDAR count to a color tier."""
    if count < min_pts:
        return ("filtered", TIER_FILTERED)
    if count < max(min_pts * 5, 5):  # borderline = up to 4 (or 4*min)
        return ("borderline", TIER_BORDER)
    return ("healthy", TIER_HEALTHY)


# --------------------------------------------------------------------- #
#  Frame selection
# --------------------------------------------------------------------- #
def score_frame(info, min_pts):
    """Higher score = more useful frame to visualize.

    Reward frames with in-FOV, in-range, camera-visible agents that are
    nonetheless filtered by the LiDAR threshold — those are the cases where
    the new filter visibly differs from the existing camera filter.
    """
    counts = info.get("num_lidar_pts", None)
    boxes = info.get("gt_boxes", None)
    cam_vis = info.get("gt_camera_visible", None)
    if counts is None or boxes is None or len(boxes) == 0:
        return -1
    score = 0
    for i in range(len(boxes)):
        rotated_xy = (-float(boxes[i, 1]), float(boxes[i, 0]))
        if not in_fov_rotated(rotated_xy):
            continue
        if not in_pc_range_rotated(rotated_xy):
            continue
        is_cam = bool(cam_vis[i]) if cam_vis is not None else True
        if not is_cam:
            continue
        if int(counts[i]) < min_pts:
            score += 5  # the showcase case: in FOV, has 2D bbox, 0 LiDAR pts
        elif int(counts[i]) < max(min_pts * 5, 5):
            score += 1  # borderline-low count
    return score


# --------------------------------------------------------------------- #
#  Drawing
# --------------------------------------------------------------------- #
def draw_camera_panel(ax, image, label2d_dict, infos_per_track, min_pts):
    """Draw camera image with 2D bboxes from label2d, colored by LiDAR tier.

    infos_per_track: dict[track_id] = (num_lidar_pts, in_fov, in_range,
                                        mapped_class, gt_camera_visible)
    Tracks present in label2d but absent from the 3D pkl are drawn grey.
    """
    ax.imshow(image)
    ax.set_xlim(0, ORIG_IMG_SIZE[0])
    ax.set_ylim(ORIG_IMG_SIZE[1], 0)
    ax.set_xticks([])
    ax.set_yticks([])

    for tid, ann in label2d_dict.items():
        b = ann["box"]
        x1, y1 = b["left"], b["top"]
        w, h = b["width"], b["height"]
        meta = infos_per_track.get(tid, None)
        if meta is None:
            color, lw, label = "#888888", 1.0, "unmatched"
        else:
            count, in_fov, in_range, _, _ = meta
            if not (in_fov and in_range):
                # Out of camera FOV / range — already filtered upstream.
                color, lw, label = "#888888", 1.0, f"oof {count}"
            else:
                _, color = tier_of(count, min_pts)
                lw = 2.4 if count < min_pts else 1.6
                label = f"{count}"
        rect = mpatches.Rectangle(
            (x1, y1), w, h, fill=False, edgecolor=color, linewidth=lw)
        ax.add_patch(rect)
        ax.text(
            x1 + 2, y1 + 2, label, color=color, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                      edgecolor="none", alpha=0.55))

    legend = [
        mpatches.Patch(color=TIER_FILTERED,
                       label=f"filtered (LiDAR<{min_pts})"),
        mpatches.Patch(color=TIER_BORDER, label="borderline"),
        mpatches.Patch(color=TIER_HEALTHY, label="healthy"),
        mpatches.Patch(color="#888888", label="out-of-FOV / no 3D match"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7,
              facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_title("camera (2D bbox color = num_lidar_pts tier)",
                 fontsize=10)


def draw_bev_panel(ax, pts_xyz_orig, gt_boxes_orig, counts, cam_vis,
                   min_pts, max_range=55.0):
    """BEV in the rotated frame, with LiDAR points + 3D box outlines."""
    # Rotate the cloud into the model frame
    if pts_xyz_orig.shape[0] > 0:
        xy_rot = rotate_xy_90ccw(pts_xyz_orig[:, :2])
        # Subsample for plotting performance
        if xy_rot.shape[0] > 80000:
            sel = np.random.choice(xy_rot.shape[0], 80000, replace=False)
            xy_rot = xy_rot[sel]
        ax.scatter(xy_rot[:, 0], xy_rot[:, 1], s=0.2, c="#9aa9c4",
                   alpha=0.55, linewidths=0)

    # FOV cone overlay
    half_deg = 30.0
    wedge = Wedge((0, 0), max_range, 90 - half_deg, 90 + half_deg,
                  facecolor="#00ff88", alpha=0.05, edgecolor="none")
    ax.add_patch(wedge)
    rx = max_range * np.sin(HALF_FOV_RAD)
    ry = max_range * np.cos(HALF_FOV_RAD)
    ax.plot([0, rx], [0, ry], "--", color="#00ff88", linewidth=1.0,
            alpha=0.7)
    ax.plot([0, -rx], [0, ry], "--", color="#00ff88", linewidth=1.0,
            alpha=0.7)

    # Boxes
    for i in range(len(gt_boxes_orig)):
        rot = rotate_box_inplace(gt_boxes_orig[i])
        rotated_xy = rot[:2]
        if not in_pc_range_rotated(rotated_xy):
            continue
        is_in_fov = in_fov_rotated(rotated_xy)
        is_cam = bool(cam_vis[i]) if cam_vis is not None else True
        c = int(counts[i])
        if not (is_in_fov and is_cam):
            color, lw = "#666666", 0.8
        else:
            _, color = tier_of(c, min_pts)
            lw = 2.0 if c < min_pts else 1.3
        corners = bottom_corners_xy(rot)
        poly = plt.Polygon(corners, closed=True, fill=False,
                           edgecolor=color, linewidth=lw, alpha=0.95)
        ax.add_patch(poly)
        # Heading arrow from box center
        cx, cy = rot[0], rot[1]
        yaw = rot[6]
        hx = cx + 1.5 * np.cos(yaw)
        hy = cy + 1.5 * np.sin(yaw)
        ax.plot([cx, hx], [cy, hy], "-", color=color, linewidth=lw * 0.9,
                alpha=0.95)
        # Annotate filtered / borderline boxes with their count
        if is_in_fov and is_cam and c < max(min_pts * 5, 5):
            ax.text(cx, cy + 1.5, f"{c}", color=color, fontsize=7,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                              edgecolor="none", alpha=0.55))

    # Ego marker
    ax.plot(0, 0, marker="o", markersize=6, color="#ffeb3b", mec="black",
            mew=0.8, zorder=10)

    ax.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
    ax.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
    ax.set_aspect("equal")
    ax.set_facecolor("#0a0e1a")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_xlabel("x (right) [m]", color="white", fontsize=8)
    ax.set_ylabel("y (forward) [m]", color="white", fontsize=8)
    ax.set_title("BEV: LiDAR sweep + 3D boxes", fontsize=10, color="white")


# --------------------------------------------------------------------- #
#  Per-frame
# --------------------------------------------------------------------- #
def visualize_frame(info, data_root, out_path, min_pts):
    """Render the two panels for one frame."""
    scenario = info["scenario"]
    fid = info["frame_idx"]
    fid_str = f"{fid:04d}"

    img_path = os.path.join(data_root, scenario, f"image_{fid_str}.png")
    pc_path = os.path.join(data_root, scenario, f"pc_{fid_str}.ply")
    label2d_path = os.path.join(data_root, scenario,
                                f"label2d_{fid_str}.json")

    if not os.path.exists(img_path):
        print(f"  skip (no image): {img_path}")
        return False

    img = np.asarray(Image.open(img_path).convert("RGB"))
    pts = load_pc_ply(pc_path) if os.path.exists(pc_path) \
        else np.zeros((0, 3), dtype=np.float32)
    label2d = load_label2d(label2d_path)

    counts = info["num_lidar_pts"]
    boxes = info["gt_boxes"]
    cam_vis = info.get("gt_camera_visible", None)

    # Build per-track 3D-side info dict for the camera panel
    # We need to read the raw label3d to get track_id, since pkl gt_inds is
    # the global numeric remap. Do it on the fly.
    raw_label3d_path = os.path.join(data_root, scenario,
                                    f"label3d_{fid_str}.txt")
    track_ids = parse_label3d_track_ids(raw_label3d_path)
    if len(track_ids) != len(boxes):
        # Defensive fallback: if mismatch, keep what we have
        track_ids = track_ids[:len(boxes)] + \
            ["__missing__"] * (len(boxes) - len(track_ids))

    infos_per_track = {}
    for i, tid in enumerate(track_ids):
        rotated_xy = (-float(boxes[i, 1]), float(boxes[i, 0]))
        infos_per_track[tid] = (
            int(counts[i]),
            in_fov_rotated(rotated_xy),
            in_pc_range_rotated(rotated_xy),
            None,
            bool(cam_vis[i]) if cam_vis is not None else True,
        )

    fig = plt.figure(figsize=(20, 9), facecolor="#0a0e1a")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.05)
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_bev = fig.add_subplot(gs[0, 1])

    draw_camera_panel(ax_cam, img, label2d, infos_per_track, min_pts)
    draw_bev_panel(ax_bev, pts, boxes, counts, cam_vis, min_pts)

    # Frame stats banner
    n_total = len(boxes)
    n_filt = int((np.array(counts) < min_pts).sum())
    n_in_fov = sum(
        1 for i in range(n_total)
        if in_fov_rotated((-float(boxes[i, 1]), float(boxes[i, 0])))
    )
    n_filt_in_fov = sum(
        1 for i in range(n_total)
        if in_fov_rotated((-float(boxes[i, 1]), float(boxes[i, 0])))
        and int(counts[i]) < min_pts
    )
    suptitle = (f"{scenario}  frame {fid_str}    "
                f"agents: {n_total}  in-FOV: {n_in_fov}  "
                f"filtered (LiDAR<{min_pts}): {n_filt}  "
                f"filtered & in-FOV: {n_filt_in_fov}")
    fig.suptitle(suptitle, color="white", fontsize=11, y=0.98)

    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return True


def parse_label3d_track_ids(path):
    """Read raw label3d_*.txt and return its track_id column in order.

    Necessary because the pkl stores `gt_inds` (a global numeric remap)
    instead of the raw UUID, but label2d_*.json keys by UUID. We assume
    the pkl preserves label3d row order — which create_loki_infos does.
    Class filtering matches LOKI_CLASS_MAP.
    """
    if not os.path.exists(path):
        return []
    out = []
    import csv
    import io
    with open(path, "r") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return out
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        reader = csv.reader(io.StringIO(line))
        parts = next(reader)
        if len(parts) < 14:
            continue
        label = parts[0].strip()
        if label not in LOKI_CLASS_MAP:
            continue
        out.append(parts[1].strip())  # track_id (UUID)
    return out


# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True,
                   help="Path to loki_infos_*.pkl")
    p.add_argument("--data-root", required=True,
                   help="LOKI data root (scenarios live here)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--min-lidar-pts", type=int, default=1,
                   help="The valid_flag threshold to visualize.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens", nargs="+", default=None,
                   help="Optional explicit list of frame tokens to render.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data \
        else data

    # Pick frames
    if args.tokens:
        wanted = set(args.tokens)
        chosen = [i for i in infos if i["token"] in wanted]
    else:
        scored = [(score_frame(i, args.min_lidar_pts), idx, i)
                  for idx, i in enumerate(infos)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        chosen = []
        seen_scenes = set()
        # Prefer 1 frame per scene to get visual variety, until we hit count
        for s, _, info in scored:
            if s <= 0:
                break
            if info["scenario"] in seen_scenes:
                continue
            chosen.append(info)
            seen_scenes.add(info["scenario"])
            if len(chosen) >= args.num_samples:
                break
        # Fall back: if not enough scoring frames, include the next-best
        if len(chosen) < args.num_samples:
            for s, _, info in scored:
                if info in chosen or s <= 0:
                    continue
                chosen.append(info)
                if len(chosen) >= args.num_samples:
                    break

    print(f"Rendering {len(chosen)} frames to {args.out_dir}")
    for k, info in enumerate(chosen):
        token = info["token"]
        out_path = os.path.join(args.out_dir, f"{k:02d}_{token}.png")
        ok = visualize_frame(info, args.data_root, out_path,
                             args.min_lidar_pts)
        if ok:
            print(f"  [{k+1}/{len(chosen)}] {out_path}")


if __name__ == "__main__":
    main()
