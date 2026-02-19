#!/usr/bin/env python3
"""
Visualize UniAD predictions vs ground truth on LOKI dataset.

Produces a combined image per frame:
  LEFT  — Camera view
            GT:   solid green 2D boxes from raw label2d_*.json files
            Pred: dashed colored 2D boxes projected from 3D model output
  RIGHT — BEV (bird's-eye view)
            GT:   solid white oriented rectangles from pkl gt_boxes
            Pred: dashed colored oriented rectangles from model output

Coordinate system (original LOKI / epoch-6):
  x = forward, y = lateral (left = +y, right = −y)
  PC range: x ∈ [0, 51.2],  y ∈ [−51.2, 51.2]

Usage:
    python tools/visualize_predictions.py \\
        --results work_dirs/base_loki_perception/results_epoch6.pkl \\
        --val-pkl  data/infos/loki_infos_val.pkl \\
        --data-root /mnt/storage/loki_data \\
        --out-dir  vis_predictions/ \\
        --num-frames 20 --score-thresh 0.4
"""

import argparse
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.lines import Line2D
from PIL import Image

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
CLASS_NAMES = [
    "Pedestrian",   # 0
    "Car",          # 1
    "Bus",          # 2
    "Truck",        # 3
    "Van",          # 4
    "Motorcyclist", # 5
    "Bicyclist",    # 6
    "Other",        # 7
]

# LOKI raw class  →  display name (for 2D GT label colouring)
LOKI_TO_DISPLAY = {
    "Pedestrian": "Pedestrian", "Car": "Car",
    "Van": "Van",               "Truck": "Truck",
    "Bus": "Bus",               "Motorcyclist": "Motorcyclist",
    "Bicyclist": "Bicyclist",
}

# pkl gt_names are lowercase  →  display name
PKL_TO_DISPLAY = {
    "pedestrian": "Pedestrian", "car": "Car",
    "van": "Van",               "truck": "Truck",
    "bus": "Bus",               "motorcycle": "Motorcyclist",
    "bicycle": "Bicyclist",
}

# Per-class colors (used for predictions; GT is white/green)
CLASS_COLORS = {
    "Pedestrian":   "#FF4500",
    "Car":          "#1E90FF",
    "Bus":          "#9400D3",
    "Truck":        "#FF8C00",
    "Van":          "#00BFFF",
    "Motorcyclist": "#00CC66",
    "Bicyclist":    "#00BFBF",
    "Other":        "#AAAAAA",
}

GT_CAM_COLOR = "#00FF00"   # solid green for camera GT
GT_BEV_COLOR = "#FFFFFF"   # solid white for BEV GT

PC_RANGE = [0.0, -51.2, -5.0, 51.2, 51.2, 3.0]  # [xmin,ymin,zmin,xmax,ymax,zmax]


# ------------------------------------------------------------------ #
#  2D GT: raw label2d JSON
# ------------------------------------------------------------------ #
def load_label2d(data_root, scenario, frame_idx):
    fid  = f"{frame_idx:04d}" if isinstance(frame_idx, int) else str(frame_idx)
    path = os.path.join(data_root, scenario, f"label2d_{fid}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    anns = []
    for raw_cls, objs in data.items():
        disp = LOKI_TO_DISPLAY.get(raw_cls)
        if disp is None or not isinstance(objs, dict):
            continue
        for _, obj in objs.items():
            box = obj.get("box")
            if box:
                anns.append(dict(cls=disp, box=box))
    return anns


# ------------------------------------------------------------------ #
#  3-D geometry helpers
# ------------------------------------------------------------------ #
def box_corners_3d(box):
    """8 corners of a 3D box. box: [x,y,z,dx,dy,dz,yaw,...]. Returns (8,3)."""
    x, y, z, dx, dy, dz, yaw = box[:7]
    hdx, hdy, hdz = dx / 2, dy / 2, dz / 2
    local = np.array([
        [-hdx, -hdy, -hdz], [-hdx, -hdy,  hdz],
        [-hdx,  hdy, -hdz], [-hdx,  hdy,  hdz],
        [ hdx, -hdy, -hdz], [ hdx, -hdy,  hdz],
        [ hdx,  hdy, -hdz], [ hdx,  hdy,  hdz],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return (R @ local.T).T + [x, y, z]


def project_box_to_2d(box, lidar2img, img_w, img_h):
    """Project 3D box corners to axis-aligned 2D bbox. Returns (x1,y1,x2,y2) or None."""
    corners = box_corners_3d(box)
    pts_h   = np.concatenate([corners, np.ones((8, 1))], axis=1)
    proj    = (lidar2img @ pts_h.T).T
    depth   = proj[:, 2]
    valid   = depth > 0.1
    if not np.any(valid):
        return None
    u, v = proj[valid, 0] / depth[valid], proj[valid, 1] / depth[valid]
    x1 = float(np.clip(u.min(), 0, img_w - 1))
    y1 = float(np.clip(v.min(), 0, img_h - 1))
    x2 = float(np.clip(u.max(), 0, img_w - 1))
    y2 = float(np.clip(v.max(), 0, img_h - 1))
    return (x1, y1, x2, y2) if (x2 - x1 >= 1 and y2 - y1 >= 1) else None


def corners_bev(box):
    """
    4 bottom-face corners of a box in the BEV (xy) plane.
    Returns (4, 2) array of [x_forward, y_lateral] positions.
    """
    x, y, _, dx, dy, _, yaw = box[:7]
    hdx, hdy = dx / 2, dy / 2
    local = np.array([
        [-hdx, -hdy], [-hdx,  hdy],
        [ hdx,  hdy], [ hdx, -hdy],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R    = np.array([[c, -s], [s, c]])
    pts  = (R @ local.T).T + [x, y]
    return pts  # (4, 2)  col0=x_fwd, col1=y_lat


def heading_arrow(box):
    """
    Returns (start_xy, end_xy) in [x_fwd, y_lat] for a heading arrow
    drawn from box centre toward the vehicle nose (local +y direction,
    which is the length axis for LOKI vehicles at typical yaw≈−π/2).
    """
    x, y, _, dx, dy, _, yaw = box[:7]
    hdy = dy / 2
    # Front midpoint = centre + (-sin(yaw), cos(yaw)) * hdy/2  (half-length)
    scale = hdy * 0.6
    fx = x + (-np.sin(yaw)) * scale
    fy = y + ( np.cos(yaw)) * scale
    return (x, y), (fx, fy)


# ------------------------------------------------------------------ #
#  Drawing helpers
# ------------------------------------------------------------------ #
def draw_cam_box(ax, x1, y1, x2, y2, color, label=None,
                 linestyle="-", linewidth=2.0, alpha=0.9):
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color, facecolor="none",
        linestyle=linestyle, alpha=alpha,
    ))
    if label:
        ax.text(x1, y1 - 3, label, fontsize=6, color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55),
                ha="left", va="bottom")


def draw_bev_box(ax, box, color, linestyle="-", linewidth=1.5,
                 label=None, alpha=0.9):
    """Draw oriented box in BEV axes (ax.x = y_lat, ax.y = x_fwd)."""
    corners = corners_bev(box)          # (4,2), col0=x_fwd, col1=y_lat
    # matplotlib: x-axis = y_lat, y-axis = x_fwd
    poly_pts = np.column_stack([corners[:, 1], corners[:, 0]])  # (4,2) [y_lat, x_fwd]
    ax.add_patch(plt.Polygon(
        poly_pts, closed=True, fill=False,
        edgecolor=color, linewidth=linewidth,
        linestyle=linestyle, alpha=alpha,
    ))
    # Heading arrow
    (cx, cy), (fx, fy) = heading_arrow(box)
    ax.annotate("", xy=(fy, fx), xytext=(cy, cx),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=linewidth * 0.8))
    if label:
        ax.text(cy, cx, label, fontsize=5, color=color,
                ha="center", va="center", alpha=0.8)


# ------------------------------------------------------------------ #
#  Per-frame combined visualization
# ------------------------------------------------------------------ #
def visualize_frame(info, pred, data_root, out_dir,
                    frame_idx_global, score_thresh=0.4):
    img_path = info["img_filename"]
    if not os.path.exists(img_path):
        return False

    img      = np.array(Image.open(img_path))
    img_h, img_w = img.shape[:2]

    lidar2img = np.array(info["lidar2img"], dtype=np.float64)
    scenario  = info.get("scenario", "?")
    frame_idx = info.get("frame_idx", "?")
    token     = info.get("token", "?")

    # GT 2D labels (camera)
    gt_anns_2d = load_label2d(data_root, scenario, frame_idx)

    # GT 3D boxes (BEV) — filter by PC range
    gt_boxes = info["gt_boxes"]   # (N, 9)
    gt_names = info["gt_names"]   # (N,)
    def in_range(b):
        return (PC_RANGE[0] <= b[0] <= PC_RANGE[3] and
                PC_RANGE[1] <= b[1] <= PC_RANGE[4])
    gt_bev = [(b, PKL_TO_DISPLAY.get(str(n), str(n)))
              for b, n in zip(gt_boxes, gt_names) if in_range(b)]

    # Predictions
    pred_boxes  = pred["boxes_3d"].tensor.numpy()
    pred_scores = pred["scores_3d"].numpy()
    pred_labels = pred["labels_3d"].numpy()
    mask        = pred_scores >= score_thresh
    pred_boxes  = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    pred_labels = pred_labels[mask]

    # ---------------------------------------------------------------- #
    #  Figure layout:  camera (left 58%)  |  BEV (right 42%)
    #  BEV range:  x ∈ [0, 51.2]m fwd,  y ∈ [−51.2, 51.2]m lat
    #  → physical aspect 102.4 : 51.2 = 2 : 1  (wide)
    #  We give BEV a square axes and let matplotlib pad.
    # ---------------------------------------------------------------- #
    fig = plt.figure(figsize=(26, 11), facecolor="#0d1117")

    ax_cam = fig.add_axes([0.01, 0.06, 0.54, 0.88])
    ax_bev = fig.add_axes([0.58, 0.06, 0.40, 0.88])

    # ==================== CAMERA PANEL ====================
    ax_cam.imshow(img)
    ax_cam.set_axis_off()

    for ann in gt_anns_2d:
        b = ann["box"]
        draw_cam_box(ax_cam,
                     float(b["left"]), float(b["top"]),
                     float(b["left"]) + float(b["width"]),
                     float(b["top"])  + float(b["height"]),
                     color=GT_CAM_COLOR, label=ann["cls"])

    n_pred_cam = 0
    seen_pred_cls = set()
    for box, score, lid in zip(pred_boxes, pred_scores, pred_labels):
        cls = CLASS_NAMES[lid] if 0 <= lid < len(CLASS_NAMES) else f"cls{lid}"
        box2d = project_box_to_2d(box, lidar2img, img_w, img_h)
        if box2d is None:
            continue
        draw_cam_box(ax_cam, *box2d,
                     color=CLASS_COLORS.get(cls, "#FFF"),
                     label=f"{cls} {score:.2f}",
                     linestyle="--")
        seen_pred_cls.add(cls)
        n_pred_cam += 1

    cam_legend = [
        mpatches.Patch(edgecolor=GT_CAM_COLOR, facecolor="none",
                       lw=2, label="GT (2D label)"),
    ] + [
        mpatches.Patch(edgecolor=CLASS_COLORS.get(c, "#FFF"),
                       facecolor="none", lw=2, linestyle="--",
                       label=f"Pred: {c}")
        for c in sorted(seen_pred_cls)
    ]
    ax_cam.legend(handles=cam_legend, loc="upper right",
                  fontsize=7, framealpha=0.65,
                  facecolor="black", labelcolor="white")
    ax_cam.set_title(
        f"{scenario}  frame {frame_idx}"
        f"  |  GT: {len(gt_anns_2d)}  |  Pred: {n_pred_cam} (thr={score_thresh})",
        color="white", fontsize=9, pad=4,
    )

    # ==================== BEV PANEL ====================
    ax_bev.set_facecolor("#0d1117")

    # Axis limits:  matplotlib x = y_loki (lateral),  y = x_loki (forward)
    # Invert x so LEFT side of vehicle appears on LEFT of plot
    ax_bev.set_xlim(51.2, -51.2)          # left edge = +51.2 (vehicle left)
    ax_bev.set_ylim(-2, 53)               # small margin below ego
    ax_bev.set_aspect("equal")

    # Grid & range boundary
    for d in range(10, 55, 10):
        ax_bev.axhline(d, color="#333344", lw=0.5, ls=":")
        ax_bev.text(52, d, f"{d}m", fontsize=6, color="#555566",
                    ha="left", va="center")
    for lat in range(-50, 51, 10):
        ax_bev.axvline(lat, color="#333344", lw=0.5, ls=":")
    ax_bev.add_patch(plt.Rectangle(
        (-51.2, 0), 102.4, 51.2,
        edgecolor="#446644", facecolor="none", lw=1.5, ls="--",
    ))

    # Ego vehicle marker (small green rectangle at origin)
    EGO_W, EGO_L = 2.0, 4.5
    ax_bev.add_patch(plt.Rectangle(
        (-EGO_W / 2, 0), EGO_W, EGO_L,
        facecolor="#22aa44", edgecolor="white", lw=1.0, alpha=0.85,
    ))
    ax_bev.text(0, -1.5, "EGO", fontsize=6, color="#22aa44",
                ha="center", va="top")

    # GT boxes (solid white)
    for box, cls_name in gt_bev:
        draw_bev_box(ax_bev, box, color=GT_BEV_COLOR,
                     linestyle="-", linewidth=1.5)

    # Predicted boxes (dashed, class color)
    for box, score, lid in zip(pred_boxes, pred_scores, pred_labels):
        cls = CLASS_NAMES[lid] if 0 <= lid < len(CLASS_NAMES) else f"cls{lid}"
        color = CLASS_COLORS.get(cls, "#FFF")
        draw_bev_box(ax_bev, box, color=color,
                     linestyle="--", linewidth=1.8,
                     label=f"{cls[0]}{score:.1f}")  # compact label

    # Axis labels
    ax_bev.set_xlabel("Lateral  y  (m)  ← left  |  right →",
                      color="#aaaaaa", fontsize=8)
    ax_bev.set_ylabel("Forward  x  (m)", color="#aaaaaa", fontsize=8)
    ax_bev.tick_params(colors="#777777")
    for spine in ax_bev.spines.values():
        spine.set_edgecolor("#444444")

    # BEV legend
    bev_legend = [
        Line2D([0], [0], color=GT_BEV_COLOR, lw=1.5, label="GT 3D box"),
    ] + [
        Line2D([0], [0], color=CLASS_COLORS.get(c, "#FFF"),
               lw=1.8, ls="--", label=f"Pred: {c}")
        for c in sorted(seen_pred_cls)
    ]
    ax_bev.legend(handles=bev_legend, loc="upper right",
                  fontsize=7, framealpha=0.65,
                  facecolor="#111122", labelcolor="white")
    ax_bev.set_title(
        f"BEV  |  GT: {len(gt_bev)}  |  Pred: {len(pred_boxes)}",
        color="white", fontsize=9, pad=4,
    )

    # Save
    plt.tight_layout(pad=0.3)
    out_path = os.path.join(out_dir, f"{frame_idx_global:04d}_{token}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Visualize UniAD predictions vs GT on LOKI (camera + BEV)")
    parser.add_argument("--results",    required=True)
    parser.add_argument("--val-pkl",    required=True)
    parser.add_argument("--data-root",  default="/mnt/storage/loki_data")
    parser.add_argument("--out-dir",    default="vis_predictions")
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--tokens",     nargs="*", default=None)
    parser.add_argument("--all-frames", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading results: {args.results}")
    with open(args.results, "rb") as f:
        results = pickle.load(f)
    pred_list    = results["bbox_results"]
    pred_by_tok  = {p["token"]: p for p in pred_list}

    print(f"Loading val info: {args.val_pkl}")
    with open(args.val_pkl, "rb") as f:
        val_data = pickle.load(f)
    info_by_tok = {i["token"]: i for i in val_data["infos"]}

    tokens = (args.tokens if args.tokens
              else [p["token"] for p in pred_list]
              if args.all_frames
              else [p["token"] for p in pred_list[:args.num_frames]])

    print(f"Visualizing {len(tokens)} frames  "
          f"(score_thresh={args.score_thresh})\n")

    n_ok = n_skip = 0
    for i, tok in enumerate(tokens):
        info = info_by_tok.get(tok)
        pred = pred_by_tok.get(tok)
        if info is None or pred is None:
            n_skip += 1
            continue
        ok = visualize_frame(
            info, pred, args.data_root, args.out_dir,
            frame_idx_global=i, score_thresh=args.score_thresh,
        )
        n_pred = int((pred["scores_3d"] >= args.score_thresh).sum())
        print(f"  [{'OK' if ok else 'SKIP'}] {i+1:3d}/{len(tokens)}  {tok}  Pred={n_pred}")
        n_ok += ok
        n_skip += not ok

    print(f"\nDone: {n_ok} saved, {n_skip} skipped → {args.out_dir}/")


if __name__ == "__main__":
    main()