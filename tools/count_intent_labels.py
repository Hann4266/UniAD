"""
Count intent label distribution in a LOKI pkl, applying the same
FOV + range + camera-visibility filters as the training pipeline.
Computes sqrt-inverse-frequency class weights for the loss.

Adapted from main branch count_intent_labels.py — simplified because
LOKI stores gt_intent_labels directly in the pkl.

Usage:
    python tools/count_intent_labels.py \
        --ann_file /mnt/storage/UniAD/data/infos/loki_infos_train_intent_filtered.pkl
"""

import argparse
import pickle
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

INTENT_NAMES = {
    -1: "INVALID / na",
    0: "STOPPED",
    1: "MOVING",
    2: "LCL",
    3: "LCR",
    4: "TL",
    5: "TR",
    6: "CROSSING",
}

VEHICLE_CLASSES = {"car", "truck", "bus", "van", "motorcyclist", "bicyclist"}
PEDESTRIAN_CLASSES = {"pedestrian"}


def agent_type(name):
    name = name.lower()
    if name in VEHICLE_CLASSES:
        return "vehicle"
    if name in PEDESTRIAN_CLASSES:
        return "pedestrian"
    return "other"


def apply_range_filter(centers_xy, pcd_range, fov_deg=60.0):
    x, y = centers_xy[:, 0], centers_xy[:, 1]
    x_min, y_min = float(pcd_range[0]), float(pcd_range[1])
    x_max, y_max = float(pcd_range[3]), float(pcd_range[4])
    tan_half_fov = np.tan(np.deg2rad(fov_deg / 2.0))
    mask_fov = (y > 0) & (np.abs(x) <= y * tan_half_fov)
    mask_x = (x >= x_min) & (x <= x_max)
    mask_y = (y >= y_min) & (y <= y_max)
    return mask_fov & mask_x & mask_y


def rotate_boxes_to_rotated_frame(gt_boxes):
    boxes = gt_boxes.copy()
    x_old = boxes[:, 0].copy()
    y_old = boxes[:, 1].copy()
    boxes[:, 0] = -y_old
    boxes[:, 1] = x_old
    return boxes


def print_report(title, counter, label_names):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)
    total = sum(counter.values())
    for lbl in sorted(label_names.keys()):
        cnt = counter.get(lbl, 0)
        print(f"  {lbl:>3}  {label_names[lbl]:<22}  {cnt:>8}  ({100 * cnt / max(total, 1):.2f}%)")
    print(f"  {'':>3}  {'TOTAL':<22}  {total:>8}")


def compute_class_weights(counter, num_classes=7):
    """Compute sqrt-inverse-frequency class weights."""
    valid_counts = np.array([counter.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    valid_counts = np.maximum(valid_counts, 1)  # avoid div by zero
    freq = valid_counts / valid_counts.sum()
    inv_freq = 1.0 / freq
    sqrt_inv = np.sqrt(inv_freq)
    # Normalize so the majority class (min weight) = 1.0
    sqrt_inv = sqrt_inv / sqrt_inv.min()
    return sqrt_inv


def main(args):
    print(f"Loading: {args.ann_file}")
    with open(args.ann_file, "rb") as f:
        data = pickle.load(f)
    data_infos = data["infos"]
    print(f"  Total frames: {len(data_infos)}")

    pcd_range = args.point_cloud_range
    fov_deg = args.fov_deg

    counters = {
        "all": Counter(),
        "vehicle": Counter(),
        "pedestrian": Counter(),
    }

    for info in tqdm(data_infos, desc="Processing"):
        gt_intent = info.get("gt_intent_labels", None)
        if gt_intent is None:
            continue

        mask_valid = info.get("valid_flag", np.ones(len(gt_intent), dtype=bool))
        gt_boxes = info["gt_boxes"][mask_valid]
        gt_intent = gt_intent[mask_valid]
        gt_names = info["gt_names"][mask_valid]

        # Camera visibility filter
        gt_cam_vis = info.get("gt_camera_visible", None)
        if gt_cam_vis is not None:
            gt_cam_vis = gt_cam_vis[mask_valid]
            gt_boxes = gt_boxes[gt_cam_vis]
            gt_intent = gt_intent[gt_cam_vis]
            gt_names = gt_names[gt_cam_vis]

        if len(gt_boxes) == 0:
            continue

        # Rotate to match training frame
        gt_boxes_rot = rotate_boxes_to_rotated_frame(gt_boxes)

        # FOV + range filter
        keep = apply_range_filter(gt_boxes_rot[:, :2], pcd_range, fov_deg)
        gt_intent = gt_intent[keep]
        gt_names = gt_names[keep]

        for lbl, name in zip(gt_intent, gt_names):
            lbl = int(lbl)
            atype = agent_type(str(name))
            counters["all"][lbl] += 1
            if atype in counters:
                counters[atype][lbl] += 1

    # ── reports ──────────────────────────────────────────────────────────
    for key, title in [
        ("all", f"ALL AGENTS  (FOV={fov_deg}°, range={pcd_range})"),
        ("vehicle", "VEHICLES ONLY"),
        ("pedestrian", "PEDESTRIANS ONLY"),
    ]:
        print_report(title, counters[key], INTENT_NAMES)

    # ── class weights ────────────────────────────────────────────────────
    weights = compute_class_weights(counters["all"], num_classes=7)
    print("\n" + "=" * 65)
    print("SQRT-INVERSE-FREQUENCY CLASS WEIGHTS (for loss_cls config)")
    print("=" * 65)
    weight_strs = []
    for i in range(7):
        cnt = counters["all"].get(i, 0)
        print(f"  [{i}] {INTENT_NAMES[i]:<22}  count={cnt:>8}  weight={weights[i]:.4f}")
        weight_strs.append(f"{weights[i]:.2f}")
    print(f"\n  class_weight=[{', '.join(weight_strs)}]")
    print("=" * 65)

    # ── imbalance ratios ─────────────────────────────────────────────────
    for group in ["all", "vehicle", "pedestrian"]:
        print(f"\n  IMBALANCE ({group}, excl. -1):")
        valid_counts = {k: v for k, v in counters[group].items() if k >= 0}
        if valid_counts:
            max_cnt = max(valid_counts.values())
            for lbl, cnt in sorted(valid_counts.items(), key=lambda x: x[1]):
                ratio = max_cnt / cnt if cnt > 0 else float("inf")
                print(f"    [{lbl}] {INTENT_NAMES[lbl]:<22}  {cnt:>8}  "
                      f"imbalance vs majority: {ratio:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file",
                        default="/mnt/storage/UniAD/data/infos/loki_infos_train.pkl")
    parser.add_argument("--point_cloud_range", type=float, nargs=6,
                        default=[-51.2, 0, -5.0, 51.2, 51.2, 3.0],
                        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"))
    parser.add_argument("--fov_deg", type=float, default=60.0)
    args = parser.parse_args()
    main(args)
