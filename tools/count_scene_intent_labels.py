"""
Count how many LOKI scenes contain at least one agent with a target
intent label (after FOV + range + camera-visibility filter).

Target labels (LOKI IDs):
  2=LCL, 3=LCR, 4=TL, 5=TR, 6=CROSSING

"Boring" scenes contain only STOPPED (0) and MOVING (1).

Adapted from main branch count_scene_labels.py — simplified because
LOKI stores gt_intent_labels directly in the pkl.

Usage:
    python tools/count_scene_intent_labels.py \
        --ann_file data/infos/loki_infos_train.pkl
"""

import argparse
import json
import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

TARGET_LABELS = {2, 3, 4, 5, 6}
TARGET_NAMES = {
    2: "LCL",
    3: "LCR",
    4: "TL",
    5: "TR",
    6: "CROSSING",
}

INTENT_NAMES = {
    -1: "INVALID",
    0: "STOPPED",
    1: "MOVING",
    2: "LCL",
    3: "LCR",
    4: "TL",
    5: "TR",
    6: "CROSSING",
}


def apply_range_filter(centers_xy, pcd_range, fov_deg=60.0):
    """FOV + BEV range filter matching training pipeline."""
    x, y = centers_xy[:, 0], centers_xy[:, 1]
    x_min, y_min = float(pcd_range[0]), float(pcd_range[1])
    x_max, y_max = float(pcd_range[3]), float(pcd_range[4])
    # FOV filter: rotated LOKI frame, +y = forward
    tan_half_fov = np.tan(np.deg2rad(fov_deg / 2.0))
    mask_fov = (y > 0) & (np.abs(x) <= y * tan_half_fov)
    mask_x = (x >= x_min) & (x <= x_max)
    mask_y = (y >= y_min) & (y <= y_max)
    return mask_fov & mask_x & mask_y


def rotate_boxes_to_rotated_frame(gt_boxes):
    """Apply 90° CCW rotation to match dataset loader convention.

    Original LOKI pkl: x=forward, y=lateral.
    Rotated frame:     x=right,   y=forward.
    Transform: x_new = -y_old, y_new = x_old
    """
    boxes = gt_boxes.copy()
    x_old = boxes[:, 0].copy()
    y_old = boxes[:, 1].copy()
    boxes[:, 0] = -y_old
    boxes[:, 1] = x_old
    return boxes


def main(args):
    print(f"Loading: {args.ann_file}")
    with open(args.ann_file, "rb") as f:
        data = pickle.load(f)
    data_infos = data["infos"]
    print(f"  Total frames: {len(data_infos)}")

    pcd_range = args.point_cloud_range
    fov_deg = args.fov_deg

    # scene_token -> set of target labels seen
    scene_labels = defaultdict(set)
    # scene_token -> set of ALL labels seen (for report)
    scene_all_labels = defaultdict(set)
    all_scene_tokens = set()

    for info in tqdm(data_infos, desc="Processing"):
        scene_token = info["scene_token"]
        all_scene_tokens.add(scene_token)

        gt_intent = info.get("gt_intent_labels", None)
        if gt_intent is None:
            continue

        mask_valid = info.get("valid_flag", np.ones(len(gt_intent), dtype=bool))
        gt_boxes = info["gt_boxes"][mask_valid]
        gt_intent = gt_intent[mask_valid]

        # Camera visibility filter
        gt_cam_vis = info.get("gt_camera_visible", None)
        if gt_cam_vis is not None:
            gt_cam_vis = gt_cam_vis[mask_valid]
            gt_boxes = gt_boxes[gt_cam_vis]
            gt_intent = gt_intent[gt_cam_vis]

        if len(gt_boxes) == 0:
            continue

        # Rotate to match training frame before applying FOV + range filter
        gt_boxes_rot = rotate_boxes_to_rotated_frame(gt_boxes)

        # FOV + range filter
        keep = apply_range_filter(gt_boxes_rot[:, :2], pcd_range, fov_deg)
        gt_intent_kept = gt_intent[keep]

        if len(gt_intent_kept) == 0:
            continue

        for lbl in gt_intent_kept:
            lbl = int(lbl)
            if lbl >= 0:
                scene_all_labels[scene_token].add(lbl)
            if lbl in TARGET_LABELS:
                scene_labels[scene_token].add(lbl)

    # ── analysis ─────────────────────────────────────────────────────────
    total_scenes = len(all_scene_tokens)
    scenes_with_any = {s for s, lbls in scene_labels.items() if lbls & TARGET_LABELS}
    boring_scenes = all_scene_tokens - scenes_with_any
    scenes_per_label = {
        lbl: {s for s, lbls in scene_labels.items() if lbl in lbls}
        for lbl in TARGET_LABELS
    }

    # Sub-groups
    scenes_turning = scenes_per_label[4] | scenes_per_label[5]
    scenes_lc = scenes_per_label[2] | scenes_per_label[3]
    scenes_crossing = scenes_per_label[6]

    # ── save scene tokens to JSON ────────────────────────────────────────
    base = os.path.splitext(os.path.basename(args.ann_file))[0]
    prefix = os.path.join(args.output_dir, base)
    os.makedirs(args.output_dir, exist_ok=True)

    out_all = f"{prefix}_interesting_scenes.json"
    with open(out_all, "w") as f:
        json.dump(sorted(scenes_with_any), f, indent=2)
    print(f"\nSaved: {out_all}  ({len(scenes_with_any)} scenes)")

    out_boring = f"{prefix}_boring_scenes.json"
    with open(out_boring, "w") as f:
        json.dump(sorted(boring_scenes), f, indent=2)
    print(f"Saved: {out_boring}  ({len(boring_scenes)} scenes)")

    out_turning = f"{prefix}_turning_scenes.json"
    with open(out_turning, "w") as f:
        json.dump(sorted(scenes_turning), f, indent=2)
    print(f"Saved: {out_turning}  ({len(scenes_turning)} scenes)")

    out_lc = f"{prefix}_lane_change_scenes.json"
    with open(out_lc, "w") as f:
        json.dump(sorted(scenes_lc), f, indent=2)
    print(f"Saved: {out_lc}  ({len(scenes_lc)} scenes)")

    # ── print report ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"LOKI Scene Intent Analysis  (FOV={fov_deg}°, range={pcd_range})")
    print(f"File: {args.ann_file}")
    print("=" * 65)
    print(f"Total scenes                               : {total_scenes}")
    print(f"Interesting (has target label)              : {len(scenes_with_any):>5}  "
          f"({100 * len(scenes_with_any) / max(total_scenes, 1):.1f}%)")
    print(f"Boring (only STOP + MOVING)                : {len(boring_scenes):>5}  "
          f"({100 * len(boring_scenes) / max(total_scenes, 1):.1f}%)")
    print("-" * 65)
    print(f"  with TURNING   (TL or TR)                : {len(scenes_turning):>5}  "
          f"({100 * len(scenes_turning) / max(total_scenes, 1):.1f}%)")
    print(f"  with LANE_CHANGE (LCL or LCR)            : {len(scenes_lc):>5}  "
          f"({100 * len(scenes_lc) / max(total_scenes, 1):.1f}%)")
    print(f"  with CROSSING                            : {len(scenes_crossing):>5}  "
          f"({100 * len(scenes_crossing) / max(total_scenes, 1):.1f}%)")
    print("-" * 65)
    print("Per-label breakdown (scenes with >= 1 occurrence after filter):")
    for lbl in sorted(TARGET_LABELS):
        n = len(scenes_per_label[lbl])
        print(f"  [{lbl}] {TARGET_NAMES[lbl]:<22}  {n:>5} scenes  "
              f"({100 * n / max(total_scenes, 1):.1f}%)")
    print("=" * 65)

    # ── list boring scenes ───────────────────────────────────────────────
    if boring_scenes:
        print(f"\nBoring scenes ({len(boring_scenes)}):")
        for s in sorted(boring_scenes):
            labels_in_scene = scene_all_labels.get(s, set())
            label_str = ", ".join(INTENT_NAMES.get(l, str(l)) for l in sorted(labels_in_scene))
            print(f"  {s}: {{{label_str}}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file",
                        default="data/infos/loki_infos_train.pkl")
    parser.add_argument("--point_cloud_range", type=float, nargs=6,
                        default=[-51.2, 0, -5.0, 51.2, 51.2, 3.0],
                        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"))
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--output_dir", default="./scene_tokens",
                        help="Directory for output JSON files")
    args = parser.parse_args()
    main(args)
