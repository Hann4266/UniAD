"""
Count intent label distribution in the training dataset,
applying the same FOV + BEV range filter as ObjectRangeFilterTrack.

Usage:
    python count_intent_labels.py \
        --ann_file /path/to/nuscenes_infos_train.pkl \
        --intent_file /zihan-west-vol/UniAD/data/nuscenes/unified_map_v2/all_scenes_compact.json \
        --point_cloud_range -51.2 -51.2 -5.0 51.2 51.2 3.0 \
        --fov_deg 70
"""

import argparse
import json
import pickle
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm


# ── label definitions (mirrors your intent_label() function) ─────────────────
INTENT_NAMES = {
    -1: "INVALID / na",
     0: "STOPPED",
     1: "MOVING",
     2: "Crossing",
     3: "TURN_RIGHT",
     4: "TURN_LEFT",
     5: "LANE_CHANGE_RIGHT",
     6: "LANE_CHANGE_LEFT",
}


def intent_label(index, action_array):
    intent_dic = {
        "Crossing":          2,
        "TURN_RIGHT":        3,
        "TURN_LEFT":         4,
        "LANE_CHANGE_RIGHT": 5,
        "LANE_CHANGE_LEFT":  6,
        "STOPPED":  0,
        "MOVING":   1,
        "Stopped":  0,
        "Moving":   1,
    }
    if index < 0 or index >= len(action_array) - 1:
        return -1

    action = action_array[index]
    end = min(len(action_array), index + 5)
    future_actions = action_array[index + 1:end]

    if len(future_actions) == 0 or all(a == "na" for a in future_actions):
        return -1

    action_set = {"Crossing", "TURN_RIGHT", "TURN_LEFT",
                  "LANE_CHANGE_RIGHT", "LANE_CHANGE_LEFT"}

    if action in action_set:
        for next_action in future_actions:
            if next_action != action and next_action != "na":
                return intent_dic[next_action]
        return intent_dic[action]
    elif action in ("STOPPED", "Stopped"):
        next_moving = False
        for next_action in future_actions:
            if next_action in action_set:
                return intent_dic[next_action]
            elif next_action in ("MOVING", "Moving"):
                next_moving = True
        return intent_dic["MOVING"] if next_moving else intent_dic["STOPPED"]
    elif action in ("MOVING", "Moving"):
        next_stop = False
        for next_action in future_actions:
            if next_action in action_set:
                return intent_dic[next_action]
            elif next_action in ("STOPPED", "Stopped"):
                next_stop = True
        return intent_dic["STOPPED"] if next_stop else intent_dic["MOVING"]
    elif action == "na":
        for next_action in future_actions:
            if next_action != "na":
                return intent_dic[next_action]
        return -1
    else:
        return -1


# ── range / FOV filter (mirrors ObjectRangeFilterTrack) ──────────────────────
def apply_range_filter(centers_xy, pcd_range, fov_deg=70.0):
    """
    centers_xy : (N, 2) numpy array  [x, y] in ego/LiDAR frame
    pcd_range  : [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns boolean mask of shape (N,)
    """
    x, y = centers_xy[:, 0], centers_xy[:, 1]

    x_min, y_min = float(pcd_range[0]), float(pcd_range[1])
    x_max, y_max = float(pcd_range[3]), float(pcd_range[4])

    half_fov_rad  = np.deg2rad(fov_deg / 2.0)
    tan_half_fov  = np.tan(half_fov_rad)

    forward_dist  = y              # ego at (0,0), forward = +y
    lateral_dist  = np.abs(x)

    mask_fov = (forward_dist > 0) & (lateral_dist <= forward_dist * tan_half_fov)
    mask_x   = (x >= x_min) & (x <= x_max)
    mask_y   = (y >= y_min) & (y <= y_max)

    return mask_fov & mask_x & mask_y


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    print(f"Loading annotation file: {args.ann_file}")
    with open(args.ann_file, "rb") as f:
        data = pickle.load(f)

    data_infos = sorted(data["infos"], key=lambda e: e["timestamp"])
    print(f"  Total frames: {len(data_infos)}")

    print(f"Loading intent file: {args.intent_file}")
    with open(args.intent_file, "r") as f:
        intent_data = json.load(f)

    pcd_range = args.point_cloud_range  # list of 6 floats
    fov_deg   = args.fov_deg

    # counters
    counter_before_filter = Counter()
    counter_after_filter  = Counter()
    per_label_frame_count = defaultdict(int)   # how many frames contain each label (after filter)

    # ── diagnostic counters for -1 reasons ──────────────────────────────────
    reason_not_in_data    = 0   # token 不在 intent_data 里
    reason_end_of_seq     = 0   # frame_idx >= len(arr)-1
    reason_na_action      = 0   # 当前 action 本身是 "na"
    reason_all_future_na  = 0   # 未来 4 步全是 "na"

    for info in tqdm(data_infos, desc="Processing frames"):
        frame_idx   = int(info["frame_idx"])
        scene_token = info["scene_token"]

        # ── pick valid boxes ────────────────────────────────────────────────
        if "valid_flag" in info:          # use_valid_flag == True path
            mask_valid = info["valid_flag"]
        else:
            mask_valid = info["num_lidar_pts"] > 0

        gt_boxes  = info["gt_boxes"][mask_valid]        # (N, ≥7)
        gt_tokens = info["gt_ins_tokens"][mask_valid]

        if len(gt_boxes) == 0:
            continue

        # ── compute intent labels + diagnose -1 reasons ─────────────────────
        agent_keys = intent_data.get(scene_token, {}).keys()
        intent_labels_raw = []
        for tok in gt_tokens:
            if tok not in agent_keys:
                reason_not_in_data += 1
                intent_labels_raw.append(-1)
                continue

            arr = intent_data[scene_token][tok]["labels"]

            if frame_idx < 0 or frame_idx >= len(arr) - 1:
                reason_end_of_seq += 1
                intent_labels_raw.append(-1)
                continue

            action = arr[frame_idx]
            future = arr[frame_idx + 1 : min(len(arr), frame_idx + 5)]

            if len(future) == 0 or all(a == "na" for a in future):
                reason_all_future_na += 1
                intent_labels_raw.append(-1)
                continue

            if action == "na" and all(a == "na" for a in future):
                reason_na_action += 1
                intent_labels_raw.append(-1)
                continue

            # normal path
            intent_labels_raw.append(intent_label(frame_idx, arr))

        intent_labels_raw = np.array(intent_labels_raw)

        for lbl in intent_labels_raw:
            counter_before_filter[int(lbl)] += 1

        # ── spatial + FOV filter ─────────────────────────────────────────────
        centers_xy = gt_boxes[:, :2]   # x, y
        keep = apply_range_filter(centers_xy, pcd_range, fov_deg)

        intent_labels_filtered = intent_labels_raw[keep]
        for lbl in intent_labels_filtered:
            counter_after_filter[int(lbl)] += 1

        # per-frame presence
        for lbl in np.unique(intent_labels_filtered):
            per_label_frame_count[int(lbl)] += 1

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INTENT LABEL COUNTS  (BEFORE spatial/FOV filter)")
    print("=" * 60)
    total_before = sum(counter_before_filter.values())
    for lbl in sorted(INTENT_NAMES.keys()):
        cnt = counter_before_filter.get(lbl, 0)
        print(f"  {lbl:>3}  {INTENT_NAMES[lbl]:<22}  {cnt:>8}  ({100*cnt/max(total_before,1):.2f}%)")
    print(f"  {'':>3}  {'TOTAL':<22}  {total_before:>8}")

    print("\n" + "=" * 60)
    print(f"INTENT LABEL COUNTS  (AFTER filter | FOV={fov_deg}°, range={pcd_range})")
    print("=" * 60)
    total_after = sum(counter_after_filter.values())
    for lbl in sorted(INTENT_NAMES.keys()):
        cnt  = counter_after_filter.get(lbl, 0)
        fcnt = per_label_frame_count.get(lbl, 0)
        print(f"  {lbl:>3}  {INTENT_NAMES[lbl]:<22}  {cnt:>8}  ({100*cnt/max(total_after,1):.2f}%)  "
              f"in {fcnt} frames")
    print(f"  {'':>3}  {'TOTAL':<22}  {total_after:>8}")

    # ── useful derived stats ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MINORITY CLASS INFO  (after filter, excluding label=-1)")
    print("=" * 60)
    valid_counts = {k: v for k, v in counter_after_filter.items() if k != -1}
    if valid_counts:
        max_cnt  = max(valid_counts.values())
        for lbl, cnt in sorted(valid_counts.items(), key=lambda x: x[1]):
            ratio = max_cnt / cnt if cnt > 0 else float("inf")
            print(f"  {lbl:>3}  {INTENT_NAMES[lbl]:<22}  {cnt:>8}  "
                  f"imbalance-ratio vs majority: {ratio:.1f}x")

    # ── -1 reason breakdown ──────────────────────────────────────────────────
    total_neg1_before = counter_before_filter.get(-1, 0)
    print("\n" + "=" * 60)
    print(f"WHY -1?  (before filter, total -1 = {total_neg1_before})")
    print("=" * 60)
    reasons = {
        "Token not in intent_data" : reason_not_in_data,
        "frame_idx at seq end"     : reason_end_of_seq,
        "Current action = 'na'"    : reason_na_action,
        "All future actions = 'na'": reason_all_future_na,
    }
    for name, cnt in reasons.items():
        pct = 100 * cnt / max(total_neg1_before, 1)
        print(f"  {name:<30}  {cnt:>8}  ({pct:.1f}%)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count intent label distribution in NuScenes training data")
    parser.add_argument("--ann_file",     required=True,
                        help="Path to nuscenes_infos_train.pkl (or mini)")
    parser.add_argument("--intent_file",
                        default="/zihan-west-vol/UniAD/data/nuscenes/unified_map_v2/all_scenes_compact.json",
                        help="Path to all_scenes_compact.json")
    parser.add_argument("--point_cloud_range", type=float, nargs=6,
                        default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        metavar=("X_MIN","Y_MIN","Z_MIN","X_MAX","Y_MAX","Z_MAX"),
                        help="Same as ObjectRangeFilterTrack point_cloud_range")
    parser.add_argument("--fov_deg", type=float, default=70.0,
                        help="Full FOV in degrees (same as ObjectRangeFilterTrack, default 70)")
    args = parser.parse_args()
    main(args)
    # python count_intent_labels.py \
    # --ann_file data/infos/nuscenes_infos_temporal_train.pkl \
    # --intent_file /zihan-west-vol/UniAD/data/nuscenes/unified_map_v2/all_scenes_compact.json \
    # --point_cloud_range -51.2 0 -5.0 51.2 51.2 3.0 \
    # --fov_deg 70