"""
Count intent label distribution in the training dataset,
applying the same FOV + BEV range filter as ObjectRangeFilterTrack.
WHY -1 breakdown is computed AFTER the spatial/FOV filter.
Counts are broken down by ALL / VEHICLE / PEDESTRIAN.
"""

import argparse
import json
import pickle
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

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

VEHICLE_CLASSES = {
    "car", "truck", "trailer", "bus",
    "construction_vehicle", "motorcycle", "bicycle",
}
PEDESTRIAN_CLASSES = {"pedestrian"}


def agent_type(name):
    if name in VEHICLE_CLASSES:
        return "vehicle"
    if name in PEDESTRIAN_CLASSES:
        return "pedestrian"
    return "other"


def intent_label(index, action_array):
    intent_dic = {
        "Crossing": 2, "TURN_RIGHT": 3, "TURN_LEFT": 4,
        "LANE_CHANGE_RIGHT": 5, "LANE_CHANGE_LEFT": 6,
        "STOPPED": 0, "MOVING": 1, "Stopped": 0, "Moving": 1,
    }
    if index < 0 or index >= len(action_array) - 1:
        return -1, "end_of_seq"

    action = action_array[index]
    end = min(len(action_array), index + 5)
    future_actions = action_array[index + 1:end]

    if len(future_actions) == 0 or all(a == "na" for a in future_actions):
        return -1, "all_future_na"

    action_set = {"Crossing", "TURN_RIGHT", "TURN_LEFT", "LANE_CHANGE_RIGHT", "LANE_CHANGE_LEFT"}

    if action in action_set:
        for next_action in future_actions:
            if next_action != action and next_action != "na":
                return intent_dic[next_action], None
        return intent_dic[action], None
    elif action in ("STOPPED", "Stopped"):
        next_moving = False
        for next_action in future_actions:
            if next_action in action_set:
                return intent_dic[next_action], None
            elif next_action in ("MOVING", "Moving"):
                next_moving = True
        return (intent_dic["MOVING"] if next_moving else intent_dic["STOPPED"]), None
    elif action in ("MOVING", "Moving"):
        next_stop = False
        for next_action in future_actions:
            if next_action in action_set:
                return intent_dic[next_action], None
            elif next_action in ("STOPPED", "Stopped"):
                next_stop = True
        return (intent_dic["STOPPED"] if next_stop else intent_dic["MOVING"]), None
    elif action == "na":
        for next_action in future_actions:
            if next_action != "na":
                return intent_dic[next_action], None
        return -1, "all_future_na"
    else:
        return -1, "unknown_action"


def apply_range_filter(centers_xy, pcd_range, fov_deg=70.0):
    x, y = centers_xy[:, 0], centers_xy[:, 1]
    x_min, y_min = float(pcd_range[0]), float(pcd_range[1])
    x_max, y_max = float(pcd_range[3]), float(pcd_range[4])
    tan_half_fov = np.tan(np.deg2rad(fov_deg / 2.0))
    mask_fov = (y > 0) & (np.abs(x) <= y * tan_half_fov)
    mask_x   = (x >= x_min) & (x <= x_max)
    mask_y   = (y >= y_min) & (y <= y_max)
    return mask_fov & mask_x & mask_y


def print_report(title, counter, why_neg1, label_names):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)
    total = sum(counter.values())
    for lbl in sorted(label_names.keys()):
        cnt = counter.get(lbl, 0)
        print(f"  {lbl:>3}  {label_names[lbl]:<22}  {cnt:>8}  ({100*cnt/max(total,1):.2f}%)")
    print(f"  {'':>3}  {'TOTAL':<22}  {total:>8}")

    total_neg1 = counter.get(-1, 0)
    if total_neg1 > 0 and why_neg1:
        print(f"\n  WHY -1? (total = {total_neg1})")
        for name, cnt in why_neg1.items():
            print(f"    {name:<30}  {cnt:>8}  ({100*cnt/max(total_neg1,1):.1f}%)")


def main(args):
    print(f"Loading: {args.ann_file}")
    with open(args.ann_file, "rb") as f:
        data = pickle.load(f)
    data_infos = sorted(data["infos"], key=lambda e: e["timestamp"])
    print(f"  Total frames: {len(data_infos)}")

    print(f"Loading: {args.intent_file}")
    with open(args.intent_file, "r") as f:
        intent_data = json.load(f)

    pcd_range = args.point_cloud_range
    fov_deg   = args.fov_deg

    # counters: all / vehicle / pedestrian
    counters = {
        "all":        Counter(),
        "vehicle":    Counter(),
        "pedestrian": Counter(),
    }
    why_neg1 = {
        "all":        defaultdict(int),
        "vehicle":    defaultdict(int),
        "pedestrian": defaultdict(int),
    }

    for info in tqdm(data_infos, desc="Processing"):
        frame_idx   = int(info["frame_idx"])
        scene_token = info["scene_token"]

        mask_valid = info["valid_flag"] if "valid_flag" in info else info["num_lidar_pts"] > 0
        gt_boxes   = info["gt_boxes"][mask_valid]
        gt_tokens  = info["gt_ins_tokens"][mask_valid]
        gt_names   = info["gt_names"][mask_valid]

        if len(gt_boxes) == 0:
            continue

        # Step 1: FOV + range filter
        keep = apply_range_filter(gt_boxes[:, :2], pcd_range, fov_deg)
        gt_boxes  = gt_boxes[keep]
        gt_tokens = gt_tokens[keep]
        gt_names  = gt_names[keep]

        if len(gt_tokens) == 0:
            continue

        # Step 2: compute intent label for filtered agents
        scene_dict = intent_data.get(scene_token, {})
        for tok, name in zip(gt_tokens, gt_names):
            atype = agent_type(name)

            if tok not in scene_dict:
                lbl, reason = -1, "not_in_data"
            else:
                arr = scene_dict[tok]["labels"]
                lbl, reason = intent_label(frame_idx, arr)

            # Step 3: count
            counters["all"][lbl] += 1
            if atype in counters:
                counters[atype][lbl] += 1

            # Step 4: WHY -1
            if lbl == -1:
                why_neg1["all"][reason] += 1
                if atype in why_neg1:
                    why_neg1[atype][reason] += 1

    # ── report ───────────────────────────────────────────────────────────────
    for key, title in [
        ("all",        f"ALL AGENTS  (FOV={fov_deg}°, range={pcd_range})"),
        ("vehicle",    "VEHICLES ONLY"),
        ("pedestrian", "PEDESTRIANS ONLY"),
    ]:
        print_report(title, counters[key], why_neg1[key], INTENT_NAMES)

    # minority class imbalance (all, excluding -1)
    print("\n" + "=" * 65)
    print("MINORITY CLASS IMBALANCE  (all agents, after filter, excl. -1)")
    print("=" * 65)
    valid_counts = {k: v for k, v in counters["all"].items() if k != -1}
    if valid_counts:
        max_cnt = max(valid_counts.values())
        for lbl, cnt in sorted(valid_counts.items(), key=lambda x: x[1]):
            ratio = max_cnt / cnt if cnt > 0 else float("inf")
            print(f"  {lbl:>3}  {INTENT_NAMES[lbl]:<22}  {cnt:>8}  "
                  f"imbalance vs majority: {ratio:.1f}x")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file",
                        default="/zihan-west-vol/UniAD/data/infos/nuscenes_infos_temporal_train.pkl")
    parser.add_argument("--intent_file",
                        default="/zihan-west-vol/UniAD/data/nuscenes/unified_map_v3/all_scenes_compact_new.json")
    parser.add_argument("--point_cloud_range", type=float, nargs=6,
                        default=[-51.2, 0, -5.0, 51.2, 51.2, 3.0],
                        metavar=("X_MIN","Y_MIN","Z_MIN","X_MAX","Y_MAX","Z_MAX"))
    parser.add_argument("--fov_deg", type=float, default=70.0)
    args = parser.parse_args()
    main(args)