"""
Count how many scenes contain at least one agent (after FOV+range filter) with:
  Crossing (2), TURN_RIGHT (3), TURN_LEFT (4),
  LANE_CHANGE_RIGHT (5), LANE_CHANGE_LEFT (6)

Usage:
    python count_scenes_with_intent.py \
        --ann_file /path/to/nuscenes_infos_train.pkl \
        --intent_file /zihan-west-vol/UniAD/data/nuscenes/unified_map_v2/all_scenes_compact.json \
        --point_cloud_range -51.2 0 -5.0 51.2 51.2 3.0 \
        --fov_deg 70
"""

import argparse
import json
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

TARGET_LABELS = {2, 3, 4, 5, 6}
TARGET_NAMES  = {
    2: "Crossing",
    3: "TURN_RIGHT",
    4: "TURN_LEFT",
    5: "LANE_CHANGE_RIGHT",
    6: "LANE_CHANGE_LEFT",
}


def intent_label(index, action_array):
    intent_dic = {
        "Crossing": 2, "TURN_RIGHT": 3, "TURN_LEFT": 4,
        "LANE_CHANGE_RIGHT": 5, "LANE_CHANGE_LEFT": 6,
        "STOPPED": 0, "MOVING": 1, "Stopped": 0, "Moving": 1,
    }
    if index < 0 or index >= len(action_array) - 1:
        return -1
    action = action_array[index]
    end = min(len(action_array), index + 5)
    future_actions = action_array[index + 1:end]
    if len(future_actions) == 0 or all(a == "na" for a in future_actions):
        return -1
    action_set = {"Crossing", "TURN_RIGHT", "TURN_LEFT", "LANE_CHANGE_RIGHT", "LANE_CHANGE_LEFT"}
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
    return -1


def apply_range_filter(centers_xy, pcd_range, fov_deg=70.0):
    x, y = centers_xy[:, 0], centers_xy[:, 1]
    x_min, y_min = float(pcd_range[0]), float(pcd_range[1])
    x_max, y_max = float(pcd_range[3]), float(pcd_range[4])
    tan_half_fov = np.tan(np.deg2rad(fov_deg / 2.0))
    mask_fov = (y > 0) & (np.abs(x) <= y * tan_half_fov)
    mask_x   = (x >= x_min) & (x <= x_max)
    mask_y   = (y >= y_min) & (y <= y_max)
    return mask_fov & mask_x & mask_y


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

    # scene_token -> set of target labels seen (after filter)
    scene_labels     = defaultdict(set)
    all_scene_tokens = set()

    for info in tqdm(data_infos, desc="Processing"):
        frame_idx   = int(info["frame_idx"])
        scene_token = info["scene_token"]
        all_scene_tokens.add(scene_token)

        mask_valid = info.get("valid_flag", info["num_lidar_pts"] > 0)
        gt_boxes   = info["gt_boxes"][mask_valid]
        gt_tokens  = info["gt_ins_tokens"][mask_valid]

        if len(gt_boxes) == 0:
            continue

        # spatial + FOV filter
        keep = apply_range_filter(gt_boxes[:, :2], pcd_range, fov_deg)
        gt_tokens_kept = gt_tokens[keep]

        if len(gt_tokens_kept) == 0:
            continue

        # compute intent for filtered agents only
        scene_dict = intent_data.get(scene_token, {})
        for tok in gt_tokens_kept:
            if tok not in scene_dict:
                continue
            arr = scene_dict[tok]["labels"]
            lbl = intent_label(frame_idx, arr)
            if lbl in TARGET_LABELS:
                scene_labels[scene_token].add(lbl)

    # ── report ───────────────────────────────────────────────────────────────
    total_scenes     = len(all_scene_tokens)
    scenes_with_any  = {s for s, lbls in scene_labels.items() if lbls & TARGET_LABELS}
    scenes_per_label = {
        lbl: {s for s, lbls in scene_labels.items() if lbl in lbls}
        for lbl in TARGET_LABELS
    }

    # ── save scene tokens to JSON ─────────────────────────────────────────────
    # 1) all target scenes
    out_all = args.output_prefix + "_all_target_scenes.json"
    with open(out_all, "w") as f:
        json.dump(sorted(scenes_with_any), f, indent=2)
    print(f"\nSaved: {out_all}  ({len(scenes_with_any)} scenes)")

    # 2) turning scenes: TURN_RIGHT (3) | TURN_LEFT (4)
    scenes_turning = scenes_per_label[3] | scenes_per_label[4]
    out_turning = args.output_prefix + "_turning_scenes.json"
    with open(out_turning, "w") as f:
        json.dump(sorted(scenes_turning), f, indent=2)
    print(f"Saved: {out_turning}  ({len(scenes_turning)} scenes)")

    # 3) lane change scenes: LANE_CHANGE_RIGHT (5) | LANE_CHANGE_LEFT (6)
    scenes_lc = scenes_per_label[5] | scenes_per_label[6]
    out_lc = args.output_prefix + "_lane_change_scenes.json"
    with open(out_lc, "w") as f:
        json.dump(sorted(scenes_lc), f, indent=2)
    print(f"Saved: {out_lc}  ({len(scenes_lc)} scenes)")

    # ── print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"FOV={fov_deg}°  range={pcd_range}")
    print("=" * 60)
    print(f"Total scenes in pkl                        : {total_scenes}")
    print(f"Scenes with ANY target label (after filter): {len(scenes_with_any)}  "
          f"({100*len(scenes_with_any)/max(total_scenes,1):.1f}%)")
    print(f"Scenes with TURNING   (L or R)             : {len(scenes_turning)}  "
          f"({100*len(scenes_turning)/max(total_scenes,1):.1f}%)")
    print(f"Scenes with LANE_CHANGE (L or R)           : {len(scenes_lc)}  "
          f"({100*len(scenes_lc)/max(total_scenes,1):.1f}%)")
    print("-" * 60)
    print("Per-label breakdown (scenes with ≥1 occurrence after filter):")
    for lbl in sorted(TARGET_LABELS):
        n = len(scenes_per_label[lbl])
        print(f"  [{lbl}] {TARGET_NAMES[lbl]:<22}  {n:>5} scenes  "
              f"({100*n/max(total_scenes,1):.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", default="/zihan-west-vol/UniAD/data/infos/nuscenes_infos_temporal_train_old.pkl")
    parser.add_argument("--intent_file",
                        default="/zihan-west-vol/UniAD/data/nuscenes/unified_map_v3/all_scenes_compact_new.json")
    parser.add_argument("--point_cloud_range", type=float, nargs=6,
                        default=[-51.2, 0, -5.0, 51.2, 51.2, 3.0],
                        metavar=("X_MIN","Y_MIN","Z_MIN","X_MAX","Y_MAX","Z_MAX"))
    parser.add_argument("--fov_deg", type=float, default=70.0)
    parser.add_argument("--output_prefix", default="./scene_tokens_train",
                        help="Prefix for output JSON files, e.g. ./scene_tokens")
    args = parser.parse_args()
    main(args)