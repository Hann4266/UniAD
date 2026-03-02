"""
Filter nuscenes_infos_temporal_train.pkl to only keep frames
belonging to scenes listed in a JSON file of scene tokens.

Usage:
    python filter_pkl_by_scene.py \
        --input_pkl  /zihan-west-vol/UniAD/data/infos/nuscenes_infos_temporal_train.pkl \
        --scene_json /path/to/scene_tokens_train_all_target_scenes.json \
        --output_pkl /zihan-west-vol/UniAD/data/infos/nuscenes_infos_temporal_train_intent.pkl
"""

import argparse
import json
import pickle
from tqdm import tqdm


def main(args):
    print(f"Loading: {args.input_pkl}")
    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    print(f"Loading scene tokens: {args.scene_json}")
    with open(args.scene_json, "r") as f:
        target_scenes = set(json.load(f))
    print(f"  Target scenes: {len(target_scenes)}")

    infos_all = data["infos"]
    print(f"  Total frames before filter: {len(infos_all)}")

    infos_filtered = [
        info for info in tqdm(infos_all, desc="Filtering")
        if info["scene_token"] in target_scenes
    ]
    print(f"  Total frames after  filter: {len(infos_filtered)}")

    # count scenes actually present
    scenes_kept = {info["scene_token"] for info in infos_filtered}
    print(f"  Scenes kept: {len(scenes_kept)} / {len(target_scenes)}")

    # build output, preserve metadata
    data_out = dict(infos=infos_filtered, metadata=data.get("metadata", {}))

    print(f"Saving: {args.output_pkl}")
    with open(args.output_pkl, "wb") as f:
        pickle.dump(data_out, f)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl",  required=True,
                        help="Path to nuscenes_infos_temporal_train.pkl")
    parser.add_argument("--scene_json", required=True,
                        help="Path to scene_tokens JSON file (list of scene tokens)")
    parser.add_argument("--output_pkl", required=True,
                        help="Output filtered pkl path")
    args = parser.parse_args()
    main(args)

    # python filter_pkl.py \
    # --input_pkl  /zihan-west-vol/UniAD/data/infos/nuscenes_infos_temporal_val.pkl \
    # --scene_json ./scene_tokens_val_all_target_scenes.json \
    # --output_pkl /zihan-west-vol/UniAD/data/infos/nuscenes_infos_temporal_val_intent.pkl