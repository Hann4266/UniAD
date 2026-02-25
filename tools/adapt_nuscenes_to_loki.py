#!/usr/bin/env python3
"""
Adapt a nuScenes-trained UniAD checkpoint for LOKI fine-tuning.

The nuScenes model has 10 classes; LOKI has 8. This script:
1. Remaps cls_branches weights from nuScenes class order to LOKI class order
   (6 shared classes are copied; 2 LOKI-only classes get random init)
2. Drops all seg_head (map) weights (LOKI has no map GT)
3. Saves a new checkpoint compatible with the 8-class LOKI model config

Usage:
    python tools/adapt_nuscenes_to_loki.py \
        --src /mnt/storage/UniAD/work_dirs/zihan_nuscenes_weight/epoch_6.pth \
        --dst work_dirs/nuscenes_adapted_for_loki.pth
"""

import argparse
import torch
import copy


# nuScenes class order (10 classes, as in checkpoint)
NUSCENES_CLASSES = [
    "car",                  # 0
    "truck",                # 1
    "construction_vehicle", # 2
    "bus",                  # 3
    "trailer",              # 4
    "barrier",              # 5
    "motorcycle",           # 6
    "bicycle",              # 7
    "pedestrian",           # 8
    "traffic_cone",         # 9
]

# LOKI class order (8 classes, as in base_loki_perception.py)
LOKI_CLASSES = [
    "Pedestrian",   # 0
    "Car",          # 1
    "Bus",          # 2
    "Truck",        # 3
    "Van",          # 4  (mapped to Car in pkl, init from car weights)
    "Motorcyclist", # 5
    "Bicyclist",    # 6
    "Other",        # 7  (no nuScenes equivalent)
]

# Mapping: LOKI class name -> nuScenes class name (lowercase)
# Van is mapped to Car in the LOKI pkl, so we init Van's head from car weights
LOKI_TO_NUSCENES = {
    "Pedestrian":   "pedestrian",
    "Car":          "car",
    "Bus":          "bus",
    "Truck":        "truck",
    "Van":          "car",       # Van mapped to Car in pkl data
    "Motorcyclist": "motorcycle",
    "Bicyclist":    "bicycle",
}

# Build index mapping: loki_idx -> nuscenes_idx (or None for LOKI-only classes)
NUSCENES_IDX = {name: i for i, name in enumerate(NUSCENES_CLASSES)}
LOKI_TO_NUSCENES_IDX = {}
for loki_idx, loki_name in enumerate(LOKI_CLASSES):
    ns_name = LOKI_TO_NUSCENES.get(loki_name)
    if ns_name is not None:
        LOKI_TO_NUSCENES_IDX[loki_idx] = NUSCENES_IDX[ns_name]
    else:
        LOKI_TO_NUSCENES_IDX[loki_idx] = None  # random init


def remap_cls_weight(weight_10, num_loki=8):
    """Remap a [10, ...] weight tensor to [8, ...] using class mapping.

    For shared classes, copy the corresponding row from the nuScenes weight.
    For LOKI-only classes (Van, Other), use Kaiming-style random init.
    """
    out_shape = (num_loki,) + weight_10.shape[1:]
    weight_8 = torch.empty(out_shape, dtype=weight_10.dtype)

    # Initialize all with small random values (for LOKI-only classes)
    torch.nn.init.xavier_uniform_(weight_8.view(num_loki, -1)
                                  if weight_8.dim() > 1
                                  else weight_8.unsqueeze(1)).squeeze()

    for loki_idx in range(num_loki):
        ns_idx = LOKI_TO_NUSCENES_IDX[loki_idx]
        if ns_idx is not None:
            weight_8[loki_idx] = weight_10[ns_idx]

    return weight_8


def remap_cls_bias(bias_10, num_loki=8):
    """Remap a [10] bias tensor to [8] using class mapping.

    For shared classes, copy from nuScenes. For LOKI-only, init to 0.
    """
    bias_8 = torch.zeros(num_loki, dtype=bias_10.dtype)
    for loki_idx in range(num_loki):
        ns_idx = LOKI_TO_NUSCENES_IDX[loki_idx]
        if ns_idx is not None:
            bias_8[loki_idx] = bias_10[ns_idx]
    return bias_8


def adapt_checkpoint(src_path, dst_path):
    print(f"Loading checkpoint: {src_path}")
    ckpt = torch.load(src_path, map_location="cpu")
    sd = ckpt["state_dict"]

    print(f"Original state_dict: {len(sd)} keys")

    # Print class mapping for verification
    print("\nClass mapping (LOKI idx <- nuScenes idx):")
    for loki_idx, loki_name in enumerate(LOKI_CLASSES):
        ns_idx = LOKI_TO_NUSCENES_IDX[loki_idx]
        if ns_idx is not None:
            print(f"  [{loki_idx}] {loki_name} <- [{ns_idx}] {NUSCENES_CLASSES[ns_idx]}")
        else:
            print(f"  [{loki_idx}] {loki_name} <- (random init)")

    new_sd = {}
    dropped_seg = 0
    remapped_cls = 0

    for key, val in sd.items():
        # Drop all seg_head weights (map module, not used in LOKI training)
        if key.startswith("seg_head."):
            dropped_seg += 1
            continue

        # Remap cls_branches final layer (the only class-dependent weights)
        # Pattern: pts_bbox_head.cls_branches.{layer}.6.weight  [10, 256]
        #          pts_bbox_head.cls_branches.{layer}.6.bias    [10]
        if "cls_branches" in key and key.endswith(".6.weight"):
            assert val.shape[0] == 10, f"Expected 10 classes, got {val.shape[0]} for {key}"
            new_sd[key] = remap_cls_weight(val)
            remapped_cls += 1
            print(f"  Remapped {key}: {val.shape} -> {new_sd[key].shape}")
            continue

        if "cls_branches" in key and key.endswith(".6.bias"):
            assert val.shape[0] == 10, f"Expected 10 classes, got {val.shape[0]} for {key}"
            new_sd[key] = remap_cls_bias(val)
            remapped_cls += 1
            print(f"  Remapped {key}: {val.shape} -> {new_sd[key].shape}")
            continue

        # All other weights pass through unchanged
        new_sd[key] = val

    print(f"\nDropped {dropped_seg} seg_head keys")
    print(f"Remapped {remapped_cls} cls_branches keys (10 -> 8 classes)")
    print(f"New state_dict: {len(new_sd)} keys")

    # Save new checkpoint (keep meta, drop optimizer since param groups changed)
    new_ckpt = {
        "state_dict": new_sd,
        "meta": ckpt.get("meta", {}),
    }
    torch.save(new_ckpt, dst_path)
    print(f"\nSaved adapted checkpoint: {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Adapt nuScenes UniAD checkpoint for LOKI fine-tuning")
    parser.add_argument("--src", required=True,
                        help="Path to nuScenes checkpoint (e.g., epoch_6.pth)")
    parser.add_argument("--dst", required=True,
                        help="Output path for adapted checkpoint")
    args = parser.parse_args()

    adapt_checkpoint(args.src, args.dst)


if __name__ == "__main__":
    main()
