"""
Compare LokiE2EDataset vs NuScenesE2EDataset output structures using REAL data.

Builds both datasets, runs each stage, and prints a side-by-side comparison
of shapes, dtypes, keys, and values so you can spot mismatches.

Requirements:
    - LOKI data at /mnt/storage/loki_data/
    - NuScenes data at /mnt/cogrob/nuScenes/
    - Symlink: /mnt/storage/UniAD/data/nuscenes -> /mnt/cogrob/nuScenes

Usage:
    cd /mnt/storage/UniAD
    PYTHONPATH=$(pwd):$PYTHONPATH python tools/debug_compare_dataloaders.py

Options:
    --loki-only     Only run LOKI dataset (skip NuScenes)
    --nusc-only     Only run NuScenes dataset (skip LOKI)
    --stage N       Only run a specific stage (1-4)
    --index N       Use a specific sample index (default: auto-pick)
"""

import sys
import os
import copy
import argparse
import numpy as np
import torch

sys.path.insert(0, os.getcwd())

# Register all custom datasets/pipelines from the plugin module
# (same as what train.py does via the plugin_dir config)
import importlib
importlib.import_module("projects.mmdet3d_plugin")

from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.pipelines import to_tensor


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def describe(val, depth=0):
    """Return a concise string describing shape/dtype/type of a value."""
    indent = "  " * depth
    if isinstance(val, DC):
        inner = describe(val.data, depth)
        return f"DC(cpu_only={val.cpu_only}, stack={val.stack}) -> {inner}"
    elif isinstance(val, torch.Tensor):
        return f"Tensor {list(val.shape)} {val.dtype}"
    elif isinstance(val, np.ndarray):
        return f"ndarray {list(val.shape)} {val.dtype}"
    elif isinstance(val, LiDARInstance3DBoxes):
        t = val.tensor
        return f"LiDARInstance3DBoxes {list(t.shape)} {t.dtype}"
    elif isinstance(val, (list, tuple)):
        tag = "list" if isinstance(val, list) else "tuple"
        if len(val) == 0:
            return f"{tag} len=0"
        first = describe(val[0], depth)
        return f"{tag} len={len(val)}, [0]: {first}"
    elif isinstance(val, dict):
        return f"dict keys={sorted(val.keys())}"
    elif isinstance(val, (int, float, bool, np.integer, np.floating)):
        return f"{type(val).__name__}={val}"
    elif isinstance(val, str):
        return f"str='{val[:60]}'"
    elif val is None:
        return "None"
    else:
        return f"{type(val).__name__}"


def sample_values(val, max_items=5):
    """Return a small sample of actual values for sanity checking."""
    if isinstance(val, DC):
        return sample_values(val.data, max_items)
    elif isinstance(val, torch.Tensor):
        flat = val.flatten()[:max_items]
        return f"[{', '.join(f'{x:.4f}' for x in flat.tolist())}]"
    elif isinstance(val, np.ndarray):
        flat = val.flatten()[:max_items]
        try:
            return f"[{', '.join(f'{x:.4f}' for x in flat)}]"
        except (TypeError, ValueError):
            return f"[{', '.join(str(x) for x in flat)}]"
    elif isinstance(val, LiDARInstance3DBoxes):
        return sample_values(val.tensor, max_items)
    elif isinstance(val, (list, tuple)):
        if len(val) == 0:
            return "[]"
        return f"[0]: {sample_values(val[0], max_items)}"
    elif isinstance(val, (int, float, np.integer, np.floating)):
        return str(val)
    else:
        return ""


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_comparison(nusc_dict, loki_dict, stage_name):
    """Print a side-by-side key comparison of two output dicts."""
    nusc_keys = set(nusc_dict.keys()) if nusc_dict else set()
    loki_keys = set(loki_dict.keys()) if loki_dict else set()
    all_keys = sorted(nusc_keys | loki_keys)

    print(f"\n  {'KEY':<30} {'NUSCENES':<45} {'LOKI':<45} {'MATCH'}")
    print(f"  {'-'*30} {'-'*45} {'-'*45} {'-'*5}")

    mismatches = []
    for k in all_keys:
        n_desc = describe(nusc_dict[k]) if k in nusc_dict else "---MISSING---"
        l_desc = describe(loki_dict[k]) if k in loki_dict else "---MISSING---"

        # Determine match status
        if k not in nusc_dict or k not in loki_dict:
            match = "MISS"
            mismatches.append(k)
        elif _shapes_match(nusc_dict.get(k), loki_dict.get(k)):
            match = "OK"
        else:
            match = "DIFF"
            mismatches.append(k)

        # Truncate descriptions for readability
        n_desc = n_desc[:44]
        l_desc = l_desc[:44]
        print(f"  {k:<30} {n_desc:<45} {l_desc:<45} {match}")

    if mismatches:
        print(f"\n  MISMATCHES ({len(mismatches)}):")
        for k in mismatches:
            print(f"    '{k}':")
            if k in nusc_dict:
                print(f"      NuScenes: {describe(nusc_dict[k])}")
                print(f"        values: {sample_values(nusc_dict[k])}")
            else:
                print(f"      NuScenes: MISSING")
            if k in loki_dict:
                print(f"      LOKI:     {describe(loki_dict[k])}")
                print(f"        values: {sample_values(loki_dict[k])}")
            else:
                print(f"      LOKI:     MISSING")
    else:
        print(f"\n  All keys match in structure!")


def _shapes_match(a, b):
    """Check if two values have compatible shapes/types (ignoring num_cams diff)."""
    ta = type(a)
    tb = type(b)

    # Unwrap DC
    if isinstance(a, DC):
        a = a.data
    if isinstance(b, DC):
        b = b.data

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # Shapes must match in ndim, allow different sizes
        return a.dim() == b.dim() and a.dtype == b.dtype
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.ndim == b.ndim and a.dtype == b.dtype
    elif isinstance(a, LiDARInstance3DBoxes) and isinstance(b, LiDARInstance3DBoxes):
        return a.tensor.shape[-1] == b.tensor.shape[-1]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) == 0 or len(b) == 0:
            return True
        return type(a[0]) == type(b[0])
    elif type(a) == type(b):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
#  Dataset builders
# ─────────────────────────────────────────────────────────────────────

def ensure_nuscenes_symlink():
    """Create data/nuscenes symlink if missing."""
    link_path = os.path.join(os.getcwd(), "data", "nuscenes")
    target = "/mnt/cogrob/nuScenes"
    if not os.path.exists(link_path):
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        os.symlink(target, link_path)
        print(f"  Created symlink: {link_path} -> {target}")
    else:
        print(f"  Symlink exists: {link_path}")


def build_nuscenes_dataset():
    """Build NuScenesE2EDataset from the stage1 config."""
    import pickle as _pickle
    from mmdet.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset

    ensure_nuscenes_symlink()

    # Monkey-patch load_annotations to handle plain string paths
    # (original code does ann_file.name which only works with pathlib.Path)
    _orig_load = NuScenesE2EDataset.load_annotations

    def _patched_load(self, ann_file):
        filepath = ann_file.name if hasattr(ann_file, 'name') else str(ann_file)
        data = _pickle.loads(self.file_client.get(filepath))
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    NuScenesE2EDataset.load_annotations = _patched_load

    cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")

    # Override paths to point to real NuScenes data
    cfg.data_root = "data/nuscenes/"
    cfg.data.train.data_root = "data/nuscenes/"
    abs_ann = os.path.join(os.getcwd(), "data", "infos", "nuscenes_infos_temporal_train.pkl")
    cfg.data.train.ann_file = abs_ann

    print("  Building NuScenesE2EDataset (this loads NuScenes SDK + maps, may take a minute)...")
    try:
        ds = build_dataset(cfg.data.train)
    finally:
        # Restore original method
        NuScenesE2EDataset.load_annotations = _orig_load
    return ds, cfg


def build_loki_dataset():
    """Build LokiE2EDataset from the LOKI config."""
    from mmdet.datasets import build_dataset

    cfg = Config.fromfile("projects/configs/loki/base_loki_perception.py")
    ds = build_dataset(cfg.data.train)
    return ds, cfg


def find_valid_index(ds, queue_length):
    """Find an index with enough temporal context in the same scene."""
    for idx in range(queue_length, min(len(ds), 500)):
        info = ds.data_infos[idx]
        first = ds.data_infos[idx - queue_length + 1]
        if info['scene_token'] == first['scene_token']:
            return idx
    return None


# ─────────────────────────────────────────────────────────────────────
#  Stage runners
# ─────────────────────────────────────────────────────────────────────

def run_stage1(nusc_ds, loki_ds, nusc_idx, loki_idx):
    """Stage 1: get_ann_info()"""
    print_section("STAGE 1: get_ann_info()")

    nusc_ann = nusc_ds.get_ann_info(nusc_idx) if nusc_ds else None
    loki_ann = loki_ds.get_ann_info(loki_idx) if loki_ds else None

    if nusc_ann and loki_ann:
        print_comparison(nusc_ann, loki_ann, "get_ann_info")
    elif nusc_ann:
        print("\n  NuScenes get_ann_info output:")
        for k in sorted(nusc_ann.keys()):
            print(f"    '{k}': {describe(nusc_ann[k])}")
            print(f"      values: {sample_values(nusc_ann[k])}")
    elif loki_ann:
        print("\n  LOKI get_ann_info output:")
        for k in sorted(loki_ann.keys()):
            print(f"    '{k}': {describe(loki_ann[k])}")
            print(f"      values: {sample_values(loki_ann[k])}")

    return nusc_ann, loki_ann


def run_stage2(nusc_ds, loki_ds, nusc_idx, loki_idx):
    """Stage 2: get_data_info()"""
    print_section("STAGE 2: get_data_info()")

    nusc_data = nusc_ds.get_data_info(nusc_idx) if nusc_ds else None
    loki_data = loki_ds.get_data_info(loki_idx) if loki_ds else None

    if nusc_data and loki_data:
        print_comparison(nusc_data, loki_data, "get_data_info")
    elif nusc_data:
        print("\n  NuScenes get_data_info output:")
        for k in sorted(nusc_data.keys()):
            print(f"    '{k}': {describe(nusc_data[k])}")
    elif loki_data:
        print("\n  LOKI get_data_info output:")
        for k in sorted(loki_data.keys()):
            print(f"    '{k}': {describe(loki_data[k])}")

    # Detail critical fields
    print("\n  Critical field value comparison:")
    critical = ['can_bus', 'l2g_r_mat', 'l2g_t', 'ego2global_rotation',
                'ego2global_translation', 'lidar2img', 'img_filename', 'timestamp']
    for field in critical:
        print(f"\n  >> {field}:")
        if nusc_data and field in nusc_data:
            print(f"    NuScenes: {describe(nusc_data[field])}")
            print(f"      values: {sample_values(nusc_data[field])}")
        elif nusc_data:
            print(f"    NuScenes: MISSING")
        if loki_data and field in loki_data:
            print(f"    LOKI:     {describe(loki_data[field])}")
            print(f"      values: {sample_values(loki_data[field])}")
        elif loki_data:
            print(f"    LOKI:     MISSING")

    return nusc_data, loki_data


def run_stage3(nusc_ds, loki_ds, nusc_idx, loki_idx):
    """Stage 3: Single frame after pipeline."""
    print_section("STAGE 3: Single frame after pipeline")

    nusc_piped = None
    loki_piped = None

    if nusc_ds:
        try:
            input_dict = nusc_ds.get_data_info(nusc_idx)
            nusc_ds.pre_pipeline(input_dict)
            nusc_piped = nusc_ds.pipeline(copy.deepcopy(input_dict))
            if nusc_piped is None:
                print("  NuScenes pipeline returned None!")
        except Exception as e:
            print(f"  NuScenes pipeline error: {e}")

    if loki_ds:
        try:
            input_dict = loki_ds.get_data_info(loki_idx)
            loki_ds.pre_pipeline(input_dict)
            loki_piped = loki_ds.pipeline(copy.deepcopy(input_dict))
            if loki_piped is None:
                print("  LOKI pipeline returned None!")
        except Exception as e:
            print(f"  LOKI pipeline error: {e}")

    if nusc_piped and loki_piped:
        print_comparison(nusc_piped, loki_piped, "post-pipeline single frame")
    elif nusc_piped:
        print("\n  NuScenes pipeline output:")
        for k in sorted(nusc_piped.keys()):
            print(f"    '{k}': {describe(nusc_piped[k])}")
    elif loki_piped:
        print("\n  LOKI pipeline output:")
        for k in sorted(loki_piped.keys()):
            print(f"    '{k}': {describe(loki_piped[k])}")

    return nusc_piped, loki_piped


def run_stage4(nusc_ds, loki_ds, nusc_idx, loki_idx):
    """Stage 4: prepare_train_data() -> union2one()."""
    print_section("STAGE 4: prepare_train_data() -> union2one()")

    nusc_result = None
    loki_result = None

    if nusc_ds:
        try:
            nusc_result = nusc_ds.prepare_train_data(nusc_idx)
            if nusc_result is None:
                print("  NuScenes prepare_train_data returned None!")
        except Exception as e:
            print(f"  NuScenes prepare_train_data error: {e}")

    if loki_ds:
        try:
            loki_result = loki_ds.prepare_train_data(loki_idx)
            if loki_result is None:
                print("  LOKI prepare_train_data returned None!")
        except Exception as e:
            print(f"  LOKI prepare_train_data error: {e}")

    if nusc_result and loki_result:
        print_comparison(nusc_result, loki_result, "union2one final output")
    elif nusc_result:
        print("\n  NuScenes union2one output:")
        for k in sorted(nusc_result.keys()):
            print(f"    '{k}': {describe(nusc_result[k])}")
    elif loki_result:
        print("\n  LOKI union2one output:")
        for k in sorted(loki_result.keys()):
            print(f"    '{k}': {describe(loki_result[k])}")

    # Deep inspection of img and img_metas
    for name, result in [("NuScenes", nusc_result), ("LOKI", loki_result)]:
        if result is None:
            continue
        print(f"\n  {name} detailed inspection:")

        if 'img' in result:
            img = result['img']
            if isinstance(img, DC):
                t = img.data
                print(f"    img: {list(t.shape)} (queue, num_cams, C, H, W)")

        if 'img_metas' in result:
            metas = result['img_metas']
            if isinstance(metas, DC):
                md = metas.data
                print(f"    img_metas: {len(md)} frames")
                for fid, meta in md.items():
                    cb = meta.get('can_bus', None)
                    cb_info = ""
                    if cb is not None:
                        shape = cb.shape if hasattr(cb, 'shape') else len(cb)
                        cb_info = f"shape={shape}, [:3]={cb[:3]}, [-1]={cb[-1]}"
                    l2i = meta.get('lidar2img', None)
                    l2i_info = ""
                    if l2i is not None:
                        if isinstance(l2i, list):
                            l2i_info = f"list len={len(l2i)}, [0].shape={np.array(l2i[0]).shape}"
                        else:
                            l2i_info = describe(l2i)
                    print(f"      frame {fid}: prev_bev={meta.get('prev_bev', '?')}, "
                          f"can_bus=({cb_info}), "
                          f"lidar2img=({l2i_info}), "
                          f"img_shape={meta.get('img_shape', '?')}")

    return nusc_result, loki_result


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare LOKI vs NuScenes dataloaders")
    parser.add_argument("--loki-only", action="store_true", help="Only run LOKI dataset")
    parser.add_argument("--nusc-only", action="store_true", help="Only run NuScenes dataset")
    parser.add_argument("--stage", type=int, default=0, help="Run specific stage (1-4), 0=all")
    parser.add_argument("--index", type=int, default=-1, help="Specific sample index to use")
    args = parser.parse_args()

    run_nusc = not args.loki_only
    run_loki = not args.nusc_only

    nusc_ds = None
    loki_ds = None
    nusc_idx = None
    loki_idx = None

    # ── Build datasets ──
    if run_nusc:
        print_section("Building NuScenes dataset")
        try:
            nusc_ds, nusc_cfg = build_nuscenes_dataset()
            print(f"  NuScenes loaded: {len(nusc_ds)} samples")
            print(f"  CLASSES: {nusc_ds.CLASSES}")
            nusc_idx = find_valid_index(nusc_ds, nusc_ds.queue_length)
            if args.index >= 0:
                nusc_idx = args.index
            print(f"  Using index: {nusc_idx}")
            if nusc_idx:
                info = nusc_ds.data_infos[nusc_idx]
                print(f"  scene_token: {info['scene_token']}")
                print(f"  frame_idx: {info['frame_idx']}")
                print(f"  timestamp: {info['timestamp']}")
        except Exception as e:
            print(f"  Failed to build NuScenes dataset: {e}")
            import traceback; traceback.print_exc()
            nusc_ds = None

    if run_loki:
        print_section("Building LOKI dataset")
        try:
            loki_ds, loki_cfg = build_loki_dataset()
            print(f"  LOKI loaded: {len(loki_ds)} samples")
            print(f"  CLASSES: {loki_ds.CLASSES}")
            loki_idx = find_valid_index(loki_ds, loki_ds.queue_length)
            if args.index >= 0:
                loki_idx = args.index
            print(f"  Using index: {loki_idx}")
            if loki_idx:
                info = loki_ds.data_infos[loki_idx]
                print(f"  scene_token: {info['scene_token']}")
                print(f"  frame_idx: {info['frame_idx']}")
                print(f"  timestamp: {info.get('timestamp', 'N/A')}")
        except Exception as e:
            print(f"  Failed to build LOKI dataset: {e}")
            import traceback; traceback.print_exc()
            loki_ds = None

    if nusc_ds is None and loki_ds is None:
        print("\nERROR: No datasets could be built!")
        return

    if (nusc_ds and nusc_idx is None) or (loki_ds and loki_idx is None):
        print("\nERROR: Could not find valid sample indices!")
        return

    # ── Run stages ──
    stages = [1, 2, 3, 4] if args.stage == 0 else [args.stage]

    if 1 in stages:
        run_stage1(nusc_ds, loki_ds, nusc_idx, loki_idx)
    if 2 in stages:
        run_stage2(nusc_ds, loki_ds, nusc_idx, loki_idx)
    if 3 in stages:
        run_stage3(nusc_ds, loki_ds, nusc_idx, loki_idx)
    if 4 in stages:
        run_stage4(nusc_ds, loki_ds, nusc_idx, loki_idx)

    print_section("COMPARISON COMPLETE")
    print("Key differences to expect (by design):")
    print("  - img: NuScenes=[Q,6,3,H,W] vs LOKI=[Q,1,3,H,W] (single camera)")
    print("  - lidar2img: NuScenes=list of 6 vs LOKI=list of 1")
    print("  - ego2global_rotation: NuScenes=quaternion[4] vs LOKI=matrix[3,3]")
    print("  - NuScenes has pts_filename, sweeps, prev_idx, next_idx (LOKI doesn't)")
    print("  - LOKI trajectories are zero-filled (no traj_api)")
    print("  - LOKI map labels are empty (no HD map)")


if __name__ == "__main__":
    main()
