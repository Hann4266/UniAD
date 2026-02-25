"""
Visualize map predictions from zero-shot inference of nuScenes model on LOKI data.

Usage:
    python tools/visualize_map_predictions.py \
        --results work_dirs/nuscenes_on_loki/results_zeroshot.pkl \
        --loki-infos data/infos/loki_infos_val.pkl \
        --data-root /mnt/storage/loki_data/ \
        --out-dir work_dirs/nuscenes_on_loki/vis_map_zeroshot \
        --num-samples 30
"""

import argparse
import os
import pickle
import numpy as np
import cv2


# BEV config (must match inference config)
BEV_H, BEV_W = 100, 200
PC_RANGE = [-51.2, 0, -5.0, 51.2, 51.2, 3.0]  # x_min, y_min, z_min, x_max, y_max, z_max

# Colors for map elements (BGR for OpenCV)
COLORS = {
    'drivable': (180, 130, 70),    # blue-ish
    'divider': (0, 0, 255),        # red
    'crossing': (0, 255, 0),       # green
    'contour': (255, 165, 0),      # orange
}


def make_bev_map_image(pts_bbox, bev_h=BEV_H, bev_w=BEV_W):
    """Render map predictions onto a BEV image (all layers overlapped).

    Args:
        pts_bbox: dict with 'drivable' (H,W bool), 'lane' (3,H,W int)
        bev_h, bev_w: BEV grid dimensions

    Returns:
        RGB image (H, W, 3) uint8
    """
    canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    # Drivable area (background layer)
    drivable = pts_bbox['drivable']
    if hasattr(drivable, 'cpu'):
        drivable = drivable.cpu().numpy()
    drivable = drivable.astype(bool)
    canvas[drivable] = COLORS['drivable']

    # Lane lines (foreground layers)
    lane = pts_bbox['lane']
    if hasattr(lane, 'cpu'):
        lane = lane.cpu().numpy()

    lane_names = ['divider', 'crossing', 'contour']
    for i, name in enumerate(lane_names):
        mask = lane[i].astype(bool)
        canvas[mask] = COLORS[name]

    return canvas


def make_bev_separate(pts_bbox, bev_h=BEV_H, bev_w=BEV_W):
    """Render each map layer as a separate BEV image.

    Returns:
        dict mapping layer name -> (H, W, 3) uint8 image
    """
    layers = {}

    drivable = pts_bbox['drivable']
    if hasattr(drivable, 'cpu'):
        drivable = drivable.cpu().numpy()
    drivable = drivable.astype(bool)
    canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    canvas[drivable] = COLORS['drivable']
    layers['drivable'] = canvas

    lane = pts_bbox['lane']
    if hasattr(lane, 'cpu'):
        lane = lane.cpu().numpy()

    lane_names = ['divider', 'crossing', 'contour']
    for i, name in enumerate(lane_names):
        canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
        mask = lane[i].astype(bool)
        canvas[mask] = COLORS[name]
        layers[name] = canvas

    return layers


def make_combined_figure(camera_img, pts_bbox, token):
    """Create a figure: camera image on top, 4 separate BEV maps in a row below.

    Layout:
        [         camera image         ]
        [drivable][divider][crossing][contour]

    Args:
        camera_img: (H, W, 3) BGR camera image
        pts_bbox: dict with 'drivable' and 'lane'
        token: sample token string

    Returns:
        Combined image uint8
    """
    cam_h, cam_w = camera_img.shape[:2]

    layers = make_bev_separate(pts_bbox)
    layer_names = ['drivable', 'divider', 'crossing', 'contour']

    # Scale each BEV panel so 4 panels side-by-side match camera width
    panel_w = cam_w // 4
    panel_h = int(panel_w * BEV_H / BEV_W)

    panels = []
    for name in layer_names:
        bev = layers[name]
        scaled = cv2.resize(bev, (panel_w, panel_h),
                            interpolation=cv2.INTER_NEAREST)
        # Ego marker at x-axis midpoint (x=0 in BEV -> column BEV_W/2)
        ego_x = panel_w // 2
        # Dashed vertical line
        dash_len = 6
        for y in range(0, panel_h, dash_len * 2):
            y_end = min(y + dash_len, panel_h)
            cv2.line(scaled, (ego_x, y), (ego_x, y_end), (0, 255, 255), 1)
        # Small triangle at bottom to mark ego position
        tri_sz = 5
        pts = np.array([
            [ego_x, panel_h - 1],
            [ego_x - tri_sz, panel_h - 1 - tri_sz * 2],
            [ego_x + tri_sz, panel_h - 1 - tri_sz * 2],
        ])
        cv2.fillPoly(scaled, [pts], (0, 255, 255))
        # Add label
        cv2.putText(scaled, name, (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Color swatch next to label
        cv2.rectangle(scaled, (panel_w - 20, 6), (panel_w - 6, 18),
                      COLORS[name], -1)
        panels.append(scaled)

    # Add 1px white border between panels
    separator = np.full((panel_h, 1, 3), 255, dtype=np.uint8)
    bev_row = panels[0]
    for p in panels[1:]:
        bev_row = np.concatenate([bev_row, separator, p], axis=1)

    # Pad or crop bev_row to match camera width
    if bev_row.shape[1] < cam_w:
        pad = np.zeros((panel_h, cam_w - bev_row.shape[1], 3), dtype=np.uint8)
        bev_row = np.concatenate([bev_row, pad], axis=1)
    else:
        bev_row = bev_row[:, :cam_w]

    # Thin white separator between camera and BEV row
    hsep = np.full((2, cam_w, 3), 255, dtype=np.uint8)

    combined = np.concatenate([camera_img, hsep, bev_row], axis=0)

    # Title on camera image
    cv2.putText(combined, f'Token: {token}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return combined


def compute_map_stats(results):
    """Compute aggregate statistics about map predictions."""
    n_total = len(results)
    n_has_drivable = 0
    n_has_lane = 0
    drivable_areas = []
    lane_areas = []

    for r in results:
        if 'pts_bbox' not in r:
            continue
        pb = r['pts_bbox']

        drv = pb['drivable']
        if hasattr(drv, 'cpu'):
            drv = drv.cpu().numpy()
        drv_area = drv.astype(bool).sum()
        drivable_areas.append(drv_area)
        if drv_area > 0:
            n_has_drivable += 1

        lane = pb['lane']
        if hasattr(lane, 'cpu'):
            lane = lane.cpu().numpy()
        lane_area = (lane.sum(0) > 0).sum()
        lane_areas.append(lane_area)
        if lane_area > 0:
            n_has_lane += 1

    total_pixels = BEV_H * BEV_W
    print(f"\n{'='*60}")
    print(f"Map Prediction Statistics ({n_total} samples)")
    print(f"{'='*60}")
    print(f"Samples with non-zero drivable area: {n_has_drivable}/{n_total} "
          f"({100*n_has_drivable/n_total:.1f}%)")
    print(f"Samples with non-zero lane lines:    {n_has_lane}/{n_total} "
          f"({100*n_has_lane/n_total:.1f}%)")
    if drivable_areas:
        print(f"Drivable area (pixels): mean={np.mean(drivable_areas):.0f}, "
              f"median={np.median(drivable_areas):.0f}, "
              f"max={np.max(drivable_areas):.0f} / {total_pixels}")
    if lane_areas:
        print(f"Lane area (pixels):     mean={np.mean(lane_areas):.0f}, "
              f"median={np.median(lane_areas):.0f}, "
              f"max={np.max(lane_areas):.0f} / {total_pixels}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize map predictions')
    parser.add_argument('--results', required=True, help='Path to results pkl')
    parser.add_argument('--loki-infos', default='data/infos/loki_infos_val.pkl',
                        help='Path to LOKI infos pkl')
    parser.add_argument('--data-root', default='/mnt/storage/loki_data/',
                        help='LOKI data root')
    parser.add_argument('--out-dir', default='work_dirs/nuscenes_on_loki/vis_map_zeroshot',
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=30,
                        help='Number of samples to visualize')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, no visualization')
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results}...")
    with open(args.results, 'rb') as f:
        data = pickle.load(f)
    results = data['bbox_results']
    print(f"Loaded {len(results)} results")

    # Check if pts_bbox exists
    if 'pts_bbox' not in results[0]:
        print("ERROR: Results do not contain 'pts_bbox' (map predictions).")
        print("The seg_head outputs were popped. Re-run inference with the fix")
        print("that keeps pts_bbox in uniad_e2e.py forward_test.")
        return

    # Print statistics
    compute_map_stats(results)

    if args.stats_only:
        return

    # Load LOKI infos for image paths
    print(f"Loading LOKI infos from {args.loki_infos}...")
    with open(args.loki_infos, 'rb') as f:
        loki_data = pickle.load(f)
    infos = loki_data['infos']
    print(f"Loaded {len(infos)} LOKI infos")

    # Build token -> info mapping
    token_to_info = {info['token']: info for info in infos}

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Sample indices evenly across the dataset
    n_samples = min(args.num_samples, len(results))
    indices = np.linspace(0, len(results) - 1, n_samples, dtype=int)

    print(f"Visualizing {n_samples} samples...")
    for idx in indices:
        r = results[idx]
        token = r['token']

        # Load camera image
        info = token_to_info.get(token)
        if info is None:
            print(f"  Warning: token {token} not found in LOKI infos, skipping")
            continue

        img_path = info.get('img_filename', info.get('cams', {}).get('CAM_FRONT', {}).get('data_path', ''))
        if isinstance(img_path, list):
            img_path = img_path[0]
        if not os.path.isabs(img_path):
            img_path = os.path.join(args.data_root, img_path)

        if os.path.exists(img_path):
            camera_img = cv2.imread(img_path)
            if camera_img is not None:
                # Resize to reasonable display size
                target_h = 450
                scale = target_h / camera_img.shape[0]
                camera_img = cv2.resize(camera_img,
                                        (int(camera_img.shape[1] * scale), target_h))
            else:
                camera_img = np.zeros((450, 800, 3), dtype=np.uint8)
        else:
            print(f"  Warning: image not found: {img_path}")
            camera_img = np.zeros((450, 800, 3), dtype=np.uint8)

        # Create combined figure with 4 separate BEV panels
        combined = make_combined_figure(camera_img, r['pts_bbox'], token)

        # Save
        out_path = os.path.join(args.out_dir, f'{idx:04d}_{token}.jpg')
        cv2.imwrite(out_path, combined)

    # Also save BEV-only images for quick browsing
    bev_dir = os.path.join(args.out_dir, 'bev_only')
    os.makedirs(bev_dir, exist_ok=True)
    for idx in indices:
        r = results[idx]
        bev_map = make_bev_map_image(r['pts_bbox'])
        # Scale up for visibility
        bev_big = cv2.resize(bev_map, (BEV_W * 4, BEV_H * 4),
                             interpolation=cv2.INTER_NEAREST)
        out_path = os.path.join(bev_dir, f'{idx:04d}_{r["token"]}.jpg')
        cv2.imwrite(out_path, bev_big)

    print(f"Saved {n_samples} visualizations to {args.out_dir}")


if __name__ == '__main__':
    main()
