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
    """Render map predictions onto a BEV image.

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


def make_combined_figure(camera_img, bev_map, token):
    """Create a side-by-side figure: camera image + BEV map prediction.

    Args:
        camera_img: (H, W, 3) BGR camera image
        bev_map: (bev_h, bev_w, 3) RGB BEV map
        token: sample token string

    Returns:
        Combined image (H, W_total, 3) uint8
    """
    # Scale BEV map to match camera image height
    cam_h, cam_w = camera_img.shape[:2]
    bev_h, bev_w = bev_map.shape[:2]

    # Scale BEV to same height as camera
    scale = cam_h / bev_h
    bev_scaled = cv2.resize(bev_map, (int(bev_w * scale), cam_h),
                            interpolation=cv2.INTER_NEAREST)

    # Add border and label to BEV
    bev_bordered = cv2.copyMakeBorder(bev_scaled, 0, 0, 2, 2,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Combine side by side
    combined = np.concatenate([camera_img, bev_bordered], axis=1)

    # Add title
    cv2.putText(combined, f'Token: {token}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, 'Camera', (10, cam_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, 'BEV Map Pred', (cam_w + 10, cam_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Legend
    y_legend = 60
    for name, color in COLORS.items():
        cv2.rectangle(combined, (cam_w + 10, y_legend - 12),
                      (cam_w + 25, y_legend), color, -1)
        cv2.putText(combined, name, (cam_w + 30, y_legend),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_legend += 20

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

        # Get BEV map
        bev_map = make_bev_map_image(r['pts_bbox'])

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

        # Create combined figure
        combined = make_combined_figure(camera_img, bev_map, token)

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
