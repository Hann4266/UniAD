"""
Visualize intent matches on front camera images.

Scene selection: only scenes where at least one match has BOTH
    gt.intent_name == TURN_LEFT  AND  pred.intent_name == TURN_LEFT

For qualifying scenes: draw ALL matches in ALL frames.

Usage:
    python visualize_intent.py \
        --json intent_matches.json \
        --nuscenes /path/to/nuscenes \
        --output ./vis_output \
        --version v1.0-trainval
"""

import os
import json
import argparse
import numpy as np
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

TARGET_INTENTS = ['LANE_CHANGE_LEFT']

_PALETTE = [
    (255,  80,  80), ( 80, 200,  80), ( 80, 120, 255),
    (255, 200,   0), (  0, 220, 220), (220,  80, 220),
    (255, 140,   0), (  0, 180, 180), (160, 255,  80),
    (255, 100, 180), (100, 255, 200), (200, 160, 255),
]
TARGET_INTENTS_Covert = {'LANE_CHANGE_RIGHT':'LCR',
                        'LANE_CHANGE_LEFT':'LCL',
                        'TURN_LEFT':'TL',
                        'TURN_RIGHT':'TR',
                        'CROSSING':'C',
                        'MOVING':'M',
                        'STOPPED':'S',


                        }

def get_3d_corners(translation, size, rotation):
    w, l, h = size
    corners = np.array([
        [ l/2,  w/2, -h/2], [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2], [-l/2,  w/2, -h/2],
        [ l/2,  w/2,  h/2], [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2], [-l/2,  w/2,  h/2],
    ]).T
    q = Quaternion(rotation)
    return q.rotation_matrix @ corners + np.array(translation).reshape(3, 1)


def project_box(corners_global, K, cam2global):
    global2cam = np.linalg.inv(cam2global)
    corners_h = np.vstack([corners_global, np.ones((1, 8))])
    corners_cam = (global2cam @ corners_h)[:3]
    if (corners_cam[2] > 0.1).sum() < 4:
        return None, False
    pts = view_points(corners_cam, K, normalize=True)
    return pts[:2], True


def draw_box(img, pts2d, color, label=None, thickness=2):
    h, w = img.shape[:2]
    pts = pts2d.T.astype(np.int32)

    def cp(p):
        return (int(np.clip(p[0], -9999, 9999)),
                int(np.clip(p[1], -9999, 9999)))

    for seq in [[0,1,2,3,0], [4,5,6,7,4]]:
        for i in range(len(seq)-1):
            cv2.line(img, cp(pts[seq[i]]), cp(pts[seq[i+1]]), color, thickness)
    for a, b in [(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(img, cp(pts[a]), cp(pts[b]), color, thickness)
    for a, b in [(0,1),(1,5),(5,4),(4,0)]:
        cv2.line(img, cp(pts[a]), cp(pts[b]), color, thickness+1)

    if label:
        vis_label = TARGET_INTENTS_Covert[label]
        cx = int(np.clip(np.mean(pts[[4,5],0]), 0, w-1))
        cy = int(np.clip(np.mean(pts[[4,5],1]) - 8, 15, h-1))
        (tw, th), _ = cv2.getTextSize(vis_label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(img, (cx-3, cy-th-3), (cx+tw+3, cy+3), (0,0,0), -1)
        cv2.putText(img, vis_label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)


def scene_has_correct_match(frames, intent):
    for frame_data in frames.values():
        for m in frame_data['matches']:
            if (m['gt']['intent_name']   == intent and
                m['pred']['intent_name'] == intent):
                return True
    return False


def main(args):
    print(f"Loading nuScenes ({args.version}) from {args.nuscenes} ...")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes, verbose=False)

    print(f"Loading intent matches from {args.json} ...")
    with open(args.json) as f:
        data = json.load(f)

    sample2cam = {s['token']: s['data']['CAM_FRONT'] for s in nusc.sample}

    total_frames = 0

    for TARGET_INTENT in TARGET_INTENTS:
        print(f"\n{'='*50}")
        print(f"Processing intent: {TARGET_INTENT}")
        print(f"{'='*50}")

        qualifying = {st: frames for st, frames in data.items()
                      if scene_has_correct_match(frames, TARGET_INTENT)}
        print(f"Qualifying scenes: {len(qualifying)} / {len(data)}")

        for si, (scene_token, frames) in enumerate(qualifying.items()):
            print(f"[{si+1}/{len(qualifying)}] {scene_token[:24]}...")

            scene_id_colors = {}
            def get_color(tid):
                if tid not in scene_id_colors:
                    scene_id_colors[tid] = _PALETTE[len(scene_id_colors) % len(_PALETTE)]
                return scene_id_colors[tid]

            pred_dir = os.path.join(args.output, TARGET_INTENT, scene_token, 'pred')
            gt_dir   = os.path.join(args.output, TARGET_INTENT, scene_token, 'gt')
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(gt_dir,   exist_ok=True)

            for frame_idx_str, frame_data in sorted(frames.items(), key=lambda x: int(x[0])):
                sample_token = frame_data['sample_token']
                frame_idx    = int(frame_idx_str)
                matches      = frame_data['matches']

                if not matches or sample_token not in sample2cam:
                    continue

                cam_data = nusc.get('sample_data', sample2cam[sample_token])
                img_path = os.path.join(nusc.dataroot, cam_data['filename'])
                base_img = cv2.imread(img_path)
                if base_img is None:
                    continue

                img_pred = base_img.copy()
                img_gt   = base_img.copy()

                cs  = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                ego = nusc.get('ego_pose',          cam_data['ego_pose_token'])
                K   = np.array(cs['camera_intrinsic'])

                cam2ego        = np.eye(4)
                cam2ego[:3,:3] = Quaternion(cs['rotation']).rotation_matrix
                cam2ego[:3, 3] = np.array(cs['translation'])

                ego2global        = np.eye(4)
                ego2global[:3,:3] = Quaternion(ego['rotation']).rotation_matrix
                ego2global[:3, 3] = np.array(ego['translation'])

                cam2global = ego2global @ cam2ego

                for match in matches:
                    gt_tid = match['gt']['tracking_id']
                    color  = get_color(gt_tid)
                    bgr    = (color[2], color[1], color[0])

                    pr = match['pred']
                    corners = get_3d_corners(pr['translation'], pr['size'], pr['rotation'])
                    pts2d, valid = project_box(corners, K, cam2global)
                    if valid:
                        draw_box(img_pred, pts2d, bgr,
                                 label=pr['intent_name'], thickness=2)

                    gt = match['gt']
                    corners = get_3d_corners(gt['translation'], gt['size'], gt['rotation'])
                    pts2d, valid = project_box(corners, K, cam2global)
                    if valid:
                        draw_box(img_gt, pts2d, bgr,
                                 label=gt['intent_name'], thickness=2)

                cv2.imwrite(os.path.join(pred_dir, f"{frame_idx}.png"), img_pred)
                cv2.imwrite(os.path.join(gt_dir,   f"{frame_idx}.png"), img_gt)
                total_frames += 1

    print(f"\nDone. {total_frames} frames saved → {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',      default='/zihan-west-vol/UniAD/test/base_intent_fornt/Thu_Mar__5_04_26_51_2026/intent/intent_matches.json',  help='Path to intent_matches.json')
    parser.add_argument('--nuscenes',  default='/zihan-west-vol/UniAD/data/nuscenes',  help='nuScenes dataroot')
    parser.add_argument('--output',    default='./vis_intent', help='Output directory')
    parser.add_argument('--version',   default='v1.0-trainval', help='nuScenes version')
    args = parser.parse_args()
    main(args)