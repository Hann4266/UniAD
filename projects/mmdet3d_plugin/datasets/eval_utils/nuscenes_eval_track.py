# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
# from nuscenes.eval.common.loaders import (
#     add_center_dist,
#     get_samples_of_custom_split,
#     load_gt,
#     load_gt_of_sample_tokens,
#     load_prediction,
#     load_prediction_of_sample_tokens,
# )
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


from .loaders import (
    add_center_dist,
    get_samples_of_custom_split,
    load_gt,
    load_gt_of_sample_tokens,
    load_prediction,
    load_prediction_of_sample_tokens,
)
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, LEGACY_METRICS, MOT_METRIC_MAP
from nuscenes.eval.tracking.data_classes import (
    TrackingBox,
    TrackingConfig,
    TrackingMetricData,
    TrackingMetricDataList,
    TrackingMetrics,
)
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking.render import recall_metric_curve, summary_plot
from nuscenes.eval.tracking.utils import print_final_metrics
from .splits import is_predefined_split

def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field
def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      filter_fov = False,
                      fov_horizontal = 102.4,
                      fov_degree = 70.0,
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)
    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter,fov_filter = 0, 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        sample_record = nusc.get('sample', sample_token)
        sd_record   = nusc.get('sample_data', sample_record['data']['CAM_FRONT'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        ego_x   = pose_record['translation'][0]
        ego_y   = pose_record['translation'][1]

        ego_q   = Quaternion(pose_record['rotation'])
        yaw_rad = ego_q.yaw_pitch_roll[0]
        ego_forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])   # (fx, fy)
        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])
        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get('sample', sample_token)['anns']
        bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
                         nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
        bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

        if filter_fov:
            half_fov = fov_horizontal / 2.0
            half_fov_degree = fov_degree / 2.0          # 35.0 degrees
            tan_half_fov = np.tan(np.radians(half_fov_degree))  # tan(35°) = 0.700

            fov_filtered_boxes = []
            for box in eval_boxes[sample_token]:
                # USE ego_translation DIRECTLY - no transformation needed!
                bx, by = box.translation[0], box.translation[1]
                dx = bx - ego_x
                dy = by - ego_y

                forward_dist =  ego_forward[0] * dx + ego_forward[1] * dy      # dot product
                lateral_dist = -ego_forward[1] * dx + ego_forward[0] * dy      # perpendicular

                # Check: in front + within ±35° cone
                if (forward_dist > 0.5 and  forward_dist<= half_fov                                     # in front of ego
                        and np.abs(lateral_dist) / (forward_dist + 1e-6) <= tan_half_fov):  # within FOV
                    fov_filtered_boxes.append(box)

            eval_boxes.boxes[sample_token] = fov_filtered_boxes
            fov_filter += len(eval_boxes.boxes[sample_token])
    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)
        if filter_fov:
            print("=> After FOV filtering: %d" % fov_filter)
    return eval_boxes
def load_pred_intent_map(result_path: str):
    """
    pred_intent[sample_token][tracking_id] = intent_label (int)
    """
    with open(result_path, 'r') as f:
        data = json.load(f)
    results = data.get('results', data)
    pred_intent = {}
    for sample_token, boxes in results.items():
        mp = {}
        if isinstance(boxes, list):
            for b in boxes:
                if isinstance(b, dict) and 'intent_label' in b:
                    mp[str(b.get('tracking_id'))] = int(b['intent_label'])
        pred_intent[sample_token] = mp
    return pred_intent
class TrackingEval:
    """
    This is the official nuScenes tracking evaluation code.
    Results are written to the provided output_dir.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/tracking for more details.
    """
    def __init__(self,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 verbose: bool = True,
                 render_classes: List[str] = None):
        """
        Initialize a TrackingEval object.
        :param config: A TrackingConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param nusc_version: The version of the NuScenes dataset.
        :param nusc_dataroot: Path of the nuScenes dataset on disk.
        :param verbose: Whether to print to stdout.
        :param render_classes: Classes to render to disk or None.
        """
        self.cfg = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.render_classes = render_classes

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Initialize NuScenes object.
        # We do not store it in self to let garbage collection take care of it and save memory.
        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)

        # Load data.
        if verbose:
            print('Initializing nuScenes tracking evaluation')

        if is_predefined_split(split_name=eval_set):
            pred_boxes, self.meta = load_prediction(
                self.result_path, self.cfg.max_boxes_per_sample, TrackingBox, verbose=verbose
            )
            gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)
        else:
            sample_tokens_of_custom_split : List[str] = get_samples_of_custom_split(split_name=eval_set, nusc=nusc)
            pred_boxes, self.meta = load_prediction_of_sample_tokens(self.result_path, self.cfg.max_boxes_per_sample,
                TrackingBox, sample_tokens=sample_tokens_of_custom_split, verbose=verbose)
            gt_boxes = load_gt_of_sample_tokens(nusc, sample_tokens_of_custom_split, TrackingBox, verbose=verbose)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances.
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering tracks')
        pred_boxes = filter_eval_boxes(nusc, pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        gt_boxes = filter_eval_boxes(nusc, gt_boxes, self.cfg.class_range, filter_fov = True, verbose=verbose)

        self.sample_tokens = gt_boxes.sample_tokens

        # Convert boxes to tracks format.
        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)
        self.nusc = nusc
        self.pred_boxes = pred_boxes
        self.gt_boxes = gt_boxes
    def evaluate_intent(self,
                    data_infos,
                    intent_data,
                    intent_label_fn,
                    output_dir=None,
                    num_intent_classes=7,
                    ignore_label=-1,
                    eval_track_names=('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                                      'motorcycle', 'bicycle', 'pedestrian'),
                    intent_class_names=('STOPPED', 'MOVING', 'CROSSING',
                                        'TURN_RIGHT', 'TURN_LEFT',
                                        'LANE_CHANGE_RIGHT', 'LANE_CHANGE_LEFT'),
                    score_thr=0.0):
        """
        Evaluate intent on matched TP pairs using tracking-style matching.
        - pred intent: read from raw json (intent_label in each box dict)
        - gt intent: computed from intent_data + (scene_token, frame_idx, instance_token)
        - only evaluate tracking boxes whose tracking_name in eval_track_names
        - report per-intent-class precision/recall/F1 + confusion matrix
        """

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'intent')
        os.makedirs(output_dir, exist_ok=True)

        eval_track_names = set(eval_track_names)
        assert len(intent_class_names) == num_intent_classes, \
            f"intent_class_names len {len(intent_class_names)} != num_intent_classes {num_intent_classes}"

        # token -> (scene_token, frame_idx)
        token2meta = {}
        for info in data_infos:
            tok = info.get('token', info.get('sample_idx', None))
            if tok is None:
                continue
            token2meta[tok] = (info['scene_token'], int(info['frame_idx']))

        # pred intent map from json (token -> tracking_id -> intent_label)
        pred_intent = load_pred_intent_map(self.result_path)

        # GT intent getter: gt_box.tracking_id is instance_token in devkit GT
        def get_gt_intent(sample_token, gt_instance_token):
            if sample_token not in token2meta:
                return ignore_label
            scene_token, frame_idx = token2meta[sample_token]
            gt_tok = str(gt_instance_token)
            if scene_token not in intent_data:
                return ignore_label
            if gt_tok not in intent_data[scene_token]:
                return ignore_label
            arr = intent_data[scene_token][gt_tok]["labels"]
            return int(intent_label_fn(frame_idx, arr))

        # Hungarian matching setup
        dist_th = float(self.cfg.dist_th_tp)
        try:
            from scipy.optimize import linear_sum_assignment
            use_scipy = True
        except Exception:
            use_scipy = False

        cm = np.zeros((num_intent_classes, num_intent_classes), dtype=np.int64)

        # per frame / per class matching
        for sample_token in self.gt_boxes.sample_tokens:
            gts_all = self.gt_boxes[sample_token]
            prs_all = self.pred_boxes[sample_token]

            gts_all = [b for b in gts_all if b.tracking_name in eval_track_names]
            prs_all = [b for b in prs_all if b.tracking_name in eval_track_names and b.tracking_score >= score_thr]

            if len(gts_all) == 0 or len(prs_all) == 0:
                continue

            # match per tracking class name (car/ped/...)
            class_names = set([b.tracking_name for b in gts_all] + [b.tracking_name for b in prs_all])
            for cls in class_names:
                gts = [b for b in gts_all if b.tracking_name == cls]
                prs = [b for b in prs_all if b.tracking_name == cls]
                if len(gts) == 0 or len(prs) == 0:
                    continue

                gt_y, gt_xy = [], []
                for b in gts:
                    y = get_gt_intent(sample_token, b.tracking_id)
                    if y < 0 or y >= num_intent_classes:
                        continue
                    gt_y.append(y)
                    gt_xy.append(np.array(b.translation[:2], dtype=np.float32))
                if len(gt_y) == 0:
                    continue

                pr_y, pr_xy = [], []
                mp = pred_intent.get(sample_token, {})
                for b in prs:
                    tid = str(b.tracking_id)
                    if tid not in mp:
                        continue
                    y = int(mp[tid][0]) if isinstance(mp[tid], tuple) else int(mp[tid])
                    if y < 0 or y >= num_intent_classes:
                        continue
                    pr_y.append(y)
                    pr_xy.append(np.array(b.translation[:2], dtype=np.float32))
                if len(pr_y) == 0:
                    continue

                G, P = len(gt_y), len(pr_y)
                cost = np.zeros((G, P), dtype=np.float32)
                for i in range(G):
                    for j in range(P):
                        cost[i, j] = np.linalg.norm(gt_xy[i] - pr_xy[j])

                if use_scipy:
                    gi, pj = linear_sum_assignment(cost)
                else:
                    # greedy fallback
                    gi, pj = [], []
                    used = set()
                    for i in range(G):
                        j = int(np.argmin(cost[i]))
                        if j not in used:
                            gi.append(i); pj.append(j); used.add(j)

                for i, j in zip(gi, pj):
                    if cost[i, j] <= dist_th:
                        cm[gt_y[i], pr_y[j]] += 1

        # ---------- metrics ----------
        total = int(cm.sum())
        acc = float(np.trace(cm) / max(total, 1))

        support = cm.sum(axis=1)   # GT count
        predcnt = cm.sum(axis=0)   # predicted count
        tp = np.diag(cm)

        eps = 1e-12
        prec = tp / np.maximum(predcnt, eps)
        rec = tp / np.maximum(support, eps)
        f1 = 2 * prec * rec / np.maximum(prec + rec, eps)

        valid = support > 0
        macro_f1 = float(np.mean(f1[valid])) if valid.any() else 0.0
        macro_rec = float(np.mean(rec[valid])) if valid.any() else 0.0

        per_class = {}
        for i, name in enumerate(intent_class_names):
            per_class[name] = {
                'precision': float(prec[i]) if (support[i] > 0 or predcnt[i] > 0) else 0.0,
                'recall': float(rec[i]) if support[i] > 0 else 0.0,
                'f1': float(f1[i]) if support[i] > 0 else 0.0,
                'support': int(support[i]),
                'tp': int(tp[i]),
                'pred_cnt': int(predcnt[i]),
            }

        summary = dict(
            matched=total,
            acc=acc,
            macro_f1=macro_f1,
            macro_recall=macro_rec,
            per_class=per_class,
            confusion=cm.tolist(),
            eval_track_names=sorted(list(eval_track_names)),
            dist_th_tp=dist_th,
            score_thr=score_thr,
        )

        with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print(f"[IntentEval] matched={total}, acc={acc:.4f}, macro_f1={macro_f1:.4f}")
            # 打印每类（更好 debug）
            for name in intent_class_names:
                m = per_class[name]
                print(f"  {name:16s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  supp={m['support']}")

        return summary
    def evaluate(self) -> Tuple[TrackingMetrics, TrackingMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg)

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = TrackingMetricDataList()

        def accumulate_class(curr_class_name):
            curr_ev = TrackingEvaluation(self.tracks_gt, self.tracks_pred, curr_class_name, self.cfg.dist_fcn_callable,
                                         self.cfg.dist_th_tp, self.cfg.min_recall,
                                         num_thresholds=TrackingMetricData.nelem,
                                         metric_worst=self.cfg.metric_worst,
                                         verbose=self.verbose,
                                         output_dir=self.output_dir,
                                         render_classes=self.render_classes)
            curr_md = curr_ev.accumulate()
            metric_data_list.set(curr_class_name, curr_md)

        for class_name in self.cfg.class_names:
            accumulate_class(class_name)

        # -----------------------------------
        # Step 2: Aggregate metrics from the metric data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        for class_name in self.cfg.class_names:
            # Find best MOTA to determine threshold to pick for traditional metrics.
            # If multiple thresholds have the same value, pick the one with the highest recall.
            md = metric_data_list[class_name]
            if np.all(np.isnan(md.mota)):
                best_thresh_idx = None
            else:
                best_thresh_idx = np.nanargmax(md.mota)

            # Pick best value for traditional metrics.
            if best_thresh_idx is not None:
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == '':
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx]
                    metrics.add_label_metric(metric_name, class_name, value)

            # Compute AMOTA / AMOTP.
            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
                assert len(values) == TrackingMetricData.nelem

                if np.all(np.isnan(values)):
                    # If no GT exists, set to nan.
                    value = np.nan
                else:
                    # Overwrite any nan value with the worst possible value.
                    np.all(values[np.logical_not(np.isnan(values))] >= 0)
                    values[np.isnan(values)] = self.cfg.metric_worst[metric_name]
                    value = float(np.nanmean(values))
                metrics.add_label_metric(metric_name, class_name, value)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, md_list: TrackingMetricDataList) -> None:
        """
        Renders a plot for each class and each metric.
        :param md_list: TrackingMetricDataList instance.
        """
        if self.verbose:
            print('Rendering curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        # Plot a summary.
        summary_plot(self.cfg, md_list, savepath=savepath('summary'))

        # For each metric, plot all the classes in one diagram.
        for metric_name in LEGACY_METRICS:
            recall_metric_curve(self.cfg, md_list, metric_name, savepath=savepath('%s' % metric_name))

    def main(self, render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: The serialized TrackingMetrics computed during evaluation.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print metrics to stdout.
        if self.verbose:
            print_final_metrics(metrics)

        # Render curves.
        if render_curves:
            self.render(metric_data_list)

        return metrics_summary


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render statistic curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--render_classes', type=str, default='', nargs='+',
                        help='For which classes we render tracking results to disk.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)
    render_classes_ = args.render_classes

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))

    nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                             render_classes=render_classes_)
    nusc_eval.main(render_curves=render_curves_)