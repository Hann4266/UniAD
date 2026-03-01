"""
LOKI Stage 2: Intent Head training.

Loads a frozen stage-1 perception checkpoint and trains the intent head only.
Based on base_loki_perception.py with:
    - Intent head added
    - BEV encoder / backbone / neck frozen
    - gt_labels_intent in pipeline
    - No map head (map_features=False in IntentTransformerDecoder)
"""

_base_ = ["../_base_/default_runtime.py"]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# --------------------------------------------------------------------- #
#  Point cloud / BEV range
# --------------------------------------------------------------------- #
point_cloud_range = [-51.2, 0, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [51.2, 102.4]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False)

# --------------------------------------------------------------------- #
#  LOKI classes
# --------------------------------------------------------------------- #
class_names = [
    "Pedestrian",
    "Car",
    "Bus",
    "Truck",
    "Van",
    "Motorcyclist",
    "Bicyclist",
    "Other"
]
num_classes = len(class_names)

# LOKI class IDs for intent head
# Pedestrian=0, Car=1, Bus=2, Truck=3, Van=4, Motorcyclist=5, Bicyclist=6, Other=7
vehicle_id_list = [1, 2, 3, 4, 5, 6]
ped_id_list = [0]
ignore_id_list = [7]

input_modality = dict(
    use_lidar=False, use_camera=True,
    use_radar=False, use_map=False, use_external=False)

# --------------------------------------------------------------------- #
#  Architecture dimensions
# --------------------------------------------------------------------- #
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 100
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)

queue_length = 3

# --------------------------------------------------------------------- #
#  Trajectory / prediction args (kept for interface compat)
# --------------------------------------------------------------------- #
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = False

occ_n_future = 4
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])

planning_steps = 6

# --------------------------------------------------------------------- #
#  Model config — perception (frozen) + intent head
# --------------------------------------------------------------------- #
train_gt_iou_threshold = 0.3

model = dict(
    type="UniAD",
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=900,
    num_classes=num_classes,
    vehicle_id_list=vehicle_id_list,
    pc_range=point_cloud_range,
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    # Stage 2: freeze perception modules
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bn=True,
    freeze_bev_encoder=True,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),
    mem_args=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=num_classes,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0,
            alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_past_traj_weight=0.0,
    ),
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=past_steps,
        fut_steps=fut_steps,
        transformer=dict(
            type="PerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(type="TemporalSelfAttention",
                             embed_dims=_dim_, num_levels=1),
                        dict(type="SpatialCrossAttention",
                             pc_range=point_cloud_range,
                             deformable_attention=dict(
                                 type="MSDeformableAttention3D",
                                 embed_dims=_dim_,
                                 num_points=8,
                                 num_levels=_num_levels_),
                             embed_dims=_dim_),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn", "norm",
                        "cross_attn", "norm",
                        "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(type="MultiheadAttention",
                             embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type="CustomMSDeformableAttention",
                             embed_dims=_dim_, num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn", "norm",
                        "cross_attn", "norm",
                        "ffn", "norm"),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -10.0, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes,
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0,
            alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    # ----- Intent Head (stage 2) -----
    intent_head=dict(
        type='IntentHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_intent=7,  # STOPPED, MOVING, LCL, LCR, TL, TR, CROSSING
        embed_dims=_dim_,
        num_cls_fcs=3,
        det_layer_num=1,
        pc_range=point_cloud_range,
        vehicle_id_list=vehicle_id_list,
        ped_id_list=ped_id_list,
        ignore_id_list=ignore_id_list,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=False,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0,
            class_weight=[1.0, 1.39, 3.21, 6.19, 5.44, 10.05, 11.15]
        ),
        transformerlayers=dict(
            type='IntentTransformerDecoder',
            pc_range=point_cloud_range,
            bev_h=bev_h_,
            bev_w=bev_w_,
            map_features=False,  # No map head for LOKI
            embed_dims=_dim_,
            num_layers=3,
            transformerlayers=dict(
                type='IntentTransformerAttentionLayer',
                batch_first=True,
                attn_cfgs=[
                    dict(
                        type='IntentDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=1,
                        num_heads=8,
                        num_points=4),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
            ),
        ),
    ),
    # Training settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(type="IoUCost", weight=0.0),
                pc_range=point_cloud_range,
            ),
        )
    ),
)

# --------------------------------------------------------------------- #
#  Dataset config
# --------------------------------------------------------------------- #
dataset_type = "LokiE2EDataset"
data_root = "/root/loki_data/"
info_root = "data/infos/"
file_client_args = dict(backend="disk")
ann_file_train = info_root + "loki_infos_train.pkl"
ann_file_val = info_root + "loki_infos_val.pkl"
ann_file_test = info_root + "loki_infos_val.pkl"

# --------------------------------------------------------------------- #
#  Data pipelines
# --------------------------------------------------------------------- #
train_pipeline = [
    dict(type="LoadLokiImage", to_float32=True,
         target_size=(1600, 900)),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="LoadAnnotations3D_E2E",
         with_bbox_3d=True,
         with_label_3d=True,
         with_future_anns=False,
         with_ins_inds_3d=True,
         ins_inds_add_1=True,
         with_intent_label_3d=True),
    dict(type="GenerateDummyOccLabels",
         bev_h=bev_h_, bev_w=bev_w_,
         n_future=occ_n_future_max),
    dict(type="ObjectRangeFilterTrack",
         point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="ObjectFOVFilterTrack", fov_deg=60.0),
    dict(type="ObjectCameraVisibleFilter"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CustomCollect3D",
         keys=[
             "gt_bboxes_3d",
             "gt_labels_3d",
             "gt_labels_intent",
             "gt_inds",
             "img",
             "timestamp",
             "l2g_r_mat",
             "l2g_t",
             "gt_fut_traj",
             "gt_fut_traj_mask",
             "gt_past_traj",
             "gt_past_traj_mask",
             "gt_sdc_bbox",
             "gt_sdc_label",
             "gt_sdc_fut_traj",
             "gt_sdc_fut_traj_mask",
             "gt_lane_labels",
             "gt_lane_bboxes",
             "gt_lane_masks",
             # Occ (dummy)
             "gt_segmentation",
             "gt_instance",
             "gt_centerness",
             "gt_offset",
             "gt_flow",
             "gt_backward_flow",
             "gt_occ_has_invalid_frame",
             "gt_occ_img_is_valid",
             # Future boxes (dummy)
             "gt_future_boxes",
             "gt_future_labels",
             # Planning (dummy)
             "sdc_planning",
             "sdc_planning_mask",
             "command",
         ]),
]

test_pipeline = [
    dict(type="LoadLokiImage", to_float32=True,
         target_size=(1600, 900)),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="LoadAnnotations3D_E2E",
         with_bbox_3d=False,
         with_label_3d=False,
         with_future_anns=False,
         with_ins_inds_3d=False,
         ins_inds_add_1=True),
    dict(type="GenerateDummyOccLabels",
         bev_h=bev_h_, bev_w=bev_w_,
         n_future=occ_n_future_max),
    dict(type="MultiScaleFlipAug3D",
         img_scale=(1600, 900),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type="DefaultFormatBundle3D",
                  class_names=class_names, with_label=False),
             dict(type="CustomCollect3D",
                  keys=[
                      "img",
                      "timestamp",
                      "l2g_r_mat",
                      "l2g_t",
                      "gt_lane_labels",
                      "gt_lane_bboxes",
                      "gt_lane_masks",
                      "gt_segmentation",
                      "gt_instance",
                      "gt_centerness",
                      "gt_offset",
                      "gt_flow",
                      "gt_backward_flow",
                      "gt_occ_has_invalid_frame",
                      "gt_occ_img_is_valid",
                      "sdc_planning",
                      "sdc_planning_mask",
                      "command",
                  ]),
         ]),
]

# --------------------------------------------------------------------- #
#  Data loader config
# --------------------------------------------------------------------- #
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        classes=class_names,
        modality=input_modality,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        samples_per_gpu=1,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        classes=class_names,
        modality=input_modality,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

# --------------------------------------------------------------------- #
#  Optimizer / schedule
# --------------------------------------------------------------------- #
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={"img_backbone": dict(lr_mult=0.1)}),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 20
evaluation = dict(interval=20, pipeline=test_pipeline)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook"),
           dict(type="TensorboardLoggerHook")])
checkpoint_config = dict(interval=1)

# Load stage-1 perception checkpoint
# TODO: Update this path to your trained stage-1 checkpoint
load_from = "/mnt/storage/UniAD/work_dirs/base_loki_perception/epoch_10.pth"

find_unused_parameters = True
