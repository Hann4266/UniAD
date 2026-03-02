import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.functional import pos2posemb2d
from .intent_head_plugin.base_intent_head import BaseIntentHead
@HEADS.register_module()
class IntentHead(BaseIntentHead):
    """
    MotionHead module for a neural network, which predicts motion trajectories and is used in an autonomous driving task.

    Args:
        *args: Variable length argument list.
        predict_steps (int): The number of steps to predict motion trajectories.
        transformerlayers (dict): A dictionary defining the configuration of transformer layers.
        bbox_coder: An instance of a bbox coder to be used for encoding/decoding boxes.
        num_cls_fcs (int): The number of fully-connected layers in the classification branch.
        bev_h (int): The height of the bird's-eye-view map.
        bev_w (int): The width of the bird's-eye-view map.
        embed_dims (int): The number of dimensions to use for the query and key vectors in transformer layers.
        num_anchor (int): The number of anchor points.
        det_layer_num (int): The number of layers in the transformer model.
        group_id_list (list): A list of group IDs to use for grouping the classes.
        pc_range: The range of the point cloud.
        use_nonlinear_optimizer (bool): A boolean indicating whether to use a non-linear optimizer for training.
        anchor_info_path (str): The path to the file containing the anchor information.
        vehicle_id_list(list[int]): class id of vehicle class, used for filtering out non-vehicle objects
    """
    def __init__(self,
                 *args,
                 num_intent=7,
                 transformerlayers=None,
                 num_cls_fcs=3,
                 bev_h=30,
                 bev_w=30,
                 embed_dims=256,
                 det_layer_num=6,
                 pc_range=None,
                 loss_cls=dict(),
                 vehicle_id_list=(0,1,2,3,4,6,7),
                 ped_id_list=(8,),
                 ignore_id_list=(5,9),
                 **kwargs):
        super(IntentHead, self).__init__()
        self.num_intent = num_intent
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.num_cls_fcs = num_cls_fcs - 1
        self.embed_dims = embed_dims        
        self.pc_range = pc_range

        self.vehicle_id_list = set(vehicle_id_list)
        self.ped_id_list = set(ped_id_list)
        self.ignore_id_list = set(ignore_id_list)

        
        self._build_loss(loss_cls)
        self._build_layers(transformerlayers, det_layer_num)
        self._init_layers()
        self.obj_type_embed = nn.Embedding(3, embed_dims)
        self.motion_encoder = nn.Sequential(
            nn.Linear(4, embed_dims // 2),
            nn.ReLU(),
            nn.Linear(embed_dims // 2, embed_dims)
        )

    def _get_obj_type_ids(self, labels):
        """
        labels: (N,) detection class ids
        returns: (N,) type ids  0=ped, 1=veh, 2=ignore
        """
        type_ids = torch.full_like(labels, 2)  # default: ignore
        for cid in self.vehicle_id_list:
            type_ids[labels == cid] = 1
        for cid in self.ped_id_list:
            type_ids[labels == cid] = 0
        return type_ids

    def forward_train(self, 
                bev_embed,
                gt_intent_labels, 
                gt_labels_3d,
                outs_track={},
                outs_seg={}):
        """Forward function
            Args:
                bev_embed (Tensor): BEV feature map with the shape of [B, C, H, W].
                gt_intent (list[torch.Tensor]) Intent labels of each sample.
                gt_labels_3d (list[torch.Tensor]): Labels of each sample.
                outs_track (dict): Outputs of track head.
                outs_seg (dict): Outputs of seg head.
            Returns:
                dict: Losses of each classfication.
        """
        track_query = outs_track['track_query_embeddings'][None, None, ...] # num_dec, B, A_track, D
        all_matched_idxes = [outs_track['track_query_matched_idxes']] #BxN
        track_boxes = outs_track['track_bbox_results']
        # cat sdc query/gt to the last
        sdc_match_index = torch.full((1,), -1, dtype=all_matched_idxes[0].dtype, device=all_matched_idxes[0].device)
        all_matched_idxes = [torch.cat([all_matched_idxes[0], sdc_match_index], dim=0)]
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']
        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)
        if outs_seg != {}:
            memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
            outs_intent = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        else:
            outs_intent = self(bev_embed, track_query, None, None, track_boxes)
        losses = self.loss(
            preds_dicts_intent=outs_intent,
            matched_idxes=all_matched_idxes,
            gt_intent_labels=gt_intent_labels,
            gt_labels_3d=gt_labels_3d,
        )

        ret_dict = dict(losses=losses, outs_intent=outs_intent, track_boxes=track_boxes)
        return ret_dict
    def forward_test(self, bev_embed, outs_track={}, outs_seg={}):
        """Test function"""
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        track_boxes = outs_track['track_bbox_results']
        
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']

        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)

        bboxes, scores, labels, bbox_index, mask = track_boxes[0]

        labels[-1] = 0

        if outs_seg != {}:
            memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
            outs_intent = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        else:
            outs_intent = self(bev_embed, track_query, None, None, track_boxes)
        

        logits = outs_intent["all_intent_logits"]
        logits_last = logits[-1] 

        allowed, _ = self._build_allowed_mask_and_ignore(labels)  # (N, C)
        allowed = allowed.to(logits_last.device)
        neg_inf = torch.finfo(logits_last.dtype).min

        B = track_query.shape[0]
        result_intent = []
        for bi in range(B):
            masked_logits = logits_last[bi].masked_fill(~allowed, neg_inf)  # (N, C)
            intent_scores = F.softmax(masked_logits, dim=-1)
            intent_label = torch.argmax(intent_scores, dim=-1)
            result_intent.append(dict(
                intent_scores=intent_scores[bi].detach().cpu().tolist(),  # [N,C]
                intent_label=intent_label[bi].detach().cpu().tolist(),    # [N]
                intent_bbox_index=bbox_index.detach().cpu().tolist(),     # [N] 
            ))

        
        return result_intent, None


    @auto_fp16(apply_to=('bev_embed', 'track_query', 'lane_query', 'lane_query_pos', 'lane_query_embed', 'prev_bev'))
    def forward(self, 
                bev_embed, 
                track_query, 
                lane_query, 
                lane_query_pos, 
                track_bbox_results):
        """
        Applies forward pass on the model for motion prediction using bird's eye view (BEV) embedding, track query, lane query, and track bounding box results.

        Args:
        bev_embed (torch.Tensor): A tensor of shape (h*w, B, D) representing the bird's eye view embedding.
        track_query (torch.Tensor): A tensor of shape (B, num_dec, A_track, D) representing the track query.
        lane_query (torch.Tensor): A tensor of shape (N, M_thing, D) representing the lane query.
        lane_query_pos (torch.Tensor): A tensor of shape (N, M_thing, D) representing the position of the lane query.
        track_bbox_results (List[torch.Tensor]): A list of tensors containing the tracking bounding box results for each image in the batch.

        Returns:
        dict: A dictionary containing the following keys and values:
        - 'all_traj_scores': A tensor of shape (num_levels, B, A_track, num_points) with trajectory scores for each level.
        - 'all_traj_preds': A tensor of shape (num_levels, B, A_track, num_points, num_future_steps, 2) with predicted trajectories for each level.
        - 'valid_traj_masks': A tensor of shape (B, A_track) indicating the validity of trajectory masks.
        - 'traj_query': A tensor containing intermediate states of the trajectory queries.
        - 'track_query': A tensor containing the input track queries.
        - 'track_query_pos': A tensor containing the positional embeddings of the track queries.
        """
        device = track_query.device

        # extract the last frame of the track query
        track_query = track_query[:, -1]
        B, A, D = track_query.shape
        #type embedding
        labels = track_bbox_results[0][2]
        obj_type_ids = self._get_obj_type_ids(labels)  
        obj_type_ids = obj_type_ids.to(device)        
        obj_type_ids = obj_type_ids.unsqueeze(0).expand(B, -1)  # (B, A)
        type_emb = self.obj_type_embed(obj_type_ids)            # (B, A, D)

        #motion embedding
        bboxes = track_bbox_results[0][0].tensor   # (A, 9)
        vx = bboxes[:, 7].to(device)              # (A,)
        vy = bboxes[:, 8].to(device)              # (A,)

        speed   = torch.sqrt(vx**2 + vy**2)       # (A,)
        heading = torch.atan2(vy, vx)             # (A,)

        motion_feat = torch.stack(
            [vx, vy, speed, heading], dim=-1      # (A, 4)
        )
        motion_emb = self.motion_encoder(motion_feat)             # (A, D)
        motion_emb = motion_emb.unsqueeze(0).expand(B, -1, -1)   # (B, A, D)
        
        intent_query = track_query + type_emb + motion_emb                # (B, A, D)

        # encode the center point of the track query
        reference_points_track = self._extract_tracking_centers(
            track_bbox_results, self.pc_range).to(device)
        track_query_pos = self.boxes_query_embedding_layer(pos2posemb2d(reference_points_track))  # B, A, D
        
      
        
        all_logits = []

        inter_states = self.intentformer(
            intent_query,  # B, A_track, D
            lane_query,  # B, M, D
            track_query_pos=track_query_pos,
            lane_query_pos=lane_query_pos,
            track_bbox_results=track_bbox_results,
            bev_embed=bev_embed,
            reference_points_track=reference_points_track,
            )

        for lid in range(inter_states.shape[0]):
            all_logits.append(self.intent_cls_branches[lid](inter_states[lid]))  # (B,A,C)
        all_logits = torch.stack(all_logits, dim=0)  # (L,B,A,C)

        return {
            "all_intent_logits": all_logits,
            "intent_query": inter_states,
            "track_query": track_query,
            "track_query_pos": track_query_pos,
        }

    

    @force_fp32(apply_to=('preds_dicts_intent',))
    def loss(self, preds_dicts_intent, matched_idxes, gt_intent_labels, gt_labels_3d):
        """
        preds_dicts_intent['all_intent_logits']: (L,B,A,C)
        matched_idxes:
        - Tensor (B,A) / (A,)
        - or list[Tensor], e.g. [Tensor(A,)] in current motion-head style
        gt_intent_labels:
        - list[Tensor(num_gt)]  (recommended)
        - or Tensor(B, num_gt)
        gt_labels_3d:
        - list[...] where last-frame labels can be extracted
        """
        all_logits = preds_dicts_intent["all_intent_logits"]  # (L,B,A,C)
        L = all_logits.shape[0]

        matched_idxes = self._normalize_matched_idxes(matched_idxes, device=all_logits.device)
        gt_det_last = self._last_frame_gt_labels(gt_labels_3d)
        gt_int_last = self._last_frame_gt_intent_labels(gt_intent_labels)
        # print("--------------")
        # print("intent",gt_int_last)
        # print("detect",gt_det_last) 
        # print("--------------")
        loss_list = []
        for lid in range(L):
            loss_i = self._loss_single_layer(
                logits=all_logits[lid],              # (B,A,C)
                matched_idxes=matched_idxes,         # (B,A)
                gt_intent_labels=gt_int_last,   # list[tensor] or tensor
                gt_det_labels_last=gt_det_last,      # list[tensor]
            )
            loss_list.append(loss_i)

        loss_dict = {"loss_intent": loss_list[-1] * self.loss_weight}
        for d, li in enumerate(loss_list[:-1]):
            loss_dict[f"d{d}.loss_intent"] = li * self.loss_weight
        return loss_dict
    def _loss_single_layer(self, logits, matched_idxes, gt_intent_labels, gt_det_labels_last):
        B, A, C = logits.shape
        device = logits.device

        xs, ys, allowed_all, is_ped_all = [], [], [], []

        for b in range(B):
            midx = matched_idxes[b]
            valid_q = (midx >= 0)
            if valid_q.sum() == 0:
                continue

            gt_idx = midx[valid_q].long()

            num_gt_int = gt_intent_labels[b].numel()
            num_gt_det = gt_det_labels_last[b].numel()
            in_range = (gt_idx >= 0) & (gt_idx < num_gt_int) & (gt_idx < num_gt_det)
            if in_range.sum() == 0:
                continue

            gt_idx = gt_idx[in_range]
            x_b   = logits[b, valid_q, :][in_range]
            y_b   = gt_intent_labels[b][gt_idx].to(device).long()
            det_b = gt_det_labels_last[b][gt_idx].to(device).long()

            keep = (y_b >= 0) & (y_b < C)
            allowed_b, ign_b = self._build_allowed_mask_and_ignore(det_b)
            keep = keep & (~ign_b)
            row_idx = torch.arange(y_b.numel(), device=device)
            keep = keep & allowed_b[row_idx, y_b.clamp(min=0, max=C - 1)]

            if keep.sum() == 0:
                continue

            #  ped 
            is_ped_b = torch.zeros(det_b.shape, dtype=torch.bool, device=device)
            for cid in self.ped_id_list:
                is_ped_b |= (det_b == cid)

            xs.append(x_b[keep])
            ys.append(y_b[keep])
            allowed_all.append(allowed_b[keep])
            is_ped_all.append(is_ped_b[keep])

        if len(xs) == 0:
            return logits.sum() * 0.0

        x       = torch.cat(xs, dim=0)
        y       = torch.cat(ys, dim=0)
        allowed = torch.cat(allowed_all, dim=0)
        is_ped  = torch.cat(is_ped_all, dim=0)   # (N_keep,) bool

        alpha = self.intent_class_weight if self.intent_class_weight is not None else self.focal_alpha

        # per-sample loss
        loss_per = self.masked_softmax_focal_loss(
            logits=x,
            target=y,
            allowed_mask=allowed,
            gamma=self.focal_gamma,
            alpha=alpha,
            reduction="none"  
        )

        # ped/veh 分组 mean，再加权合并
        is_veh = ~is_ped
        veh_loss = loss_per[is_veh].mean() if is_veh.any() else loss_per.sum() * 0.0
        ped_loss = loss_per[is_ped].mean() if is_ped.any() else loss_per.sum() * 0.0

        return veh_loss + ped_loss * self.ped_loss_weight
    def _last_frame_gt_intent_labels(self, gt_intent_labels):
        """
        Normalize gt_intent_labels to list[Tensor(num_gt)] for the LAST frame.

        Supports:
        - list[Tensor(num_gt)]
        - list[np.ndarray(num_gt)]
        - list[list/tuple[...]]            # temporal queue, take [-1]
        - Tensor(B, N)                     # padded style (less common)
        """
        # case: Tensor(B, N)
        if torch.is_tensor(gt_intent_labels):
            if gt_intent_labels.dim() == 1:
                return [gt_intent_labels.long()]
            elif gt_intent_labels.dim() == 2:
                return [gt_intent_labels[b].long() for b in range(gt_intent_labels.size(0))]
            else:
                raise ValueError(f"Unsupported gt_intent_labels tensor shape: {tuple(gt_intent_labels.shape)}")

        # case: list/tuple
        if not isinstance(gt_intent_labels, (list, tuple)):
            raise TypeError(f"Unsupported gt_intent_labels type: {type(gt_intent_labels)}")

        out = []
        for item in gt_intent_labels:
            # queue/list/tuple -> take last frame
            if isinstance(item, (list, tuple)):
                last = item[-1]
            else:
                last = item

            # numpy -> tensor
            if not torch.is_tensor(last):
                last = torch.as_tensor(last)

            # if accidentally has extra temporal dim, take last
            if last.dim() >= 2:
                last = last[-1]

            out.append(last.long())
        return out

    def _last_frame_gt_labels(self, gt_labels_3d):
        """
        Returns list[Tensor(num_gt)] for last-frame detection labels.
        Supports common UniAD formats:
        - list[Tensor(num_gt)]               (already last-frame labels)
        - list[list/tuple[...]]              (queue, take [-1])
        - list[Tensor(T, num_gt)]            (rare; if 2D, take [-1])
        """
        out = []
        for item in gt_labels_3d:
            # queue/list/tuple -> take last frame
            if isinstance(item, (list, tuple)):
                last = item[-1]
            else:
                last = item

            if not torch.is_tensor(last):
                # sometimes nested one more level
                if isinstance(last, (list, tuple)) and torch.is_tensor(last[-1]):
                    last = last[-1]
                else:
                    raise TypeError(f"Unsupported gt_labels_3d element type: {type(last)}")

            # if shape like (T, N), take last frame
            if last.dim() >= 2:
                # Usually labels are 1D; if not, assume first dim is time/queue
                last = last[-1]

            out.append(last.long())
        return out

    def _normalize_matched_idxes(self, matched_idxes, device=None):
        """
        Normalize matched_idxes to Tensor(B, A), dtype long.
        Accepts:
        - Tensor(A,)
        - Tensor(B,A)
        - list[Tensor(A,)]  (common in your current style)
        - list[Tensor(B,A)] (rare)
        """
        if isinstance(matched_idxes, (list, tuple)):
            if len(matched_idxes) == 0:
                raise ValueError("matched_idxes is empty list/tuple")

            # case: [Tensor(A,)]
            if len(matched_idxes) == 1 and torch.is_tensor(matched_idxes[0]) and matched_idxes[0].dim() == 1:
                out = matched_idxes[0].unsqueeze(0)  # (1, A)
            else:
                # stack list of (A,) -> (B,A), or list of (B,A) (less common)
                tensors = []
                for x in matched_idxes:
                    if not torch.is_tensor(x):
                        raise TypeError(f"matched_idxes element must be Tensor, got {type(x)}")
                    if x.dim() == 1:
                        x = x.unsqueeze(0)  # (1,A)
                    tensors.append(x)
                # If they are already (B,A) and list length==1, cat works too
                out = torch.cat(tensors, dim=0)
        elif torch.is_tensor(matched_idxes):
            if matched_idxes.dim() == 1:
                out = matched_idxes.unsqueeze(0)
            elif matched_idxes.dim() == 2:
                out = matched_idxes
            else:
                raise ValueError(f"matched_idxes tensor must be 1D or 2D, got shape {tuple(matched_idxes.shape)}")
        else:
            raise TypeError(f"Unsupported matched_idxes type: {type(matched_idxes)}")

        out = out.long()
        if device is not None:
            out = out.to(device)
        return out

    def _build_allowed_mask_and_ignore(self, det_cls):
        """
        det_cls: (N,) detection class ids for matched GT objects
        Returns:
        allowed: (N, C) bool
        ign:     (N,) bool   -> True means drop this sample from intent loss
        """
        device = det_cls.device
        N = det_cls.numel()
        C = self.num_intent

        # intent ids
        INT_STOP   = 0
        INT_MOVING = 1
        INT_CROSS  = 2 
        INT_TR     = 3  # ← TURN_RIGHT
        INT_TL     = 4  # ← TURN_LEFT
        INT_LCR    = 5  # ← LANE_CHANGE_RIGHT
        INT_LCL    = 6  # ← LANE_CHANGE_LEFT

        allowed = torch.zeros((N, C), dtype=torch.bool, device=device)

        veh_mask = torch.zeros(N, dtype=torch.bool, device=device)
        ped_mask = torch.zeros(N, dtype=torch.bool, device=device)
        ign_mask = torch.zeros(N, dtype=torch.bool, device=device)

        for cid in self.vehicle_id_list:
            veh_mask |= (det_cls == cid)
        for cid in self.ped_id_list:
            ped_mask |= (det_cls == cid)
        for cid in self.ignore_id_list:
            ign_mask |= (det_cls == cid)

        # vehicle: allow stop/moving/lcl/lcr/tl/tr ; disallow cross
        if veh_mask.any():
            allowed[veh_mask, INT_STOP] = True
            allowed[veh_mask, INT_MOVING] = True
            allowed[veh_mask, INT_LCL] = True
            allowed[veh_mask, INT_LCR] = True
            allowed[veh_mask, INT_TL] = True
            allowed[veh_mask, INT_TR] = True
            # INT_CROSS = False

        # pedestrian: allow stop/moving/cross ; disallow lcl/lcr/tl/tr
        if ped_mask.any():
            allowed[ped_mask, INT_STOP] = True
            allowed[ped_mask, INT_MOVING] = True
            allowed[ped_mask, INT_CROSS] = True

        # unknown classes (not veh/ped/ign) -> ignore as well (safer)
        unk_mask = ~(veh_mask | ped_mask | ign_mask)
        ign_mask = ign_mask | unk_mask

        return allowed, ign_mask

    @staticmethod
    def masked_softmax_focal_loss(logits, target, allowed_mask=None,
                              gamma=2.0, alpha=None, reduction="mean",
                              eps=1e-8):
        """
        logits: (N, C)
        target: (N,) long, in [0, C-1]
        allowed_mask: (N, C) bool, True means class is allowed for this sample
                    if provided, invalid classes are masked out from softmax denominator
        alpha:
        - None
        - scalar float
        - Tensor shape (C,)  (per-class weights)
        """
        assert logits.dim() == 2
        assert target.dim() == 1
        N, C = logits.shape
        device = logits.device

        if N == 0:
            return logits.sum() * 0.0

        # mask invalid classes before softmax
        if allowed_mask is not None:
            allowed_mask = allowed_mask.to(device=device, dtype=torch.bool)
            # avoid all-False rows causing NaN
            row_has_any = allowed_mask.any(dim=1)
            if not torch.all(row_has_any):
                # fallback: if a row is all False, allow all classes (shouldn't happen if caller checks)
                allowed_mask = allowed_mask.clone()
                allowed_mask[~row_has_any] = True

            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(~allowed_mask, neg_inf)
        else:
            masked_logits = logits

        log_prob = F.log_softmax(masked_logits, dim=-1)     # (N, C)
        prob = log_prob.exp()                               # (N, C)

        idx = torch.arange(N, device=device)
        log_pt = log_prob[idx, target]                      # (N,)
        pt = prob[idx, target].clamp(min=eps, max=1.0)      # (N,)

        focal_factor = (1.0 - pt).pow(gamma)

        # alpha / class weight
        if alpha is None:
            alpha_t = 1.0
        elif torch.is_tensor(alpha):
            alpha = alpha.to(device=device, dtype=logits.dtype)
            alpha_t = alpha[target]  # per-class
        else:
            alpha_t = float(alpha)   # scalar

        loss = -alpha_t * focal_factor * log_pt  # (N,)

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
