#---------------------------------------------------------------------------------#
# Intent deformable attention for UniAD-style intent head (no anchors/refinement) #
#---------------------------------------------------------------------------------#

import copy
import warnings
import math
import torch
import torch.nn as nn

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import (
    build_attention, build_feedforward_network, build_norm_layer
)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import ConfigDict, deprecated_api_warning

from projects.mmdet3d_plugin.uniad.modules.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32
)


@TRANSFORMER_LAYER.register_module()
class IntentTransformerAttentionLayer(BaseModule):
    """Transformer layer wrapper for intent BEV interaction.
    Usually used with operation_order=('cross_attn', 'norm', 'ffn', 'norm')
    and attn_cfgs=[dict(type='IntentDeformableAttention', ...)].
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs'
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'`{ori_name}` is deprecated. Please use `{new_name}` in `ffn_cfgs`.',
                    DeprecationWarning
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)
        self.batch_first = batch_first

        assert operation_order is not None, "operation_order must be provided"
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')

        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert len(attn_cfgs) == num_attn, \
                f'Length of attn_cfgs ({len(attn_cfgs)}) != num_attn ({num_attn})'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'

        self.attentions = ModuleList()
        attn_idx = 0
        for op_name in operation_order:
            if op_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[attn_idx]:
                    assert self.batch_first == attn_cfgs[attn_idx]['batch_first']
                else:
                    attn_cfgs[attn_idx]['batch_first'] = self.batch_first

                attention = build_attention(attn_cfgs[attn_idx])
                attention.operation_name = op_name
                self.attentions.append(attention)
                attn_idx += 1

        self.embed_dims = self.attentions[0].embed_dims

        # FFNs
        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns

        for i in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[i]:
                ffn_cfgs[i]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[i]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[i], dict(type='FFN'))
            )

        # Norms
        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        query: (B, Nq, D) when batch_first=True
        value: (Nv, B, D) or (B, Nv, D) depending on attention implementation
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions in {self.__class__.__name__}')
        else:
            assert len(attn_masks) == self.num_attn, \
                f'len(attn_masks)={len(attn_masks)} must equal num_attn={self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

            else:
                raise ValueError(f'Unsupported operation {layer}')

        return query


@ATTENTION.register_module()
class IntentDeformableAttention(BaseModule):
    """Deformable attention for intent BEV interaction (no trajectory anchors).
    
    Key difference from MotionDeformableAttention:
      - query shape: (B, A, D)
      - reference input: reference_points (B, A, num_levels, 2) in normalized BEV coords
      - no reference_trajs / no bbox-based coordinate transform needed
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 num_steps=1,       # kept for config compatibility; should be 1 in intent
                 sample_index=-1,    # kept for config compatibility; unused
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)

        if embed_dims % num_heads != 0:
            raise ValueError(
                f'embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}'
            )

        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(f'invalid input for _is_power_of_2: {n} (type: {type(n)})')
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                'It is better for dim_per_head to be power of 2 for CUDA efficiency.'
            )

        if num_steps != 1:
            warnings.warn(
                f'IntentDeformableAttention is designed for num_steps=1, but got {num_steps}. '
                f'Will still build, but output_proj assumes step fusion is trivial.'
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_steps = num_steps  # compatibility
        self.sample_index = sample_index  # compatibility (unused)

        # For intent query (B, A, D): predict offsets/weights directly per query
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )

        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True)
        )

        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)

        self.sampling_offsets.bias.data = grid_init.reshape(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'}, cls_name='IntentDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                reference_points=None,
                bbox_results=None,   # kept for interface compatibility; unused
                flag='decoder',
                **kwargs):
        """
        Args:
            query: (B, A, D)
            value: (H*W, B, D)  [same as UniAD BEV embed convention]
            query_pos: (B, A, D)
            reference_points: (B, A, num_levels, 2), normalized to [0,1]
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
        Returns:
            out: (B, A, D)
        """
       
        bs, num_query, _ = query.shape

        if value is None:
            value = query  # fallback, though not typical for cross-attn
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        assert reference_points is not None, "IntentDeformableAttention requires reference_points"
        assert spatial_shapes is not None, "spatial_shapes is required"
        assert level_start_index is not None, "level_start_index is required"

        # reference_points: (B, A, L, 2)
        assert reference_points.dim() == 4, \
            f"reference_points must be (B, A, L, 2), got {tuple(reference_points.shape)}"
        assert reference_points.size(0) == bs and reference_points.size(1) == num_query
        assert reference_points.size(2) == self.num_levels, \
            f"reference_points num_levels {reference_points.size(2)} != configured {self.num_levels}"
        assert reference_points.size(-1) == 2, "Only 2D reference_points supported"

        # value comes as (H*W, B, D) in UniAD
        if value.dim() != 3:
            raise ValueError(f"value must be 3D, got {tuple(value.shape)}")
        if value.shape[1] == bs:
            # (Nv, B, D) -> (B, Nv, D)
            value = value.permute(1, 0, 2)
        elif value.shape[0] == bs:
            # already (B, Nv, D)
            pass
        else:
            raise ValueError(
                f"Cannot infer value layout. Expected (Nv,B,D) or (B,Nv,D), got {tuple(value.shape)}"
            )

        bs_v, num_value, _ = value.shape
        assert bs_v == bs
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value, \
            "sum(H_l * W_l) must equal num_value"

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)  # (B, Nv, H, Dh)

        # offsets / weights from query
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # normalize offsets by feature map size
        # offset_normalizer: (L, 2) = (W, H)
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1
        ).to(query.device).to(query.dtype)

        # reference_points: (B, A, L, 2)
        # expand to (B, A, H, L, P, 2)
        ref = reference_points[:, :, None, :, None, :]
        sampling_locations = ref + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        # deformable attn core
        if torch.cuda.is_available() and value.is_cuda:
            # use fp32 function for stability
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        # output: (B, A, D)
        output = self.output_proj(output)
        return self.dropout(output) + identity
