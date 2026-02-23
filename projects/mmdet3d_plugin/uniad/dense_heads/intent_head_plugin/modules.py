#---------------------------------------------------------------------------------#
# UniAD-style Intent Transformer Decoder (no anchors / no trajectory refinement)  #
# Interface + style aligned with MotionTransformerDecoder                         #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class IntentTransformerDecoder(BaseModule):
    """
    IntentTransformerDecoder:
      - Keeps the same interface style as MotionTransformerDecoder
      - Removes: P modes, anchors, reference_trajs, trajectory refinement
      - Uses: agent-agent interaction + agent-map interaction + (optional) agent-bev interaction
    Shapes:
      track_query:      (B, A, D)
      lane_query:       (B, M, D)
      track_query_pos:  (B, A, D)
      lane_query_pos:   (B, M, D)
      output inter_states: (L, B, A, D)
    """

    def __init__(self, pc_range=None, bev_h=100, bev_w=200, embed_dims=256, transformerlayers=None, num_layers=3, **kwargs):
        super(IntentTransformerDecoder, self).__init__()
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.num_layers = num_layers

        # agent-agent and agent-map interactions (same spirit as MotionTransformerDecoder)
        self.track_agent_interaction_layers = nn.ModuleList(
            [TrackAgentInteraction(embed_dims=embed_dims) for _ in range(self.num_layers)]
        )
        self.map_interaction_layers = nn.ModuleList(
            [MapInteraction(embed_dims=embed_dims) for _ in range(self.num_layers)]
        )
        self.bev_interaction_layers = nn.ModuleList(
            [build_transformer_layer(transformerlayers) for _ in range(self.num_layers)]
        )

        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 4, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )

    def forward(self,
                track_query,
                lane_query,
                track_query_pos=None,
                lane_query_pos=None,
                track_bbox_results=None,
                bev_embed=None,
                reference_points_track=None,
                **kwargs):
        """
        Returns:
          inter_states: (L, B, A, D)
          inter_dummy:  (L, 0)  # to mimic MotionTransformerDecoder's second return (reference trajs)
        """
        intermediate = []

        # base query state
        query_embed = track_query

        for lid in range(self.num_layers):
            # agent-agent interaction
            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query, query_pos=track_query_pos, key_pos=track_query_pos
            )

            # agent-map interaction
            map_query_embed = self.map_interaction_layers[lid](
                query_embed, lane_query, query_pos=track_query_pos, key_pos=lane_query_pos
            )

            # agent-bev interaction 
            bev_query_embed = self.bev_interaction_layers[lid](
                query_embed,                     # (B, A, D)
                value=bev_embed,                 # (H*W, B, D)
                query_pos=track_query_pos,       # (B, A, D)
                reference_points=reference_points_track.unsqueeze(2),  # (B, A, 1, 2), num_levels=1
                spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query_embed.device),
                level_start_index=torch.tensor([0], device=query_embed.device),
            )
            query_embed = torch.cat(
                [track_query_embed, map_query_embed, bev_query_embed, track_query + track_query_pos],
                dim=-1
            )
            # query_embed = torch.cat(
            #     [track_query_embed, map_query_embed, track_query + track_query_pos],
            #     dim=-1
            # )
            query_embed = self.out_query_fuser(query_embed)
            intermediate.append(query_embed)

        inter_states = torch.stack(intermediate)  # (L, B, A, D)
        return inter_states


class TrackAgentInteraction(BaseModule):
    """
    Interaction between agents (A tokens) using a standard TransformerDecoderLayer:
      tgt = query (B,A,D)
      mem = key   (B,A,D)
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first
        )

    def forward(self, query, key, query_pos=None, key_pos=None):
        """
        query: (B, A, D)
        key:   (B, A, D)
        """
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        return self.interaction_transformer(query, key)


class MapInteraction(BaseModule):
    """
    Interaction between agent tokens (A) and map tokens (M) using TransformerDecoderLayer:
      tgt = query (B,A,D)
      mem = key   (B,M,D)
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first
        )

    def forward(self, query, key, query_pos=None, key_pos=None):
        """
        query: (B, A, D)
        key:   (B, M, D)
        """
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        return self.interaction_transformer(query, key)