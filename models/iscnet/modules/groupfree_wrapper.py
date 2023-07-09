import numpy as np

import torch

from models.registers import MODULES
from external.groupfree3d.models.detector import GroupFreeDetector
from configs.path_config import ScanNet_OBJ_CLASS_IDS as OBJ_CLASS_IDS

'''
This is a custom wrapper for the GroupFree module. We register the module using RfDNets @MODULES#register_module framework. 
The forward function is copied to enable easier modifications.
'''
@MODULES.register_module
class GroupFree(GroupFreeDetector):
  def __init__(self, cfg, optim_spec):
    super().__init__(
      len(OBJ_CLASS_IDS),
      cfg.config['data']['groupfree']['num_heading_bin'],
      len(OBJ_CLASS_IDS),
      np.load('datasets/scannet/scannet_means.npz')['arr_0']
    )

  def forward(self, inputs):
    """ Forward pass of the network
    Args:
        inputs: dict
            {point_clouds}
            point_clouds: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
    Returns:
        end_points: dict
    """
    end_points = {}
    end_points = self.backbone_net(inputs['point_clouds'], end_points)
    # Query Points Generation
    points_xyz = end_points['fp2_xyz']
    points_features = end_points['fp2_features']
    xyz = end_points['fp2_xyz']
    features = end_points['fp2_features']
    end_points['seed_inds'] = end_points['fp2_inds']
    end_points['seed_xyz'] = xyz
    end_points['seed_features'] = features
    if self.sampling == 'fps':
        xyz, features, sample_inds = self.fps_module(xyz, features)
        cluster_feature = features
        cluster_xyz = xyz
        end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
        end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
    elif self.sampling == 'kps':
        points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
        points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
        sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
        xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
        cluster_feature = features
        cluster_xyz = xyz
        end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
        end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
    else:
        raise NotImplementedError
    
    end_points['features_for_skip_propagation'] = features

    # Proposal
    proposal_center, proposal_size = self.proposal_head(cluster_feature,
                                                        base_xyz=cluster_xyz,
                                                        end_points=end_points,
                                                        prefix='proposal_')  # N num_proposal 3
    base_xyz = proposal_center.detach().clone()
    base_size = proposal_size.detach().clone()
    # Transformer Decoder and Prediction
    if self.num_decoder_layers > 0:
        query = self.decoder_query_proj(cluster_feature)
        key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
    # Position Embedding for Cross-Attention
    if self.cross_position_embedding == 'none':
        key_pos = None
    elif self.cross_position_embedding in ['xyz_learned']:
        key_pos = points_xyz
    else:
        raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")
    for i in range(self.num_decoder_layers):
        prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'
        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            query_pos = None
        elif self.self_position_embedding == 'xyz_learned':
            query_pos = base_xyz
        elif self.self_position_embedding == 'loc_learned':
            query_pos = torch.cat([base_xyz, base_size], -1)
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")
        # Transformer Decoder Layer
        query = self.decoder[i](query, key, query_pos, key_pos)
        # Prediction
        base_xyz, base_size = self.prediction_heads[i](query,
                                                       base_xyz=cluster_xyz,
                                                       end_points=end_points,
                                                       prefix=prefix)
        base_xyz = base_xyz.detach().clone()
        base_size = base_size.detach().clone()
    return end_points