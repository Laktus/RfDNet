import torch
import torch.nn as nn

from external.groupfree3d.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from models.registers import MODULES

@MODULES.register_module  # register module here0
class DetrEncoderToSkip(nn.Module):
    def __init__(self, cfg, optim_spec):
        super(DetrEncoderToSkip, self).__init__()
        
        # 1.self.downsampling = nn.Conv1d(1024, 128, kernel_size=1)
        
        self.downsampling = PointnetSAModuleVotes(
            radius=0.4,                     # Radius for local region
            nsample=32,                     # Number of points to sample in each local region
            npoint=128,                     # Number of centroids (downsampled points)
            mlp=[256, 256, 256],            # MLP layers for local feature extraction
        )

        self.downsampling = self.downsampling.to('cuda')
    
    def forward(self, xyz, proposal_features):
        # proposal_features is npoints x batch x channel. make batch x channel x npoints
        proposal_features = proposal_features.permute(0, 2, 1)

        # must make contiguous
        proposal_features = proposal_features.contiguous()

        xyz, proposal_features, xyz_inds = self.downsampling(xyz, proposal_features)
        
        # swap back
        proposal_features = proposal_features.permute(0, 1, 2)

        return proposal_features