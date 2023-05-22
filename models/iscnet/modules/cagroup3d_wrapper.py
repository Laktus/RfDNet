import torch
import torch.nn as nn
from pcdet.models.detectors.cagroup3d import CAGroup3D
from models.registers import MODULES

@MODULES.register_module
class CAGroup3DWrapper(nn.Module):
    def __init__(self, cfg, optim_spec = None):
        super(CAGroup3DWrapper, self).__init__()
        self.cagroup = CAGroup3D(cfg)
    
    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        return self.cagroup({})