import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from .mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger

#from .CoaT.src.models.coat import *
from .multimodal_coat import *

class MultiModalCoaT(nn.Module):
    def __init__(self,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 coat_type='lite_tiny',
                 rec_enable = False
                 ):
        super().__init__()

        self.frozen_stages = frozen_stages

        self._freeze_stages()
        if coat_type == 'tiny':
            self.coat = coat_tiny()
        elif coat_type == 'mini':
            self.coat = coat_mini()
        elif coat_type == 'small':
            self.coat = coat_small()
        elif coat_type == 'lite_tiny':
            self.coat = coat_lite_tiny()
        elif coat_type == 'lite_mini':
            self.coat = coat_lite_mini()
        elif coat_type == 'lite_small':
            self.coat = coat_lite_samll()
        elif coat_type == 'lite_medium':
            self.coat = coat_lite_medium()
        else:
            assert False

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            m = self.coat
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        print('Randomly initialize Multi-modal CoaT weights.')
        self.coat.apply(self.coat._init_weights)
        if pretrained:
            print('Loading pretrained CoaT weights from ' + pretrained)
            self.coat.load_checkpoint(pretrained)

    def forward(self, x, l, l_mask, rec_enable):
        """Forward function."""
        outs, bbox = self.coat(x, l, l_mask, rec_enable)
        return tuple(outs), bbox

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MultiModalCoaT, self).train(mode)
        self._freeze_stages()
