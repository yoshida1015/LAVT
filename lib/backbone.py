import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from .mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger

from .CoaT.src.models.coat import *

class MultiModalCoaT(nn.Module):
    def __init__(self,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 coat_type='lite_tiny'
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
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=('upernet' in pretrained), logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, l, l_mask):
        """Forward function."""
        outs = self.coat(x, l, l_mask)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MultiModalCoaT, self).train(mode)
        self._freeze_stages()
