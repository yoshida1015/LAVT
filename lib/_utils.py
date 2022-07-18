from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, clip_txt, l_mask):
        input_shape = x.shape[-2:]
        features, bbox = self.backbone(x, l_feats, clip_txt, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x, bbox


class LAVT(_LAVTSimpleDecode):
    pass
