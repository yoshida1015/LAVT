import torch
import torch.nn as nn
from .mask_predictor import SimpleDecoding_CoaT
from .backbone import MultiModalCoaT
from ._utils import LAVT

__all__ = ['lavt']


# LAVT
def _segm_lavt(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.coat_type == 'tiny':
        embed_dims=[152, 152, 152, 152]
    elif args.coat_type == 'mini':
        embed_dims=[152, 216, 216, 216]
    elif args.coat_type == 'small':
        embed_dims=[152, 320, 320, 320]
    elif args.coat_type == 'lite_tiny':
        embed_dims=[64, 128, 256, 320]
    elif args.coat_type == 'lite_mini':
        embed_dims=[64, 128, 320, 512]
    elif args.coat_type == 'lite_small':
        embed_dims=[64, 128, 320, 512]
    elif args.coat_type == 'lite_medium':
        embed_dims=[128, 256, 320, 512]
    else:
        assert False

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalCoaT(use_checkpoint=False, coat_type=args.coat_type, rec_head=args.rec_enable)
    backbone.init_weights(pretrained=pretrained)

    #if pretrained:
    #    print('Initializing Multi-modal CoaT weights from ' + pretrained)
    #    backbone.init_weights(pretrained=pretrained)
    #else:
    #    print('Randomly initialize Multi-modal CoaT weights.')
    #    backbone.init_weights()

    model_map = [SimpleDecoding_CoaT, LAVT]

    classifier = model_map[0](embed_dims)
    base_model = model_map[1]

    model = base_model(backbone, classifier)
    return model


def _load_model_lavt(pretrained, args):
    model = _segm_lavt(pretrained, args)
    return model


def lavt(pretrained='', args=None):
    return _load_model_lavt(pretrained, args)
