# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
from torch import nn

# from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
# from .build import BACKBONE_REGISTRY
# from fastreid.utils import comm
from ..modules import MLP,MLP_SWITCHNET
from .slimmable_ops import SwitchableBatchNorm2d
import torch.nn.functional as F

from .slimmable_ops import SlimmableConv2d, SlimmableLinear
logger = logging.getLogger(__name__)
model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}

class Flatten(torch.nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)

def selectAdaptivePool2d(pool_type='avg', flatten=None):
    if pool_type == 'avg':
        pool_func = F.adaptive_avg_pool2d
    elif pool_type == 'max':
        pool_func = F.adaptive_max_pool2d
    else:
        raise ValueError("Invalid pool_type. Supported types are 'avg' and 'max'.")

    if flatten is None:
        return pool_func

    def adaptive_pool_with_flatten(x):
        x = pool_func(x, 1)  # Apply adaptive pooling
        x = flatten(x)  # Apply flattening
        return x

    return adaptive_pool_with_flatten

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, FLAGS, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = SlimmableConv2d(FLAGS, [int(i) for i in inplanes], [int(i) for i in planes], kernel_size=3, stride=stride, padding=1, bias=False)
        '''if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)'''
        #print(planes)
        self.bn1 = SwitchableBatchNorm2d( FLAGS, [int(i) for i in planes]  )
        self.conv2 = SlimmableConv2d(FLAGS,[int(i) for i in planes], [int(i) for i in planes], kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = get_norm(bn_norm, planes)
        self.bn2 = SwitchableBatchNorm2d( FLAGS, [int(i) for i in planes]  )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, FLAGS, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 =  SlimmableConv2d(FLAGS, [int(i) for i in inplanes] , [int(i) for i in planes] , kernel_size=1, bias=False)
        self.bn1 =  SwitchableBatchNorm2d( FLAGS, [int(i) for i in planes]  )
        self.conv2 = SlimmableConv2d(FLAGS, [int(i) for i in planes], [int(i) for i in planes], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SwitchableBatchNorm2d(FLAGS, [int(i) for i in planes])
        self.conv3 = SlimmableConv2d(FLAGS, [int(i) for i in planes], [int(i * self.expansion) for i in planes], kernel_size=1, bias=False)
        self.bn3 = SwitchableBatchNorm2d( FLAGS, [int(i * self.expansion) for i in planes] )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
    
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)





feat_dim_list = {
        '18x': 512,
        '50x': 2048
    }




class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

class SwitchResNet(nn.Module):
    def __init__(self, cfg,FLAGS,  last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.inplanes = 64
        self.cfg = cfg
        super().__init__()
        self.FLAGS = FLAGS 
        self.feat_dim = feat_dim_list[cfg.MODEL.BACKBONE.DEPTH]
        self.conv1 = SlimmableConv2d(FLAGS, [3 for i in FLAGS.WIDTH_MULT_LIST], [int(64*i) for i in FLAGS.WIDTH_MULT_LIST], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SwitchableBatchNorm2d( FLAGS, [int(64*i) for i in FLAGS.WIDTH_MULT_LIST]  )
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)
        # if self.cfg.SUB_MODEL.SSPL:
        #     self.whiten = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        #     self.pooling = GeM()
        self.flatten = Flatten()
        self.selectAdaptivePool2d = selectAdaptivePool2d(flatten=self.flatten)
        if self.cfg.MODEL.PROJECTION_LAYERS!=-1:
            self.setup_projector(cfg.MODEL)
        else:
            self.projector = nn.Identity()
        self.setup_head(cfg.MODEL)
        
        
        # for name,m in self.layer4.named_modules():
        #     if isinstance(m, SlimmableConv2d):
        #         print(name)
        #         finalconv = m
        #     elif isinstance(m, SwitchableBatchNorm2d):
        #         print(name)
        #         finalbn = name
        # finalout = [finalconv.out_channels_list[-1] for i in FLAGS.WIDTH_MULT_LIST ]
        # finalconv.out_channels_list = finalout
        # _set_module(self.layer4, finalbn, SwitchableBatchNorm2d( FLAGS,finalout))
        
        
        self.random_init()       

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SlimmableConv2d(self.FLAGS,   [int(self.inplanes*i) for i in self.FLAGS.WIDTH_MULT_LIST] , [ int(planes *i * block.expansion) for i in self.FLAGS.WIDTH_MULT_LIST],
                          kernel_size=1, stride=stride, bias=False),
                SwitchableBatchNorm2d(self.FLAGS, [int(planes *i * block.expansion) for i in self.FLAGS.WIDTH_MULT_LIST] ),
            )

        layers = []
        layers.append(block(self.FLAGS , [(self.inplanes*i) for i in self.FLAGS.WIDTH_MULT_LIST], [ (planes *i) for i in self.FLAGS.WIDTH_MULT_LIST], bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.FLAGS , [(self.inplanes*i) for i in self.FLAGS.WIDTH_MULT_LIST],  [ (planes *i) for i in self.FLAGS.WIDTH_MULT_LIST], bn_norm, with_ibn, with_se))
        return nn.Sequential(*layers)
    
    def setup_projector(self, model_cfg):
        if self.cfg.SUB_MODEL.USE_SWITCHNET:
            self.projector = MLP_SWITCHNET(
                cfg=self.cfg,
                FLAGS=self.FLAGS,
                input_dim=self.feat_dim,
                mlp_dims=[model_cfg.PROJECTION_DIM] * model_cfg.PROJECTION_LAYERS + [model_cfg.PROJECTION_DIM],
                final_norm = True,
            )
        else:
            self.projector = MLP(
                input_dim=self.feat_dim,
                mlp_dims=[model_cfg.PROJECTION_DIM] * model_cfg.PROJECTION_LAYERS + [model_cfg.PROJECTION_DIM],
                special_bias=False,final_norm = True,bn_track=self.cfg.BN_TRACK
            )
    def setup_head(self, model_cfg):
        # input_dim = model_cfg.PROJECTION_DIM if model_cfg.PROJECTION_LAYERS!=-1 else self.feat_dim\
 
        self.projection_cls = MLP(
            input_dim=model_cfg.PROJECTION_DIM,
            mlp_dims=[int(model_cfg.DATA.NUMBER_CLASSES * model_cfg.DATA.TRAIN_RATIO)],# 支持按比例划分数据集
            special_bias=False,
            bn_track=self.cfg.BN_TRACK
        )  
        
    def forward(self, x,return_feature=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
        # if self.cfg.SUB_MODEL.SSPL:
        #     # print(x.shape)
        #     global_feature = F.normalize(self.pooling(x), p=2.0, dim=1)
        #     global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        #     global_feature = F.normalize(global_feature, p=2.0, dim=-1)        
        #     print(global_feature.shape) # B x 512
        #     y = self.projection_cls(x)
        #     return global_feature,None,None
        # print(x.shape)
        x = self.selectAdaptivePool2d(x)
        # out = out.view(out.size(0), -1)
        x = self.projector(x)
        

        
        y = self.projection_cls(x)
        
        # if self.cfg.SUB_MODEL.SSPL:
        #     return F.normalize(x,dim=-1),None,None
        if return_feature:
            return x, None, y
        return y

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



# def init_pretrained_weights(key):
#     """Initializes model with pretrained weights.

#     Layers that don't match with pretrained layers in name or size are kept unchanged.
#     """
#     import os
#     import errno
#     import gdown

#     def _get_torch_home():
#         ENV_TORCH_HOME = 'TORCH_HOME'
#         ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
#         DEFAULT_CACHE_DIR = '~/.cache'
#         torch_home = os.path.expanduser(
#             os.getenv(
#                 ENV_TORCH_HOME,
#                 os.path.join(
#                     os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
#                 )
#             )
#         )
#         return torch_home

#     torch_home = _get_torch_home()
#     model_dir = os.path.join(torch_home, 'checkpoints')
#     try:
#         os.makedirs(model_dir)
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             # Directory already exists, ignore.
#             pass
#         else:
#             # Unexpected OSError, re-raise.
#             raise

#     filename = model_urls[key].split('/')[-1]

#     cached_file = os.path.join(model_dir, filename)

#     if not os.path.exists(cached_file):
#         if comm.is_main_process():
#             gdown.download(model_urls[key], cached_file, quiet=False)

#     comm.synchronize()

#     logger.info(f"Loading pretrained model from {cached_file}")
#     state_dict = torch.load(cached_file, map_location=torch.device('cpu'))

#     return state_dict


# @BACKBONE_REGISTRY.register()
def build_switch_resnet_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '18x': BasicBlock,
        '34x': BasicBlock,
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    model = SwitchResNet( cfg,cfg.SNET, last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage)
    # if pretrain:
    #     # Load pretrain path if specifically
    #     if pretrain_path:
    #         try:
    #             state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
    #             logger.info(f"Loading pretrained model from {pretrain_path}")
    #         except FileNotFoundError as e:
    #             logger.info(f'{pretrain_path} is not found! Please check this path.')
    #             raise e
    #         except KeyError as e:
    #             logger.info("State dict keys error! Please check the state dict.")
    #             raise e
    #     else:
    #         key = depth
    #         if with_ibn: key = 'ibn_' + key
    #         if with_se:  key = 'se_' + key

    #         state_dict = init_pretrained_weights(key)
    #     new_state_dict={}
    #     for i in state_dict:
    #         if 'bn' in i:
    #             k = i[:-len(i.split('.')[-1])]+'bn.'+str(len(cfg.SNET.WIDTH_MULT_LIST)-1)+'.'+i.split('.')[-1]
    #             new_state_dict[k] = state_dict[i]
    #         elif "downsample" in i:
    #             k = i.replace("downsample.1.", "downsample.1.bn."+str(len(cfg.SNET.WIDTH_MULT_LIST)-1)+".")
    #             new_state_dict[k] = state_dict[i]
    #     for i in new_state_dict:
    #         state_dict[i]=new_state_dict[i]

    #     incompatible = model.load_state_dict(state_dict, strict=False)
    #     if incompatible.missing_keys:
    #         logger.info(
    #             get_missing_parameters_message(incompatible.missing_keys)
    #         )
    #     if incompatible.unexpected_keys:
    #         logger.info(
    #             get_unexpected_parameters_message(incompatible.unexpected_keys)
    #         )

    return model
