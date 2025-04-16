import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.nn.functional import relu, avg_pool2d
from copy import deepcopy
from ..modules import MLP
from .subnet import SubnetConv2d, SubnetLinear
import torch.nn.functional as F
# Multiple Input Sequential

class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        mask = inputs[1]
        mode = inputs[2]
        inputs = inputs[0]
        for module in self._modules.values():
            if isinstance(module, SubnetBasicBlock) or isinstance(module, SubnetBottleneck):
                inputs = module(inputs, mask, mode)
            else:
                inputs = module(inputs)

        return inputs

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

class Flatten(torch.nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)
    
## Define ResNet50 model

def subnet_conv1x1(cfg,in_planes, out_planes, stride=1, sparsity=0.5):
    return SubnetConv2d(cfg,in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, sparsity=sparsity)
    
def subnet_conv3x3(cfg,in_planes, out_planes, stride=1, sparsity=0.5):
    return SubnetConv2d(cfg,in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, sparsity=sparsity)

def subnet_conv7x7(cfg,in_planes, out_planes, stride=1, sparsity=0.5):
    return SubnetConv2d(cfg,in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=(3,3), bias=False, sparsity=sparsity)

class SubnetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, cfg,in_planes, planes, stride=1, sparsity=0.5, name=""):
        super(SubnetBasicBlock, self).__init__()
        self.name = name
        self.cfg = cfg
        self.bn_track = cfg.BN_TRACK
        self.affine = True
        self.conv1 = subnet_conv3x3(cfg,in_planes, planes, stride, sparsity=sparsity)
        if self.affine:
            self.bn1 = nn.BatchNorm2d(planes, track_running_stats=self.bn_track, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(planes, track_running_stats=self.bn_track)
        self.conv2 = subnet_conv3x3(cfg,planes, planes, sparsity=sparsity)
        if self.affine:
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=self.bn_track, affine=True)
        else:
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=self.bn_track)

        # Shortcut
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = 1
            self.conv3 = SubnetConv2d(cfg,in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, sparsity=sparsity)
            if self.affine:
                self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=self.bn_track, affine=True)
            else:
                self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=self.bn_track)
        self.count = 0

    def forward(self, x, sparsity, mask, mode='train'):
        name = self.name + ".conv1"
        out = relu(self.bn1(self.conv1(x, sparsity, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode)))
        name = self.name + ".conv2"
        out = self.bn2(self.conv2(out, sparsity, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode))
        if self.shortcut is not None:
            name = self.name + ".conv3"
            out += self.bn3(self.conv3(x, sparsity, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode))
        else:
            out += x
        out = relu(out)
        return out
    
class SubnetBottleneck(nn.Module):
    expansion = 4
    def __init__(self, cfg, in_planes, planes, stride=1, sparsity=0.5, name=""):
        super(SubnetBottleneck, self).__init__()
        self.name = name
        self.cfg = cfg
        self.bn_track = cfg.BN_TRACK
        self.affine = True
        self.conv1 = subnet_conv1x1(cfg, in_planes, planes, sparsity=sparsity)
        if self.affine:
            self.bn1 = nn.BatchNorm2d(planes, track_running_stats=self.bn_track, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(planes, track_running_stats=self.bn_track)
        self.conv2 = subnet_conv3x3(cfg, planes, planes, stride, sparsity=sparsity)
        if self.affine:
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=self.bn_track, affine=True)
        else:
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=self.bn_track)
        self.conv3 = subnet_conv1x1(cfg, planes, planes * self.expansion, sparsity=sparsity)
        if self.affine:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion, track_running_stats=self.bn_track, affine=True)
        else:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion, track_running_stats=self.bn_track)

        # Shortcut
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = 1
            self.conv4 = subnet_conv1x1(cfg, in_planes, self.expansion * planes, stride=stride, sparsity=sparsity)
            if self.affine:
                self.bn4 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=self.bn_track, affine=True)
            else:
                self.bn4 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=self.bn_track)
        self.count = 0

    def forward(self, x, sparsity, mask, mode='train'):
        name = self.name + ".conv1"
        out = relu(self.bn1(self.conv1(x, sparsity, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode)))
        name = self.name + ".conv2"
        out = relu(self.bn2(self.conv2(out, sparsity, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode)))
        name = self.name + ".conv3"
        out = self.bn3(self.conv3(out, sparsity, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode))
        if self.shortcut is not None:
            name = self.name + ".conv4"
            out += self.bn4(self.conv4(x, sparsity, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode))
        else:
            out += x
        out = relu(out)
        return out

class SubnetResNet(nn.Module):
    def __init__(self, block, num_blocks, nf, sparsity, cfg, model_cfg):
        super(SubnetResNet, self).__init__()
        self.sparsity = sparsity
        self.in_planes = nf
        self.cfg = cfg
        self.bn_track = self.cfg.BN_TRACK
        self.sparsity_list = self.cfg.SUB_MODEL.SPARSITY
        self.conv1 = subnet_conv7x7(cfg, 3, nf * 1, (2,2), sparsity=sparsity)
        if True:
            self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=self.bn_track, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=self.bn_track)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, sparsity=sparsity, name="layer1", cfg=self.cfg)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, sparsity=sparsity, name="layer2", cfg=self.cfg)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, sparsity=sparsity, name="layer3", cfg=self.cfg)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, sparsity=sparsity, name="layer4", cfg=self.cfg)
        self.flatten = Flatten()
        self.selectAdaptivePool2d = selectAdaptivePool2d(flatten=self.flatten)
        self.feat_dim = 2048
        if self.cfg.MODEL.PROJECTION_LAYERS != -1:
            self.setup_projector(model_cfg)
        else:
            self.projector = nn.Identity()
        self.setup_head(model_cfg)
        self.act = OrderedDict()

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def setup_projector(self, model_cfg):
        self.projector = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[model_cfg.PROJECTION_DIM] * model_cfg.PROJECTION_LAYERS + [model_cfg.PROJECTION_DIM],
            special_bias=False, final_norm=True
        )

    def setup_head(self, model_cfg):
        self.projection_cls = MLP(
            input_dim=model_cfg.PROJECTION_DIM,
            mlp_dims=[int(model_cfg.DATA.NUMBER_CLASSES * model_cfg.DATA.TRAIN_RATIO)],
            special_bias=False,
            bn_track=self.bn_track
        )

    def _make_layer(self, block, planes, num_blocks, stride, sparsity, name, cfg):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        name_count = 0
        for stride in strides:
            new_name = name + "." + str(name_count)
            layers.append(block(cfg, self.in_planes, planes, stride, sparsity, new_name))
            self.in_planes = planes * block.expansion
            name_count += 1
        return mySequential(*layers)

    def forward(self, x, sparsity, mask, mode="train", epoch=1, return_feature=False):
        if mask is None:
            mask = self.none_masks
        out = relu(self.bn1(self.conv1(x, sparsity, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)))
        out = self.layer1(out, sparsity, mask, mode, epoch)
        out = self.layer2(out, sparsity, mask, mode, epoch)
        out = self.layer3(out, sparsity, mask, mode, epoch)
        out = self.layer4(out, sparsity, mask, mode, epoch)
        out = self.selectAdaptivePool2d(out)
        x = self.projector(out)
        y = self.projection_cls(x)

        if return_feature:
            return x, None, y
        return y

    def get_masks(self, sparsity):
        task_mask = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                weight_mask = GetSubnetFaster.apply(module.w_m.abs(),
                                                    module.zeros_weight,
                                                    module.ones_weight,
                                                    sparsity)
                task_mask[name + '.weight'] = (weight_mask.clone() > 0).type(torch.uint8)
                if getattr(module, 'bias') is not None:
                    bias_mask = GetSubnetFaster.apply(module.b_m.abs(),
                                                      module.zeros_weight,
                                                      module.ones_weight,
                                                      sparsity)
                    task_mask[name + '.bias'] = (bias_mask.clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
        return task_mask

def SubnetResNet50(cfg, model_cfg, nf=64, sparsity=0.5):
    return SubnetResNet(SubnetBottleneck, [3, 4, 6, 3], nf, sparsity, cfg, model_cfg)

