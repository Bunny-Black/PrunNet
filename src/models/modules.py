#!/usr/bin/env python3
"""
Modified from: fbcode/multimo/models/encoders/mlp.py
"""
import math
import torch

from torch import nn
import torch.nn.functional as F
from typing import List, Type
from .subnetworks.slimmable_ops import SlimmableLinear,SwitchableBatchNorm1d
from ..utils import logging
from torch.autograd import Function
logger = logging.get_logger("FCLearning")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
        final_norm = False,
        bn_track = True,
    ):
        super(MLP, self).__init__()
        self.final_norm = final_norm
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim,bias=True)
        if final_norm:
            self.last_bn = nn.BatchNorm1d(last_dim,track_running_stats=bn_track)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        if self.final_norm:
            x = self.last_bn(x)
        return x

class ElasticBoundary(nn.Module):
    def __init__(self,num_class):
        super(ElasticBoundary, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_class), requires_grad=True)
        self.register_parameter('bias', None)
        self.weight.data.uniform_(-1, 1)

    def forward(self,x):
        '''
        x: rmax-rmin,shape:[num_class]
        '''
        x = x*F.sigmoid(self.weight)
        return x

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
    
class MLP_SWITCHNET(nn.Module):
    def __init__(
        self,
        cfg,
        FLAGS,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
        final_norm = False,
        bn_track = True,
    ):
        super(MLP_SWITCHNET, self).__init__()
        self.final_norm = final_norm
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            if isinstance(projection_prev_dim,List):
                fc_layer = SlimmableLinear(FLAGS=FLAGS,in_features_list=projection_prev_dim,out_features_list=[mlp_dim for i in cfg.SNET.WIDTH_MULT_LIST])
            else:
                fc_layer = SlimmableLinear(FLAGS=FLAGS,in_features_list=[int(projection_prev_dim*i) for i in cfg.SNET.WIDTH_MULT_LIST],out_features_list=[mlp_dim for i in cfg.SNET.WIDTH_MULT_LIST])
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(SwitchableBatchNorm1d(FLAGS=FLAGS,num_features_list=[mlp_dim for i in cfg.SNET.WIDTH_MULT_LIST]))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        if isinstance(projection_prev_dim,List):
            self.last_layer = SlimmableLinear(FLAGS=FLAGS,in_features_list=projection_prev_dim,out_features_list=[last_dim for i in cfg.SNET.WIDTH_MULT_LIST])
        else:
            self.last_layer = SlimmableLinear(FLAGS=FLAGS,in_features_list=[int(projection_prev_dim*i) for i in cfg.SNET.WIDTH_MULT_LIST],out_features_list=[last_dim for i in cfg.SNET.WIDTH_MULT_LIST])
        if final_norm:
            self.last_bn = SwitchableBatchNorm1d(FLAGS=FLAGS,num_features_list=[last_dim for i in cfg.SNET.WIDTH_MULT_LIST])
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        if self.final_norm:
            x = self.last_bn(x)
        return x