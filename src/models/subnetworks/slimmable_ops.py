import torch.nn as nn
import torch

#from utils.config import FLAGS

class SwitchableLayerNorm(nn.LayerNorm):
    def __init__(self, FLAGS, num_features_list):
        super(SwitchableLayerNorm, self).__init__(normalized_shape=max(num_features_list))
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        self.FLAGS = FLAGS
        self.width_mult = max(self.FLAGS.WIDTH_MULT_LIST)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = self.FLAGS.WIDTH_MULT_LIST.index(self.width_mult)
        in_channels = self.num_features_list[idx]
        normalized_shape = (self.num_features_list[idx],)
        normalized_shape = tuple(normalized_shape)
        weight = self.weight[:in_channels]
        bias = self.bias[:in_channels]
        return nn.functional.layer_norm(
            input, normalized_shape,weight, bias, self.eps)
    
class SwitchableBatchNorm1d(nn.Module):
    def __init__(self, FLAGS, num_features_list ):
        super(SwitchableBatchNorm1d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        self.FLAGS = FLAGS
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm1d(i,track_running_stats=True))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(self.FLAGS.WIDTH_MULT_LIST)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = self.FLAGS.WIDTH_MULT_LIST.index(self.width_mult)
        y = self.bn[idx](input)
        return y

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, FLAGS, num_features_list ):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        self.FLAGS = FLAGS
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i,track_running_stats=True))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(self.FLAGS.WIDTH_MULT_LIST)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = self.FLAGS.WIDTH_MULT_LIST.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, FLAGS, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.FLAGS=FLAGS
        self.width_mult = max(self.FLAGS.WIDTH_MULT_LIST)
        

    def forward(self, input):
        # print(self.FLAGS.WIDTH_MULT_LIST)
        idx = self.FLAGS.WIDTH_MULT_LIST.index(self.width_mult)
        # print(idx)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels//self.groups, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        # print('weight',self.weight.shape)
        # print('out_channels',self.out_channels_list)
        # print('in_channels',self.in_channels_list)
        # print('group',self.groups_list)
        # print('weight',weight.shape)
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, FLAGS, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.FLAGS=FLAGS
        self.width_mult = max(self.FLAGS.WIDTH_MULT_LIST)

    def forward(self, input):
        idx = self.FLAGS.WIDTH_MULT_LIST.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

class SlimmableLinear_qkv(nn.Linear):
    def __init__(self, FLAGS, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear_qkv, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.max_out_feat = max(self.out_features_list) / 3
        self.q = 0
        self.k = int(self.max_out_feat)
        self.v = int(self.max_out_feat * 2)
        self.FLAGS=FLAGS
        self.width_mult = max(self.FLAGS.WIDTH_MULT_LIST)

    def forward(self, input):
        idx = self.FLAGS.WIDTH_MULT_LIST.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = int(self.out_features_list[idx] / 3)        
        weight = torch.cat((self.weight[:self.out_features,:self.in_features],self.weight[self.k:self.k +self.out_features,:self.in_features],self.weight[self.v:self.v + self.out_features,:self.in_features]),dim=0)
        if self.bias is not None:
            bias = self.bias[:self.out_features]
            bias = torch.cat((self.bias[:self.out_features],self.bias[self.k:self.k+self.out_features],self.bias[self.v:self.v+self.out_features]),dim=0)
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)