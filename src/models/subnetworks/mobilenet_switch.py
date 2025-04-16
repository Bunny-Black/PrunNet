import math
import torch.nn as nn
import torch
from .subnet import SubnetLinear,SubnetConv2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear, SwitchableBatchNorm1d, SwitchableBatchNorm2d

from ..modules import MLP,MLP_SWITCHNET

# class ConvBNReLU(nn.Sequential):#depthwise conv+BN+relu6，用于构建InvertedResidual。 
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1): 
#         #参数：输入的tensor的通道数，输出通道数，卷积核大小，卷积步距，输入与输出对应的块数（改为输入的层数就是depth wise conv了，详见https://blog.csdn.net/weixin_43572595/article/details/110563397） 
#         padding = (kernel_size - 1) // 2#根据卷积核大小获取padding大小 
#         super(ConvBNReLU, self).__init__( 
#             SubnetConv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False), 
#             #构建depthwise conv卷积，不使用偏置，因为后面有BN 
#             nn.BatchNorm2d(out_channel),#BN层，输入参数为输入的tensor的层数 
#             nn.ReLU6(inplace=True)#relu6激活函数，inplace原地操作tensor减少内存占用 
#         )
    
class ConvBNReLU(nn.Module):  # depthwise conv + BN + ReLU6
    def __init__(self, cfg, in_channel, out_channel, kernel_size=3, stride=1, groups_list=[1]):
        super(ConvBNReLU, self).__init__()
        # 计算 padding 大小
        padding = (kernel_size - 1) // 2
        
        # 定义各层
        self.conv = SlimmableConv2d(cfg.SNET, in_channels_list=in_channel, out_channels_list=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups_list=groups_list)
        self.bn = SwitchableBatchNorm2d(cfg.SNET,out_channel)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        # 前向传播
        x = self.conv(x)  # 进行卷积
        x = self.bn(x)    # 进行批归一化
        x = self.relu6(x) # 应用 ReLU6 激活函数
        return x
    
class InvertedResidual(nn.Module):#逆向残差结构 
    def __init__(self, cfg, in_channel, out_channel, stride, expand_ratio): 
        #参数：输入的tensor的通道数，输出的通道数，中间depthwise conv的步距，第一个1x1普通卷积的channel放大倍数 
        super(InvertedResidual, self).__init__() 
        hidden_channel = [i * expand_ratio for i in in_channel]#隐层的输入通道数，对应中间depthwise conv的输入通道数 
        
        self.use_shortcut = stride == 1 and in_channel == out_channel 
        #使用shortcut的条件：输入与输出shape一样，即没有缩放（stride=1）,维度一样（in_channel = out_channel） 

        layers = []#搜集各个层 
        if expand_ratio != 1:#这个是由于第一个卷积没有维度放大，因而不需要第一个1x1普通卷积 
            # 1x1 pointwise conv，第一个普通1x1卷积，升维 
            layers.append(ConvBNReLU(cfg, in_channel, hidden_channel, kernel_size=1)) 
        layers.extend([ 
            # 3x3 depthwise conv 
            ConvBNReLU(cfg,hidden_channel, hidden_channel, stride=stride, groups_list=hidden_channel), 
            # 1x1 pointwise conv(linear)第二个普通1x1卷积，降维 
            SlimmableConv2d(cfg.SNET,hidden_channel, out_channel, kernel_size=1, bias=False), 
            SwitchableBatchNorm2d(cfg.SNET,out_channel),#BN 
        ]) 
        # print(layers)

        self.conv = MySequential(*layers)#将上面的集成到一起 

    def forward(self, x):#前向传播过程，直接用上面弄好的conv层 
        if self.use_shortcut: 
            ret = self.conv(x) 
            ret += x
            return ret
        else: 
            ret = self.conv(x) 
            return ret 
        
def _make_divisible(ch, divisor=8, min_ch=None):#用于获取channel离divisor倍数最近的数，这个大概是为了方便底层调用的，具体原理我不懂 
    #参数：原来的通道数，谁的倍数，最小不能小于这个 
    """ 
    This function is taken from the original tf repo. 
    It ensures that all layers have a channel number that is divisible by 8 
    It can be seen here: 
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py 
    """ 
    if min_ch is None:#没输min_ch就默认为divisor 
        min_ch = divisor 
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)#获取通道数离divisor最近的倍数 
    # Make sure that round down does not go down by more than 10%. 
    if new_ch < 0.9 * ch: 
        new_ch += divisor 
    return new_ch

class MySequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.block_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.block_list:
            x = module(x) 
        return x
    
class Mobile_switch(nn.Module):#整体的网络 
    def __init__(self, cfg, num_classes=1000, alpha=1.0, round_nearest=8): 
        #参数：类的个数，各个层的深度缩放系数（width multiplier），深度数需要改成离其最近的倍数 
        super(Mobile_switch, self).__init__() 
        self.cfg = cfg
        block = InvertedResidual#将逆向卷积赋给block，改个名 
        input_channel = _make_divisible(32 * alpha, round_nearest) 
        channels = [
            _make_divisible(32 * alpha* width_mult)
            for width_mult in self.cfg.SNET.WIDTH_MULT_LIST]
        
        #获取原始图片进来后的卷积的输出通道数离round_nearest最近的倍数 
        last_channel = _make_divisible(1280 * alpha , round_nearest) 
        #获取特征提取部分最后的输出通道数离round_nearest最近的倍数 
        inverted_residual_setting = [ 
            #配置表，这些都是逆向残差层   t：维度放大倍数，c：该层的输出通道数，n：该层的重复次数，s：该层的中间的depthwise conv的步距（仅限第一个inverted_residual）。 
            # t, c, n, s 
            [1, 16, 1, 1], 
            [6, 24, 2, 2], 
            [6, 32, 3, 2], 
            [6, 64, 4, 2], 
            [6, 96, 3, 1], 
            [6, 160, 3, 2], 
            [6, 320, 1, 1], 
        ] 

        features = []#收集特征提取部分网络层 
        # conv1 layer，第一层，一个3x3普通卷积，步距为2 
        features.append(ConvBNReLU(self.cfg, [3 for _ in range(len(channels))], channels, stride=2)) 
        # building inverted residual residual blockes，遍历上面的配置，构建中间的逆向残差层 
        for t, c, n, s in inverted_residual_setting: 
            output_channel = _make_divisible(c * alpha, round_nearest) 
            outp = [
                _make_divisible(c * width_mult)
                for width_mult in self.cfg.SNET.WIDTH_MULT_LIST]
            #获取缩放后的通道数离round_nearest最近的倍数 
            for i in range(n):#构建n层逆向残差 
                stride = s if i == 0 else 1#只有第一层逆向残差层的depthwise conv才有步距不为1 
                features.append(block(self.cfg, channels, outp, stride, expand_ratio=t)) 
                channels = outp 
        self.feat_dim = last_channel
        # building last several layers，构建逆向残差后面的1x1普通卷积 
        features.append(ConvBNReLU(self.cfg, channels, [int(last_channel * width_mult) for width_mult in self.cfg.SNET.WIDTH_MULT_LIST], 1)) 
        # combine feature layers，整合特征提取部分的网络 
        self.features = MySequential(*features) 

        # building classifier，自适应池化层，参数是输出的tensor的大小 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        # self.classifier = nn.Sequential(#分类层，drop+全连接层 
        #     nn.Dropout(0.2), 
        #     nn.Linear(last_channel, num_classes) 
        # ) 

        # weight initialization，初始化各层 
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):#卷积层 
                nn.init.kaiming_normal_(m.weight, mode='fan_out')#权重用何凯明初始化 
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)#偏置初始化为0 
            elif isinstance(m, nn.BatchNorm2d):#BN，（x-bias）/weight 
                nn.init.ones_(m.weight)#初始化为1 
                nn.init.zeros_(m.bias)#初始化为0 
            elif isinstance(m, nn.Linear):#全连接层 
                nn.init.normal_(m.weight, 0, 0.01)#权重初始化为均值为0.方差为0.01的正态分布 
                nn.init.zeros_(m.bias)#偏置初始化为0 

        self.setup_head(cfg.MODEL)
        if self.cfg.MODEL.PROJECTION_LAYERS!=-1:
            self.setup_projector(cfg.MODEL)
        else:
            self.projector = nn.Identity()

    def setup_projector(self, model_cfg):
        if self.cfg.SUB_MODEL.USE_SWITCHNET:
            self.projector = MLP_SWITCHNET(
                cfg=self.cfg,
                FLAGS=self.cfg.SNET,
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
        input_dim = model_cfg.PROJECTION_DIM if model_cfg.PROJECTION_LAYERS!=-1 else self.feat_dim
        self.projection_cls = MLP(
            input_dim=input_dim,
            mlp_dims=[int(model_cfg.DATA.NUMBER_CLASSES * model_cfg.DATA.TRAIN_RATIO)],# 支持按比例划分数据集
            special_bias=True,bn_track=self.cfg.BN_TRACK
        )
        
                
    def forward(self,x,return_feature=True):#前向传播过程 
        x = self.features(x)#获得特征图 
        x = self.avgpool(x)#自适应池化，把每层大小从7x7变成了1x1 
        x = torch.flatten(x, 1)#展平操作，从dim=1开始 
        x = self.projector(x)
        y = self.projection_cls(x)
        if return_feature:
            return x, None, y
        return y
