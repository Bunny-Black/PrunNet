import timm
import torch.nn as nn
import torch
import os.path as osp
from .modules import MLP, ElasticBoundary, ReverseLayerF
class ResNet(nn.Module):
    def __init__(self,cfg,model_cfg,adversarial=False, eboundary=False,
                 old_embedding_dim = 256, load_pretrain=True):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.adversarial = adversarial
        self.eboundary = eboundary
        self.old_embedding_dim = old_embedding_dim
        self.embedding_dim = model_cfg.PROJECTION_DIM
        self.arch = model_cfg.DATA.FEATURE
        if model_cfg.DATA.FEATURE in timm.list_models(pretrained=True):
            if self.arch == 'resnet50': 
                self.enc = timm.create_model(self.arch, pretrained=False, checkpoint_path = "pretrained_resnet/checkpoint/resnet50_ram-a26f946b.pth") #模型下载不下来,就用本地的了
                self.enc.fc = nn.Identity()
            else:
                self.enc = timm.create_model(self.arch, pretrained=True, num_classes=0)
        else:
            raise Exception("not supported model architecture")

        self.feat_dim = self.enc.num_features
        self.setup_head(model_cfg)
        if self.cfg.MODEL.PROJECTION_LAYERS!=-1:
            self.setup_projector(model_cfg)
        else:
            self.projector = nn.Identity()
        
        if self.adversarial:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('d_fc1', nn.Linear(old_embedding_dim, 100))
            self.discriminator.add_module('d_bn1', nn.BatchNorm1d(100))
            self.discriminator.add_module('d_relu1', nn.ReLU(True))
            self.discriminator.add_module('d_fc2', nn.Linear(100, 2))
            self.discriminator.add_module('d_softmax', nn.LogSoftmax(dim=1))
        if self.eboundary:
            self.eboundary = ElasticBoundary(self.cfg.NEW_MODEL.DATA.NUMBER_CLASSES)

    def setup_projector(self, model_cfg):
        self.projector = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[model_cfg.PROJECTION_DIM] * model_cfg.PROJECTION_LAYERS + [model_cfg.PROJECTION_DIM],
            special_bias=False,final_norm = True
        )
        # self.projector = nn.Linear(self.feat_dim, model_cfg.PROJECTION_DIM, bias=False)
    
    def setup_head(self, model_cfg):
        # input_dim = model_cfg.PROJECTION_DIM if model_cfg.PROJECTION_LAYERS!=-1 else self.feat_dim
        self.projection_cls = MLP(
            input_dim=model_cfg.PROJECTION_DIM,
            mlp_dims=[int(model_cfg.DATA.NUMBER_CLASSES * model_cfg.DATA.TRAIN_RATIO)],
            special_bias=False
        )
        # self.projection_cls = nn.Linear(model_cfg.PROJECTION_DIM, model_cfg.DATA.NUMBER_CLASSES, bias=False)

    def forward(self, x_in, feat_old=None, alpha=0, radius=None, return_feature=False):
        print(x_in.shape)
        x = self.enc(x_in).view(-1, self.feat_dim)  # batch_size x self.feat_dim
        x_k = None
        if isinstance(x, tuple):
            x = x[0]

        x = self.projector(x)

        if self.adversarial and self.training:
            if self.old_embedding_dim < self.embedding_dim:
                x_adversarial = x[:, :self.old_embedding_dim]
            else:
                x_adversarial = x
            reverse_feature_new = ReverseLayerF.apply(x_adversarial, alpha)
            model_out_new = self.discriminator(reverse_feature_new)
            model_out_old = self.discriminator(feat_old)
        

        y = self.projection_cls(x)
        
        if self.training:
            if self.adversarial:# 默认返回feature
                if self.eboundary and radius is not None:
                    radius = self.eboundary(radius)
                    return x, x_k, y, model_out_new, model_out_old, radius
                else:
                    return x, x_k, y, model_out_new, model_out_old
            elif self.eboundary and radius is not None:
                radius = self.eboundary(radius)
                return x, x_k, y, radius
            
        if return_feature:
            return x, x_k, y
        return y

