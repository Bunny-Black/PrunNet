#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import os
import torch
import torch.nn as nn

from timm.models import create_model
from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import (
    build_vit_sup_models, build_mocov3_model, 
    build_mae_model, build_vit_timm_cm_models
)
from .modules import MLP, ElasticBoundary, ReverseLayerF
from ..utils import logging
logger = logging.get_logger("FCLearning")

class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, model_cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        if model_cfg.TRANSFER_TYPE != "end2end" and "prompt" not in model_cfg.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False
        
        if model_cfg.TRANSFER_TYPE == "adapter":
            adapter_cfg = model_cfg.ADAPTER
        else:
            adapter_cfg = None

        self.build_backbone(model_cfg, adapter_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.setup_side(model_cfg)
        self.setup_head(model_cfg)
        self.projection_dim = 0
        if self.model_cfg.PROJECTION_LAYERS!=-1:
            self.setup_projector(model_cfg)
        else:
            self.projector = nn.Identity()

    def setup_projector(self, model_cfg):
        self.projector = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[model_cfg.PROJECTION_DIM] * model_cfg.PROJECTION_LAYERS + [model_cfg.PROJECTION_DIM],
            special_bias=True,
            final_norm=True
        )
    
    def setup_head(self, model_cfg):
        input_dim = model_cfg.PROJECTION_DIM if model_cfg.PROJECTION_LAYERS!=-1 else self.feat_dim
        self.projection_cls = MLP(
            input_dim=input_dim,
            mlp_dims=[int(model_cfg.DATA.NUMBER_CLASSES * model_cfg.DATA.TRAIN_RATIO)],
            special_bias=True
        )


    def setup_side(self, model_cfg):
        if model_cfg.TRANSFER_TYPE != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.enc.parameters(), self.enc_momentum.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def build_backbone(self, model_cfg, adapter_cfg, load_pretrain, vis):
        transfer_type = model_cfg.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            model_cfg.DATA.FEATURE, model_cfg.DATA.CROPSIZE, model_cfg.MODEL_ROOT, adapter_cfg, load_pretrain, vis)
        if model_cfg.MOMENTUM:
            self.enc_momentum, _ = build_vit_sup_models(
                model_cfg.DATA.FEATURE, model_cfg.DATA.CROPSIZE, model_cfg.MODEL_ROOT, adapter_cfg, load_pretrain, vis)
            for param_b, param_m in zip(self.enc.parameters(), self.enc_momentum.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

        # adapter
        if transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


    def forward(self, x_in, return_feature=False):
        if self.side is not None:
            side_output = self.side(x_in)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)
        
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x_in)  # batch_size x self.feat_dim

        if self.model_cfg.MOMENTUM:
            with torch.no_grad(), torch.cuda.amp.autocast():
                self._update_momentum_encoder(self.model_cfg.MOMENTUM_M)# TBD: following moco v3
                x_k = self.enc_momentum(x_in)
                x_k = self.projector(x_k)
        else:
            x_k = None

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output
            if self.model_cfg.MOMENTUM:
                raise ValueError("We do not support the combination of the side mode and momentum for now!")
        
        x = self.projector(x)
        y = self.projection_cls(x)
        if return_feature:
            return x, x_k, y
        return y
    
    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        if self.model_cfg.MOMENTUM:
            with torch.no_grad(), torch.cuda.amp.autocast():
                x_k = self.enc_momentum(x)
        else:
            x_k = None
        return x


class SSLViT(ViT):
    """moco-v3 and mae model."""

    def __init__(self, cfg, model_cfg, load_pretrain=True, vis=False):
        super(SSLViT, self).__init__(cfg, model_cfg, load_pretrain=load_pretrain, vis=vis)

    def build_backbone(self, model_cfg, adapter_cfg, load_pretrain, vis):
        if "moco" in model_cfg.DATA.FEATURE:
            build_fn = build_mocov3_model
        elif "mae" in model_cfg.DATA.FEATURE:
            build_fn = build_mae_model
        else:
            raise ValueError("unknown MODEL.DATA.FEATURE!")
        self.enc, self.feat_dim = build_fn(model_cfg.DATA.FEATURE, model_cfg.DATA.CROPSIZE,
            model_cfg.MODEL_ROOT, adapter_cfg=adapter_cfg)
        if model_cfg.MOMENTUM:
            self.enc_momentum, _ = build_fn(model_cfg.DATA.FEATURE, model_cfg.DATA.CROPSIZE,
            model_cfg.MODEL_ROOT, adapter_cfg=adapter_cfg)
            for param_b, param_m in zip(self.enc.parameters(), self.enc_momentum.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

        transfer_type = model_cfg.TRANSFER_TYPE
        
        if transfer_type == "end2end":
            logger.info("Enable all parameters update during training")
        
        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

class CM_TiMM_ViT(nn.Module):
    """ Support for adapter and prompt tuning 
    """
    def __init__(self, cfg, model_cfg, adversarial=False, eboundary=False,
                 old_embedding_dim = 256, load_pretrain=True):
        super(CM_TiMM_ViT, self).__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.adversarial = adversarial
        self.eboundary = eboundary
        self.old_embedding_dim = old_embedding_dim

        if model_cfg.TRANSFER_TYPE != "end2end" and "prompt" not in model_cfg.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False
        
        if model_cfg.TRANSFER_TYPE == "adapter" or "adapter+partial" in model_cfg.TRANSFER_TYPE:
            adapter_cfg = model_cfg.ADAPTER
        else:
            adapter_cfg = None

        self.build_backbone(model_cfg, adapter_cfg, load_pretrain)

        self.embedding_dim = self.feat_dim if self.cfg.NEW_MODEL.PROJECTION_LAYERS < 0 else self.cfg.NEW_MODEL.PROJECTION_DIM
        
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

    def build_backbone(self, model_cfg, adapter_cfg, load_pretrain):
        self.enc, self.feat_dim = build_vit_timm_cm_models(
            model_cfg.DATA.FEATURE, model_cfg.DATA.CROPSIZE, model_cfg.MODEL_ROOT, adapter_cfg, load_pretrain)
        if model_cfg.MOMENTUM:
            self.enc_momentum, _ = build_vit_timm_cm_models(
            model_cfg.DATA.FEATURE, model_cfg.DATA.CROPSIZE, model_cfg.MODEL_ROOT, adapter_cfg, load_pretrain)
            for param_b, param_m in zip(self.enc.parameters(), self.enc_momentum.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

        transfer_type = model_cfg.TRANSFER_TYPE
        
        if transfer_type == "end2end":
            logger.info("Enable all parameters update during training")
        
        elif transfer_type == "adapter+partial-1":# 
            total_layer = len(self.enc.blocks)
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            for k, p in self.enc.named_parameters():
                if "adapter" not in k and "blocks.{}".format(total_layer - 1) not in k and "norm." not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "adapter+partial-2":# 
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "adapter" not in k and "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "norm." not in k: # noqa
                    print(k, "turn to False")
                    p.requires_grad = False

        elif transfer_type == "adapter+partial-4":# 
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "adapter" not in k and "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and \
                   "blocks.{}".format(total_layer - 3) not in k and "blocks.{}".format(total_layer - 4) not in k and "norm." not in k: # noqa
                    p.requires_grad = False
        
        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if ("adapter" not in k) and ("lora_" not in k):
                    p.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.enc.parameters(), self.enc_momentum.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def setup_projector(self, model_cfg):
        self.projector = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[model_cfg.PROJECTION_DIM] * model_cfg.PROJECTION_LAYERS + [model_cfg.PROJECTION_DIM],
            special_bias=True,
            final_norm=True
        )
    
    def setup_head(self, model_cfg):
        input_dim = model_cfg.PROJECTION_DIM if model_cfg.PROJECTION_LAYERS!=-1 else self.feat_dim
        self.projection_cls = MLP(# self.projection_cls = MLP(   self.head = MLP(
            input_dim=input_dim,
            mlp_dims=[int(model_cfg.DATA.NUMBER_CLASSES * model_cfg.DATA.TRAIN_RATIO)],# 支持按比例划分数据集
            special_bias=True
        )

    def forward(self, x_in, feat_old=None, alpha=0, radius=None, return_feature=False):
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x_in)  # batch_size x self.feat_dim
        
        if self.model_cfg.MOMENTUM:
            with torch.no_grad(), torch.cuda.amp.autocast():
                self._update_momentum_encoder(self.model_cfg.MOMENTUM_M)# TBD: following moco v3
                x_k = self.enc_momentum(x_in)
                x_k = self.projector(x_k)
        else:
            x_k = None
        x = self.projector(x)

        if self.adversarial and self.training:
            if self.old_embedding_dim < self.embedding_dim:
                x_adversarial = x[:, :self.old_embedding_dim]
            else:
                x_adversarial = x
            reverse_feature_new = ReverseLayerF.apply(x_adversarial, alpha)
            model_out_new = self.discriminator(reverse_feature_new)
            model_out_old = self.discriminator(feat_old)

        y = self.projection_cls(x) # head(x)

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

class TiMM_ViT(nn.Module):
    def __init__(self, cfg, model_cfg, adversarial=False, eboundary=False,
                 old_embedding_dim = 256, load_pretrain=True):
        super(TiMM_ViT, self).__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.adversarial = adversarial
        self.eboundary = eboundary
        self.old_embedding_dim = old_embedding_dim
        # checkpoint_path = os.path.join(cfg.MODEL.MODEL_ROOT, 'checkpoints', cfg.DATA.FEATURE+'.pth')
        self.enc = create_model(
            model_cfg.DATA.FEATURE,
            pretrained = load_pretrain,
            num_classes = 0,
            drop_rate = model_cfg.DROP,
            drop_path_rate = model_cfg.DROP_PATH,
            drop_block_rate = None)

        if model_cfg.MOMENTUM:
            self.enc_momentum = create_model(
                model_cfg.DATA.FEATURE,
                pretrained = False,
                num_classes = 0,
                drop_rate = model_cfg.DROP,
                drop_path_rate = model_cfg.DROP_PATH,
                drop_block_rate = None)
            for param_b, param_m in zip(self.enc.parameters(), self.enc_momentum.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient
        
        assert(model_cfg.TRANSFER_TYPE in ["end2end", "fix_bias"] or "partial" in model_cfg.TRANSFER_TYPE), "Other transfer type for timm model isn't supported!"
        
        for k, p in self.enc.named_parameters():
            print(k, p.requires_grad)
            
        if model_cfg.TRANSFER_TYPE == "partial-1":# 只fine-tune encoder最后一层
            total_layer = len(self.enc.blocks)
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            for k, p in self.enc.named_parameters():
                if "adapter" not in k and "blocks.{}".format(total_layer - 1) not in k and "norm." not in k: # noqa
                    p.requires_grad = False

        elif model_cfg.TRANSFER_TYPE == "partial-2":# 只fine-tune encoder最后两层
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "adapter" not in k and "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "norm." not in k: # noqa
                    print(k, "turn to False")
                    p.requires_grad = False

        elif model_cfg.TRANSFER_TYPE == "partial-4":# 只fine-tune encoder最后四层
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "adapter" not in k and "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and \
                    "blocks.{}".format(total_layer - 3) not in k and "blocks.{}".format(total_layer - 4) not in k and "norm." not in k: # noqa
                    p.requires_grad = False
        
        elif model_cfg.TRANSFER_TYPE == "partial-6":# 只fine-tune encoder最后四层
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "adapter" not in k and "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and \
                    "blocks.{}".format(total_layer - 3) not in k and "blocks.{}".format(total_layer - 4) not in k and \
                    "blocks.{}".format(total_layer - 5) not in k and "blocks.{}".format(total_layer - 6) not in k and \
                    "norm." not in k: # noqa
                    p.requires_grad = False
        
        elif model_cfg.TRANSFER_TYPE == "partial-12":# 只fine-tune encoder最后四层
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "adapter" not in k and "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and \
                    "blocks.{}".format(total_layer - 3) not in k and "blocks.{}".format(total_layer - 4) not in k and \
                    "blocks.{}".format(total_layer - 5) not in k and "blocks.{}".format(total_layer - 6) not in k and \
                    "blocks.{}".format(total_layer - 7) not in k and "blocks.{}".format(total_layer - 8) not in k and \
                    "blocks.{}".format(total_layer - 9) not in k and "blocks.{}".format(total_layer - 10) not in k and \
                    "blocks.{}".format(total_layer - 11) not in k and "blocks.{}".format(total_layer - 12) not in k and \
                    "norm." not in k: # noqa
                    p.requires_grad = False
        elif model_cfg.TRANSFER_TYPE == "fix_bias":
            for k, p in self.enc.named_parameters():
                if "attn.qkv.bias" in k or "attn.proj.bias" in k:
                    p.requires_grad = False

        self.feat_dim = self.enc.embed_dim
        # print(self.enc.embed_dim)
        # print(self.enc.embed_dim)
        # print(self.enc.embed_dim)
        # print(self.enc.embed_dim)
        # exit()
        self.embedding_dim = self.feat_dim if self.cfg.NEW_MODEL.PROJECTION_LAYERS < 0 else self.cfg.NEW_MODEL.PROJECTION_DIM

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
            special_bias=True,
            final_norm=True
        )
    
    def setup_head(self, model_cfg):
        input_dim = model_cfg.PROJECTION_DIM if model_cfg.PROJECTION_LAYERS!=-1 else self.feat_dim
        self.projection_cls = MLP(
            input_dim=input_dim,
            mlp_dims=[int(model_cfg.DATA.NUMBER_CLASSES * model_cfg.DATA.TRAIN_RATIO)],# 支持按比例划分数据集
            special_bias=True
        )

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.enc.parameters(), self.enc_momentum.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
    
    def forward(self, x_in, feat_old=None, alpha=0, radius=None, return_feature=True):
        x = self.enc(x_in)  # batch_size x self.feat_dim

        if self.model_cfg.MOMENTUM:
            with torch.no_grad(), torch.cuda.amp.autocast():
                self._update_momentum_encoder(self.model_cfg.MOMENTUM_M)# TBD: following moco v3
                x_k = self.enc_momentum(x_in)
                if isinstance(x_k, tuple):
                    x_k =x_k[0]
                    x_k = self.projector(x_k)
        else:
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

