#!/usr/bin/env python3
"""
Model construction functions.
"""
from tabnanny import verbose
import torch
from fvcore.common.checkpoint import Checkpointer
from copy import deepcopy

from .timm_resnet import ResNet
from .vit_models import ViT, SSLViT, TiMM_ViT, CM_TiMM_ViT
from .subnetworks.resnet18_group import SubnetResNet18_group
from .subnetworks.resnet18 import SubnetResNet18
from .subnetworks.resnext_sub import Resnext_sub
from .subnetworks.resnext_switchnet import Resnext_switch
from .subnetworks.mobilenet_switch import Mobile_switch
from .subnetworks.convnext import Our_Convnext
from .subnetworks.resnet50 import SubnetResNet50
from .subnetworks.switchnet import build_switch_resnet_backbone
from .subnetworks.mobilenet_sub import MobileNetV2
# from .subnetworks.subnet_resnet import build_sub_resnet_backbone
from .vit_backbones.vit_ours import build_vit_ours as VIT_ours
from .vit_backbones.vit_timm_ours import Our_Timm_VIT
from .vit_backbones.vit_timm_switch import Our_Timm_VIT_Switch
from .vit_backbones.vit_timm_sub import Our_Timm_VIT as Our_Timm_VIT_sub
from .vit_backbones.swin_vit_sub import Our_Swin_VIT
from ..utils import logging
logger = logging.get_logger("FCLearning")
# Supported model types
_MODEL_TYPES = {
    "resnet": ResNet,
    "vit": ViT,
    "ssl-vit": SSLViT,
    "timm-vit": TiMM_ViT,
    'swin-vit': Our_Swin_VIT,
    "cm-timm-vit": CM_TiMM_ViT,# cm: custom-made
    'sub_resnet18': SubnetResNet18,
    'sub_resnet50': SubnetResNet50,
    'our_vit': VIT_ours,
    'our_timm_vit': Our_Timm_VIT,
    'our_mobilenet': MobileNetV2,
    'resnet18_group': SubnetResNet18_group,
    'resnext': Resnext_sub,
    'convnext': Our_Convnext,
    'resnext_switch': Resnext_switch,
    'mobile_switch': Mobile_switch,
}

def build_sub_models(cfg,sparsity,model=None):
    sub_model_list = []
    if model==None:
        if cfg.MODEL.TYPE == 'our_timm_vit':
            if cfg.SUB_MODEL.USE_SWITCHNET:
                model = Our_Timm_VIT_Switch(cfg,pretrained=cfg.MODEL.PRETRAIN)
            else:
                model = Our_Timm_VIT_sub(cfg,pretrained=cfg.MODEL.PRETRAIN)
        elif cfg.MODEL.TYPE == 'swin-vit':
            model = Our_Swin_VIT(cfg,pretrained=cfg.MODEL.PRETRAIN)
        elif cfg.MODEL.TYPE == 'convnext':
            model = Our_Convnext(cfg,pretrained=cfg.MODEL.PRETRAIN,in_22k=True)
        elif cfg.MODEL.TYPE == 'resnext':
            model = Resnext_sub(cfg)        
        elif cfg.MODEL.TYPE == 'resnext_switch':
            model = Resnext_switch(cfg)       
        elif cfg.MODEL.TYPE == 'mobile_switch' :
            model = Mobile_switch(cfg)
        elif cfg.MODEL.TYPE == 'our_mobilenet':
            model = MobileNetV2(cfg)
        elif cfg.MODEL.TYPE == 'our_vit':
            model = VIT_ours(cfg)
        elif cfg.MODEL.TYPE == 'timm-vit':
            model = TiMM_ViT(cfg,cfg.MODEL,load_pretrain=True)
        elif cfg.SUB_MODEL.USE_SWITCHNET:
            model = build_switch_resnet_backbone(cfg)
        elif cfg.MODEL.TYPE == 'resnet18_group':
            model = SubnetResNet18_group(cfg,cfg.MODEL,nf=64,sparsity=0)
        else:
            if cfg.MODEL.DATA.FEATURE == 'resnet18':
                model = SubnetResNet18(cfg,cfg.MODEL,nf=64,sparsity=0)
            elif cfg.MODEL.DATA.FEATURE == 'resnet50':
                model = SubnetResNet50(cfg,cfg.MODEL,nf=64,sparsity=0)
                
        model, device = load_model_to_device(model, cfg)
        return model,device
    else:
        if not isinstance(sparsity,list):
            return get_submodel(model,sparsity=sparsity[0],cfg=cfg)
        else:
            for sparsity_i in sparsity:
                if sparsity_i != 1.0:
                    sub_model = get_submodel(model,sparsity=sparsity_i,cfg=cfg)
                    sub_model_list.append(sub_model)
            return sub_model_list


def build_model(cfg, model_cfg, adversarial=False, eboundary=False, old_feat_dim=-1):
    """
    build model here
    """
    assert (
        model_cfg.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(model_cfg.TYPE)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    train_type = model_cfg.TYPE
    model = _MODEL_TYPES[train_type](cfg, model_cfg, adversarial, eboundary, old_feat_dim)

    log_model_info(model, verbose=cfg.DBG)

    if cfg.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")
    return model, device


def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    backbone_total_params = sum(p.numel() for n, p in model.named_parameters() if 'enc.' in n)
    backbone_grad_params = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and 'enc.' in n)
    head_params = sum(p.numel() for n, p in model.named_parameters() if 'enc.' not in n)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    logger.info("Total Backbone Parameters: {0}\t Gradient Backbone Parameters: {1}\t LoRA Parameters: {2}".format(
        backbone_total_params, backbone_grad_params, lora_params))
    logger.info("tuned percent:%.3f"%(backbone_grad_params/backbone_total_params*100))
    logger.info("LoRA percent:%.3f"%(lora_params/backbone_total_params*100))
    logger.info("Head (and neck) Parameters: {0}".format(head_params))


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device

def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=True,
            )
    else:
        model = model.to(cur_device)
    return model, cur_device

def set_bn_track(model,cfg):
    bn_track = cfg.BN_TRACK
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d) or \
           isinstance(module, torch.nn.BatchNorm2d) or \
           isinstance(module, torch.nn.BatchNorm3d):
            module.track_running_stats = bn_track
    return model


def get_submodel(model,sparsity,cfg=None):
    if cfg.MODEL.TYPE not in ['our_timm_vit']:
        sub_model = deepcopy(model)
    else:
        sub_model = Our_Timm_VIT_sub(cfg,pretrained=False)
        sub_model, device = load_model_to_device(sub_model, cfg)
        sub_model.load_state_dict(model.state_dict())

    sub_model.sparsity = sparsity
    return sub_model