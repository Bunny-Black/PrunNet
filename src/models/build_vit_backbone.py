#!/usr/bin/env python3
import numpy as np
import torch
import os
from .vit_backbones.swin_transformer import SwinTransformer
from .vit_backbones.vit import VisionTransformer
from .vit_backbones.vit_moco import vit_base, vit_small
from .vit_backbones.vit_mae import build_model as mae_vit_model

from .vit_prompt.vit import PromptedVisionTransformer
from .vit_prompt.swin_transformer import PromptedSwinTransformer
from .vit_prompt.vit_moco import vit_base as prompt_vit_base
from .vit_prompt.vit_moco import vit_small as prompt_vit_small
from .vit_prompt.vit_mae import build_model as prompt_mae_vit_model

from .vit_adapter.vit_mae import build_model as adapter_mae_vit_model
from .vit_adapter.vit_moco import vit_small as adapter_vit_small
from .vit_adapter.vit_moco import vit_base as adapter_vit_base
from .vit_adapter.vit_timm import ADPT_TiMM_VisionTransformer
from .vit_adapter.vit import ADPT_VisionTransformer
MODEL_ZOO = {
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224":  "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
    "mae_vith14": "mae_pretrain_vit_huge.pth",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mae_vitl16": "mae_pretrain_vit_large.pth",
    "mocov3_vitb": "mocov3_linear-vit-b-300ep.pth.tar",
    "mocov3_vits": "mocov3_linear-vit-s-300ep.pth.tar"
}


def build_mae_model(model_type, crop_size, 
                    model_root=None, adapter_cfg=None):
    if adapter_cfg is not None:
        model = adapter_mae_vit_model(model_type, adapter_cfg)
    else:
        model = mae_vit_model(model_type)
    out_dim = model.embed_dim

    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']

    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_mocov3_model(model_type, crop_size,
                       model_root=None, adapter_cfg=None):
    if "mocov3" not in model_type:
        raise ValueError("Does not support other arch")
    if "vitb" in model_type:
        if adapter_cfg is not None:
            model = adapter_vit_base(adapter_cfg)
        else:
            model = vit_base()
        out_dim = 768
    elif "vits" in model_type:
        if adapter_cfg is not None:
            model = adapter_vit_small(adapter_cfg)
        else:
            model = vit_small()
        out_dim = 384
    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_vit_timm_cm_models(model_type, crop_size, 
                             model_root=None, adapter_cfg=None, pretrained=True):
    m2ckpt_name = {
        "vit_tiny_patch16_224_in21k": "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
        "vit_small_patch16_224_in21k": "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
        "vit_base_patch16_224_in21k": "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
        "vit_base_patch8_224_in21k": "B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
        "vit_large_patch16_224_in21k": "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz",
        "vit_huge_patch14_224_in21k": "ViT-H_14.npz"
    }

    if adapter_cfg is not None:
        if model_type == "vit_tiny_patch16_224_in21k":
            model = ADPT_TiMM_VisionTransformer(adapter_cfg, patch_size=16, embed_dim=192, depth=12, num_heads=3)
            out_dim = 192

        elif model_type == "vit_small_patch16_224_in21k":
            model = ADPT_TiMM_VisionTransformer(adapter_cfg, patch_size=16, embed_dim=384, depth=12, num_heads=6)
            out_dim = 384

        elif model_type == "vit_base_patch16_224_in21k":
            model = ADPT_TiMM_VisionTransformer(adapter_cfg, patch_size=16, embed_dim=768, depth=12, num_heads=12)
            out_dim = 768
        
        elif model_type == "vit_large_patch16_224_in21k":
            model = ADPT_TiMM_VisionTransformer(adapter_cfg, patch_size=16, embed_dim=1024, depth=24, num_heads=16)
            out_dim = 1024

        elif model_type == "vit_huge_patch14_224_in21k":
            model = ADPT_TiMM_VisionTransformer(adapter_cfg, patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280)
            out_dim = 1280

        else:
            raise ValueError("Does not support other arch")
    if pretrained:
        ckpt = os.path.join(model_root, "checkpoints", m2ckpt_name[model_type])
        model.load_from(np.load(ckpt))
        
    model.head = torch.nn.Identity()
    return model, out_dim

def build_vit_sup_models(model_type, crop_size, 
                         model_root=None, adapter_cfg=None, 
                         load_pretrain=True, vis=False):
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb8_imagenet21k": 768,
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    }
    if adapter_cfg is not None:
        model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1, adapter_cfg=adapter_cfg)

    else:
        model = VisionTransformer(
            model_type, crop_size, num_classes=-1, vis=vis)
    
    if load_pretrain:
        model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]

