#!/usr/bin/env python3
'''
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import Attention
from timm.models.vision_transformer import Block

from ...utils import logging
from .adapter import Adapter, LoRA_Adapter
from .timm_block import LoRA_Attention, LoRA_Block
logger = logging.get_logger("FCLearning")


class Pfeiffer_Block(Block):

    def __init__(self, adapter_config, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super(Pfeiffer_Block, self).__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            drop=drop, 
            attn_drop=attn_drop,
            drop_path=drop_path, 
            act_layer=act_layer, 
            norm_layer=norm_layer)
        
        self.adapter_config = adapter_config

        if adapter_config.STYLE == "Pfeiffer":
            self.adapter_downsample = nn.Linear(
                dim,
                dim // adapter_config.REDUCATION_FACTOR
            )
            self.adapter_upsample = nn.Linear(
                dim // adapter_config.REDUCATION_FACTOR,
                dim
            )
            self.adapter_act_fn = act_layer()

            nn.init.zeros_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)

            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)
        else:
            raise ValueError("Other adapter styles are not supported.")

    def forward(self, x):

        if self.adapter_config.STYLE == "Pfeiffer":
            # same as reguluar ViT block
            h = x
            x = self.norm1(x)
            x = self.attn(x)
            x = self.drop_path(x)
            x = x + h

            h = x
            x = self.norm2(x)
            x = self.mlp(x)

            # start to insert adapter layers...
            adpt = self.adapter_downsample(x)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_upsample(adpt)
            x = adpt + x
            # ...end

            x = self.drop_path(x)
            x = x + h 
            
            return x

class PA_FFN_Block(Block):
    """
    Support for no adapter in model structure
    """
    def __init__(self, adapter_config, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(PA_FFN_Block, self).__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            drop=drop, 
            attn_drop=attn_drop,
            drop_path=drop_path, 
            act_layer=act_layer, 
            norm_layer=norm_layer)
        
        self.adapter_config = adapter_config

        if adapter_config.TYPE == "LoRA":
            self.adapter = LoRA_Adapter(adapter_config, d_model=dim)
        elif adapter_config.TYPE == "normal":
            self.adapter = Adapter(adapter_config, d_model=dim)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.adapter_config.STYLE == "parallel_ffn":
            adapter_x = self.adapter(x, add_residual=False)

        residual = x
        x = self.drop_path(self.mlp(self.norm2(x)))
        
        if self.adapter_config.STYLE == "sequential_ffn":
            x = self.adapter(x, add_residual=True)
        elif self.adapter_config.STYLE == "parallel_ffn":
            x += adapter_x

        x = residual + x
        return x
    
class PA_FFN_LoRA_Attn_Block(LoRA_Block):
    """
    Support for no adapter in model structure
    """
    def __init__(self, adapter_config, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, r=0, lora_alpha=1, lora_dropout=0.):
        super(PA_FFN_LoRA_Attn_Block, self).__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            drop=drop, 
            attn_drop=attn_drop,
            drop_path=drop_path, 
            act_layer=act_layer, 
            norm_layer=norm_layer,
            r = adapter_config["LoRA_R"], 
            lora_alpha = adapter_config["LoRA_ALPHA"], 
            lora_dropout = adapter_config["LoRA_DROPOUT"])
        
        self.adapter_config = adapter_config

        if adapter_config.STYLE in ["parallel_ffn_lora_attn", "sequential_ffn_lora_attn"]:
            self.adapter = Adapter(adapter_config, d_model=dim)
            

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.adapter_config.STYLE in "parallel_ffn_lora_attn":
            adapter_x = self.adapter(x, add_residual=False)

        residual = x
        x = self.drop_path(self.mlp(self.norm2(x)))
        
        if self.adapter_config.STYLE == "sequential_ffn_lora_attn":
            x = self.adapter(x, add_residual=True)
        elif self.adapter_config.STYLE == "parallel_ffn_lora_attn":
            x += adapter_x

        x = residual + x
        return x


class PA_Attn_FFN_Block(Block):
    """
    Support for no adapter in model structure
    """
    def __init__(self, adapter_config, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(PA_Attn_FFN_Block, self).__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            drop=drop, 
            attn_drop=attn_drop,
            drop_path=drop_path, 
            act_layer=act_layer, 
            norm_layer=norm_layer)
        
        self.adapter_config = adapter_config

        if adapter_config.TYPE == "LoRA":
            self.ffn_adapter = LoRA_Adapter(adapter_config, d_model=dim, isattn=False)
            self.attn_adapter = LoRA_Adapter(adapter_config, d_model=dim, isattn=True)
        elif adapter_config.TYPE == "normal":
            self.ffn_adapter = Adapter(adapter_config, d_model=dim, isattn=False)
            self.attn_adapter = Adapter(adapter_config, d_model=dim, isattn=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        # attn block
        if self.adapter_config.STYLE == "parallel_attn_ffn":
            adapter_x = self.attn_adapter(x, add_residual=False)
        
        residual = x
        x = self.drop_path(self.attn(self.norm1(x)))
        if self.adapter_config.STYLE == "sequential_attn_ffn":
            x = self.attn_adapter(x, add_residual=True)
        elif self.adapter_config.STYLE == "parallel_attn_ffn":
            x += adapter_x
        x = residual + x

        # ffn block
        if self.adapter_config.STYLE == "parallel_attn_ffn":
            adapter_x = self.ffn_adapter(x, add_residual=False)

        residual = x
        x = self.drop_path(self.mlp(self.norm2(x)))
        
        if self.adapter_config.STYLE == "sequential_attn_ffn":
            x = self.ffn_adapter(x, add_residual=True)
        elif self.adapter_config.STYLE == "parallel_attn_ffn":
            x += adapter_x

        x = residual + x
        return x