#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from functools import partial

import torch, math
import torch.nn as nn

import numpy as np
from scipy import ndimage
from timm.models.vision_transformer import VisionTransformer, _cfg

from .adapter_block import Pfeiffer_Block, PA_FFN_Block, PA_Attn_FFN_Block, PA_FFN_LoRA_Attn_Block
from timm.models.layers import PatchEmbed
from ...utils import logging
logger = logging.get_logger("FCLearning")


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    #_logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    #_logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def _n2p(weight, transposed=True):
        if weight.ndim == 4 and weight.shape[0] == weight.shape[1] == weight.shape[2] == 1:
            weight = weight.flatten()
        if transposed:
            if weight.ndim == 4:
                weight = weight.transpose([3, 2, 0, 1])
            elif weight.ndim == 3:
                weight = weight.transpose([2, 0, 1])
            elif weight.ndim == 2:
                weight = weight.transpose([1, 0])
        return torch.from_numpy(weight)

class ADPT_TiMM_VisionTransformer(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, 
                 adapter_cfg, 
                 img_size = 224, 
                 patch_size = 16, 
                 in_chans = 3, 
                 num_classes = 1000, 
                 embed_dim = 768, 
                 depth = 12,
                 num_heads = 12, 
                 mlp_ratio = 4., 
                 qkv_bias = True, 
                 representation_size = None, 
                 distilled = False,
                 drop_rate = 0., 
                 attn_drop_rate = 0., 
                 drop_path_rate = 0., 
                 embed_layer = PatchEmbed, 
                 norm_layer = None,
                 act_layer = None, 
                 weight_init = '',
                 **kwargs):
        super(ADPT_TiMM_VisionTransformer, self).__init__(
                 img_size = img_size, 
                 patch_size = patch_size, 
                 in_chans = in_chans, 
                 num_classes = num_classes, 
                 embed_dim = embed_dim, 
                 depth = depth,
                 num_heads = num_heads, 
                 mlp_ratio = mlp_ratio, 
                 qkv_bias = qkv_bias, 
                 representation_size = None, 
                 distilled = False,
                 drop_rate = 0., 
                 attn_drop_rate = 0., 
                 drop_path_rate = 0., 
                 embed_layer = PatchEmbed, 
                 norm_layer = None,
                 act_layer = None, 
                 weight_init = '',
                 **kwargs)
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        if adapter_cfg['STYLE'] == "Pfeiffer":
            self.blocks = nn.Sequential(*[
                Pfeiffer_Block(
                    adapter_config = adapter_cfg, 
                    dim = embed_dim, 
                    num_heads = num_heads, 
                    mlp_ratio = mlp_ratio, 
                    qkv_bias = qkv_bias, 
                    drop = drop_rate,
                    attn_drop = attn_drop_rate, 
                    drop_path = dpr[i], 
                    norm_layer = norm_layer, 
                    act_layer = act_layer) for i in range(depth)])
            
        elif adapter_cfg['STYLE'] in ["parallel_ffn", "sequential_ffn"]:
            self.blocks = nn.Sequential(*[
                PA_FFN_Block(
                    adapter_config = adapter_cfg, 
                    dim = embed_dim, 
                    num_heads = num_heads, 
                    mlp_ratio = mlp_ratio, 
                    qkv_bias = qkv_bias, 
                    drop = drop_rate,
                    attn_drop = attn_drop_rate, 
                    drop_path = dpr[i], 
                    norm_layer = norm_layer, 
                    act_layer = act_layer) for i in range(depth)])
            
        elif adapter_cfg['STYLE'] in ["parallel_attn_ffn", "sequential_attn_ffn"]:
            self.blocks = nn.Sequential(*[
                PA_Attn_FFN_Block(
                    adapter_config = adapter_cfg, 
                    dim = embed_dim, 
                    num_heads = num_heads, 
                    mlp_ratio = mlp_ratio, 
                    qkv_bias = qkv_bias, 
                    drop = drop_rate,
                    attn_drop = attn_drop_rate, 
                    drop_path = dpr[i], 
                    norm_layer = norm_layer, 
                    act_layer = act_layer) for i in range(depth)])
        
        elif adapter_cfg['STYLE'] in ["parallel_ffn_lora_attn", ]:
            self.blocks = nn.Sequential(*[
                PA_FFN_LoRA_Attn_Block(
                    adapter_config = adapter_cfg, 
                    dim = embed_dim, 
                    num_heads = num_heads, 
                    mlp_ratio = mlp_ratio, 
                    qkv_bias = qkv_bias, 
                    drop = drop_rate,
                    attn_drop = attn_drop_rate, 
                    drop_path = dpr[i], 
                    norm_layer = norm_layer, 
                    act_layer = act_layer) for i in range(depth)])
        
        else:
            raise ValueError("Other adapter styles are not supported.")

    @torch.no_grad()
    def load_from(self, weights, prefix=''):
        if not prefix and 'opt/target/embedding/kernel' in weights:
            prefix = 'opt/target/'
        
        embed_conv_w = adapt_input_conv(self.patch_embed.proj.weight.shape[1], _n2p(weights[f'{prefix}embedding/kernel']))
        self.patch_embed.proj.weight.copy_(embed_conv_w)
        self.patch_embed.proj.bias.copy_(_n2p(weights[f'{prefix}embedding/bias']))
        self.cls_token.copy_(_n2p(weights[f'{prefix}cls'], transposed=False))
        pos_embed_w = _n2p(weights[f'{prefix}Transformer/posembed_input/pos_embedding'], transposed=False)
        if pos_embed_w.shape != self.pos_embed.shape:
            pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w, self.pos_embed, getattr(self, 'num_tokens', 1), self.patch_embed.grid_size)
        self.pos_embed.copy_(pos_embed_w)
        self.norm.weight.copy_(_n2p(weights[f'{prefix}Transformer/encoder_norm/scale']))
        self.norm.bias.copy_(_n2p(weights[f'{prefix}Transformer/encoder_norm/bias']))
        if isinstance(self.head, nn.Linear) and self.head.bias.shape[0] == weights[f'{prefix}head/bias'].shape[-1]:
            self.head.weight.copy_(_n2p(weights[f'{prefix}head/kernel']))
            self.head.bias.copy_(_n2p(weights[f'{prefix}head/bias']))
        if isinstance(getattr(self.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in weights:
            self.pre_logits.fc.weight.copy_(_n2p(weights[f'{prefix}pre_logits/kernel']))
            self.pre_logits.fc.bias.copy_(_n2p(weights[f'{prefix}pre_logits/bias']))
        for i, block in enumerate(self.blocks.children()):
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
            block.norm1.weight.copy_(_n2p(weights[f'{block_prefix}LayerNorm_0/scale']))
            block.norm1.bias.copy_(_n2p(weights[f'{block_prefix}LayerNorm_0/bias']))
            block.attn.qkv.weight.copy_(torch.cat([
                _n2p(weights[f'{mha_prefix}{n}/kernel'], transposed=False).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.qkv.bias.copy_(torch.cat([
                _n2p(weights[f'{mha_prefix}{n}/bias'], transposed=False).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.proj.weight.copy_(_n2p(weights[f'{mha_prefix}out/kernel']).flatten(1))
            block.attn.proj.bias.copy_(_n2p(weights[f'{mha_prefix}out/bias']))
            for r in range(2):
                getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(weights[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
                getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(weights[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
            block.norm2.weight.copy_(_n2p(weights[f'{block_prefix}LayerNorm_2/scale']))
            block.norm2.bias.copy_(_n2p(weights[f'{block_prefix}LayerNorm_2/bias']))


"""
if __name__ == '__main__':
    def vit_base_patch16(adapter_cfg, **kwargs):
        model = ADPT_VisionTransformer(
            adapter_cfg,
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.load_from(np.load("/he_zy/zikun/code_remote/FCL_ViT/pretrained_vit/checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"))
        return model

    adapter_cfg = {'STYLE': "Pfeiffer"}
    model = vit_base_patch16(adapter_cfg, )
    print(model)
    model.load_from(np.load("/he_zy/zikun/code_remote/FCL_ViT/pretrained_vit/checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"))
    for name, param in model.named_parameters():
        print(name)
"""