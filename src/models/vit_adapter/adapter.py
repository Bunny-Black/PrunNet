# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 adapter_config=None,
                 d_model=None, 
                 isattn=False):
        super().__init__()
        self.n_embd = d_model
        if isattn:
            self.down_size = d_model // adapter_config.ATTN_REDUCATION_FACTOR
        else:
            self.down_size = d_model // adapter_config.REDUCATION_FACTOR

        #_before
        self.adapter_layernorm_option = adapter_config.ADAPTER_LAYERNORM_OPTION

        self.adapter_layer_norm_before = None
        if self.adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_config.ADAPTER_SCALAR == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        elif isattn:
            self.scale = float(adapter_config.ATTN_ADAPTER_SCALAR)
        else:
            self.scale = float(adapter_config.ADAPTER_SCALAR)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = adapter_config.DROPOUT
        if adapter_config.INIT_OPTION == "bert":
            raise NotImplementedError
        elif adapter_config.INIT_OPTION == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                #nn.init.zeros_(self.down_proj.weight)
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
    
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

class LoRA_Adapter(nn.Module, LoRALayer):
    def __init__(self,
                 adapter_config=None,
                 d_model=None,
                 isattn=False):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, 
                           r=adapter_config.LoRA_R,
                           lora_alpha=adapter_config.LoRA_ALPHA, 
                           lora_dropout=adapter_config.LoRA_DROPOUT)
        
        self.n_embd = d_model
        self.head_num = adapter_config.LoRA_HEADS
        if isattn:
            self.down_size = d_model // adapter_config.ATTN_REDUCATION_FACTOR
        else:
            self.down_size = d_model // adapter_config.REDUCATION_FACTOR

        #_before
        self.adapter_layernorm_option = adapter_config.ADAPTER_LAYERNORM_OPTION

        self.adapter_layer_norm_before = None
        if self.adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_config.ADAPTER_SCALAR == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        elif isattn:
            self.scale = float(adapter_config.ATTN_ADAPTER_SCALAR)
        else:
            self.scale = float(adapter_config.ADAPTER_SCALAR)

        assert self.r > 0, "LoRA Adapter requires a postive rank!"

        self.down_lora_Ps = nn.ModuleList([nn.Linear(self.n_embd, self.r, bias=False) for i in range(self.head_num)])
        self.up_lora_Qs = nn.ModuleList([nn.Linear(self.r, self.n_embd, bias=False) for i in range(self.head_num)])
        if self.r != self.down_size:
            self.down_lora_Qs = nn.ModuleList([nn.Linear(self.r, self.down_size, bias=False) for i in range(self.head_num)])
            self.lora_Ks = nn.ModuleList([nn.Linear(self.r, self.r, bias=False) for i in range(self.head_num)])
            self.up_lora_Ps = nn.ModuleList([nn.Linear(self.down_size, self.r, bias=False) for i in range(self.head_num)])
        else:
            self.down_lora_Qs = nn.ModuleList([nn.Identity() for i in range(self.head_num)])
            self.lora_Ks = nn.ModuleList([nn.Identity() for i in range(self.head_num)])
            self.up_lora_Ps = nn.ModuleList([nn.Identity() for i in range(self.head_num)])
            
        self.down_bias = nn.Parameter(self.down_lora_Ps[0].weight.new_zeros(1, 1, self.down_size))
        self.up_bias = nn.Parameter(self.up_lora_Qs[0].weight.new_zeros(1, 1, self.n_embd))
        self.non_linear_func = nn.ReLU()

        self.dropout = adapter_config.DROPOUT
        if adapter_config.INIT_OPTION == "bert":
            raise NotImplementedError
        elif adapter_config.INIT_OPTION == "lora":
            with torch.no_grad():
                for i in range(self.head_num):
                    nn.init.kaiming_uniform_(self.down_lora_Ps[i].weight, a=math.sqrt(5))
                    nn.init.zeros_(self.up_lora_Qs[i].weight)
                    if self.r != self.down_size:
                        nn.init.zeros_(self.down_lora_Qs[i].weight)
                        nn.init.zeros_(self.lora_Ks[i].weight)
                        nn.init.zeros_(self.up_lora_Ps[i].weight)
                        

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        # down
        down = torch.sum(torch.stack([self.down_lora_Qs[i](self.lora_Ks[i](self.down_lora_Ps[i](x))) for i in range(self.head_num)], 
                                     dim=-1), dim = -1) + self.down_bias
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        up = torch.sum(torch.stack([self.up_lora_Qs[i](self.lora_Ks[i](self.up_lora_Ps[i](down))) for i in range(self.head_num)], 
                                   dim = -1), dim = -1) + self.up_bias
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output