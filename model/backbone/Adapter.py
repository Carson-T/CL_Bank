import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 d_model=768,
                 bottleneck=64,
                 dropout=0.0,
                 nonlinear=nn.ReLU,
                 is_ss=False,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option=None):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.is_ss = is_ss
        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nonlinear()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        if self.is_ss:
            self.alpha = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.ones(d_model))
        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        if self.is_ss:
            x = self.alpha*x + self.beta
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
