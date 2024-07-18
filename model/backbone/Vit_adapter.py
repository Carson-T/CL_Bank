import math
import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed
from timm.models import register_model
from functools import partial
from collections import OrderedDict
import copy


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ffn_adapt_option=None):
        super().__init__()
        self.ffn_adapt_option = ffn_adapt_option
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, adapt=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None
            # print("use PTM backbone without adapter.")

        residual = x
        x = self.drop_path(self.mlp(self.norm2(x)))

        if adapt_x is not None:
            if self.ffn_adapt_option == 'sequential':
                x = adapt(x)
            elif self.ffn_adapt_option == 'parallel':
                x = x + adapt_x
            else:
                pass

        x = residual + x

        return x


class VisionTransformer_adapter(nn.Module):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, pos_drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, **kwargs):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=pos_drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=nn.GELU,
                ffn_adapt_option="parallel"
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.global_pool = global_pool

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_no_old(self, x, cur_adapter):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            x = blk(x, cur_adapter[idx])

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            out = self.fc_norm(x)
        else:
            x = self.norm(x)
            out = x[:, 0]

        return out

    def forward_old(self, x, old_adapters, cur_adapter, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        features = []

        if use_init_ptm:
            x = copy.deepcopy(x_init)
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
                x = self.norm(x)
            features.append(x[:, 0, :])  # B N 768

        assert old_adapters is not None
        for i in range(len(old_adapters)):
            x = copy.deepcopy(x_init)
            for j in range(len(self.blocks)):
                adapt = old_adapters[i][j]
                x = self.blocks[j](x, adapt)
            x = self.norm(x)
            features.append(x[:, 0, :])  # old_task B N 768

        x = copy.deepcopy(x_init)
        for i in range(len(self.blocks)):
            adapt = cur_adapter[i]
            x = self.blocks[i](x, adapt)
        x = self.norm(x)
        features.append(x[:, 0, :])   # B N 768

        return features  # 1+cur_task B 1 768

    def forward(self, x, old_adapters=None, cur_adapter=None, use_old=False, use_init_ptm=False):
        if not use_old:
            output = self.forward_no_old(x, cur_adapter)
        else:
            features = self.forward_old(x, old_adapters, cur_adapter, use_init_ptm)
            output = torch.Tensor().to(features[0].device)
            for cls in features:
                output = torch.cat((output, cls), dim=1)

        return output  # B,768 or  B,768*(1+cur_task)


@register_model
def vit_adapter(**kwargs):
    model = VisionTransformer_adapter(**kwargs)
    return model

