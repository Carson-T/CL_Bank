import torch
import torch.nn as nn
from functools import partial
import timm
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_, DropPath
from safetensors.torch import load_file
from timm.models import register_model


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, prompt=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # B num_head N dim
        q, k = self.q_norm(q), self.k_norm(k)
        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, prompt=None):
        x = x + self.drop_path(self.attn(self.norm1(x), prompt=prompt))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer_prompt(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, pos_drop_rate=0, attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, pt_type=None, **kwargs):

        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pt_type = pt_type
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=pos_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # def forward(self, x, prompt=None, q=None, task_id=None):
    #     B = x.shape[0]
    #     x = self.patch_embed(x)
    #
    #     cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_tokens, x), dim=1)
    #
    #     x = x + self.pos_embed[:, :x.size(1), :]
    #     x = self.pos_drop(x)
    #
    #     prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
    #     for i, blk in enumerate(self.blocks):
    #         if prompt is not None:
    #                 p_list, loss, x = prompt.forward(q, i, x, train=True, task_id=task_id)
    #                 prompt_loss += loss
    #         else:
    #             p_list = None
    #
    #         x = blk(x, prompt=p_list)
    #         # if i == 11: x = x.detach()
    #
    #     x = self.norm(x)
    #
    #     return x, prompt_loss

    def prefix_t_forward(self, x, all_prompts=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            if str(i) in all_prompts:
                p_list = all_prompts[str(i)]
            else:
                p_list = None
            x = blk(x, prompt=p_list)
        x = self.norm(x)

        return x

    def prompt_t_forward(self, x, prompts=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        if prompts is not None:
            x = torch.cat([x[:, 0, :].unsqueeze(dim=1), prompts, x[:, 1:, :]], dim=1)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, all_prompts=None):
        if all_prompts is not None:
            if self.pt_type == "prompt_t":
                assert len(all_prompts) == 1, "prompt tuning only at the first layer"
                out = self.prompt_t_forward(x, prompts=all_prompts["0"])
            elif self.pt_type == "prefix_t":
                out = self.prefix_t_forward(x, all_prompts)
        else:
            out = self.prompt_t_forward(x, prompts=None)

        return out
    # @torch.jit.ignore()
    # def load_pretrained(self, checkpoint_path, prefix=''):
    #     _load_weights(self, checkpoint_path, prefix)


@register_model
def vit_prompt(pt_type, **kwargs):
    model = VisionTransformer_prompt(pt_type=pt_type, **kwargs)
    return model


# @register_model
# def vit_prefix_t(config, **kwargs):
#     model = VisionTransformer(img_size=config.img_size, pt_type=config.pt_type, drop_rete=config.drop_rate,
#                               drop_path_rate=config.drop_path_rate, **kwargs)
#     return model
