from model.backbone.clip.clip import load, tokenize
from model.backbone.Adapter import Adapter
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Base_Net import Base_Net
import copy
from utils.functions import *


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Proof_Net(Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        assert config.backbone == "CLIP"
        self.img_projector_list = nn.ModuleList([])
        self.text_projector_list = nn.ModuleList([])
        self.fusion_module = None
        self.context_prompt = nn.ParameterList([])
        self.prompt_length = config.prompt_length
        self.attention_heads = config.attention_heads
        self.logit_scale1 = None
        self.logit_scale2 = None
        self.logit_scale3 = None

    def model_init(self):
        self.backbone, _ = load(self.config.pretrained_path, jit=False)
        self.output_dim = self.backbone.output_dim
        self.logger.info("model loaded!")
        self.fusion_module = nn.Sequential(
            # nn.LayerNorm(self.output_dim),
            Attention(dim=self.output_dim, num_heads=self.attention_heads),
        )
        self.normlayer = nn.LayerNorm(self.output_dim)
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def update_model(self, task_id):
        for param in self.img_projector_list.parameters():
            param.requires_grad = False
        for param in self.text_projector_list.parameters():
            param.requires_grad = False
        img_new_projector = nn.Linear(self.output_dim, self.output_dim)
        text_new_projector = nn.Linear(self.output_dim, self.output_dim)
        self.img_projector_list.append(img_new_projector)
        self.text_projector_list.append(text_new_projector)

        for param in self.context_prompt.parameters():
            param.requires_grad = False
        self.context_prompt.append(tensor_prompt(self.prompt_length, self.output_dim))

    def text_tokenize(self, cur_class_names, prompt_template):
        text_tokens = tokenize([prompt_template.format(c) for c in cur_class_names])
        return text_tokens

    def forward_proj(self, features, proj_list):
        for i in range(len(proj_list)):
            if i == 0:
                out = proj_list[i](features)
            else:
                out += proj_list[i](features)
        return out

    def forward_fusion(self, query_token, img_tokens, text_tokens):
        B = query_token.shape[0]
        query_token = query_token.reshape(B, 1, -1)    # (B,1,512)
        img_tokens = img_tokens.expand(B, -1, -1)  # (B,cur_class,512)
        text_tokens = text_tokens.expand(B, -1, -1)  # (B,cur_class,512)
        prompt_tokens = torch.cat([i for i in self.context_prompt], dim=0).expand(B, -1, -1)  # (B,task*prompt_length,512)
        fusion_inputs = torch.cat([query_token, img_tokens, text_tokens, prompt_tokens], dim=1)
        out = fusion_inputs + self.fusion_module(fusion_inputs)
        out = self.normlayer(out)

        return out

    def forward(self, image, text_tokens=None, img_proto=None, text_proto=None):
        B = image.shape[0]
        if text_tokens is None:
            image_features = self.backbone.encode_image(image)
            return {"features": image_features}
        else:
            image_features = self.backbone.encode_image(image)
            query_token = self.forward_proj(image_features, self.img_projector_list)
            img_tokens = self.forward_proj(img_proto, self.img_projector_list)
            text_tokens = self.forward_proj(text_proto, self.text_projector_list)
            fused_out = self.forward_fusion(query_token, img_tokens, text_tokens)
            fused_img_feature = fused_out[:, 0, :]
            fused_img_proto = fused_out[:, 1:1+len(img_proto), :]
            fused_text_proto = fused_out[:, 1+len(img_proto):1+len(img_proto)+len(text_proto), :]

            image_features_normed = fused_img_feature / fused_img_feature.norm(dim=-1, keepdim=True)    #  B,512
            fused_img_proto = fused_img_proto / fused_img_proto.norm(dim=-1, keepdim=True)    # B,cur_class,512
            fused_text_proto = fused_text_proto / fused_text_proto.norm(dim=-1, keepdim=True)   # B,cur_class,512

            query_token_normed = query_token / query_token.norm(dim=-1, keepdim=True)    #  B,512
            text_tokens_normed = text_tokens / text_tokens.norm(dim=-1, keepdim=True)    #  B,512
            pm_logits = self.logit_scale1.exp()*query_token_normed @ text_tokens_normed.t()
            vm_logits = self.logit_scale2.exp()*(image_features_normed.reshape(B, 1, -1) @ fused_img_proto.transpose(-1, -2)).squeeze()
            tm_logits = self.logit_scale3.exp()*(image_features_normed.reshape(B, 1, -1) @ fused_text_proto.transpose(-1, -2)).squeeze()

            return {"pm_logits": pm_logits, "vm_logits": vm_logits, "tm_logits": tm_logits, "features": image_features}

    # def forward(self, image, text_tokens):
    #     logits_per_image, logits_per_text, image_features, text_features = self.backbone(image, text_tokens, self.img_adapter_list, self.text_adapter_list)
    #     logits_per_image = logits_per_image.softmax(dim=-1)
    #     return {"logits": logits_per_image, "features": image_features}



