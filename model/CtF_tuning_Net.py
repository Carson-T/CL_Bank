from model.backbone.clip.clip import load, tokenize
from model.backbone.Adapter import Adapter
import numpy as np
import torch
import torch.nn as nn
from model.Base_Net import Base_Net
from utils.functions import *


class CtF_tuning_Net(Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        assert config.backbone == "CLIP"
        self.use_desc = config.use_desc
        self.desc_num = config.desc_num
        self.topk = config.topk
        self.mlp_ratio = 4
        self.lora_r = config.lora_r
        self.prompt_length = 0

        self.vis_mlp = None
        self.text_mlp = None
        self.prompt_param = None

        self.img_adapter_list = nn.ModuleList([])
        # self.img_adapter_list = None
        # self.text_adapter_list = nn.ModuleList([])
        self.text_adapter_list = None
        self.img_final_adapter = None
        self.text_final_adapter = None

    def model_init(self):
        # self.prompt_length = 40
        self.backbone, _ = load(self.config.pretrained_path, jit=False, lora_r=self.lora_r, prompt_length=self.prompt_length)
        self.feature_dim = self.backbone.output_dim  # 512
        self.logger.info("clip model loaded!")
        # self.vis_mlp = nn.Sequential(*[
        #         nn.Linear(self.feature_dim, self.feature_dim*self.mlp_ratio),
        #         nn.GELU(),
        #         nn.Linear(self.feature_dim*self.mlp_ratio, self.feature_dim)
        # ])

        # self.text_mlp = nn.Sequential(*[
        #     nn.Linear(self.feature_dim, self.feature_dim*self.mlp_ratio),
        #     nn.GELU(),
        #     nn.Linear(self.feature_dim*self.mlp_ratio, self.feature_dim)
        # ])
        # self.prompt_param = tensor_prompt(self.prompt_length, self.backbone.transformer_width)

        for i in range(self.backbone.vision_layers):  #self.backbone.vision_layers
            img_adapter = Adapter(d_model=self.backbone.vision_width,
                              dropout=0.1,
                              bottleneck=64,
                              init_option="lora",
                              adapter_scalar="0.1",
                              adapter_layernorm_option=None)
            self.img_adapter_list.append(img_adapter)

        self.img_final_adapter = Adapter(d_model=self.backbone.output_dim,
                              dropout=0.1,
                              bottleneck=64,
                              init_option="lora",
                              adapter_scalar="0.1",
                              adapter_layernorm_option=None)

        # for j in range(self.backbone.transformer_layers):
        #     text_adapter = Adapter(d_model=self.backbone.transformer_width,
        #                       dropout=0.1,
        #                       bottleneck=64,
        #                       init_option="lora",
        #                       adapter_scalar="0.1",
        #                       adapter_layernorm_option=None)
        #     self.text_adapter_list.append(text_adapter)
        #
        # self.text_final_adapter = Adapter(d_model=self.backbone.output_dim,
        #                       dropout=0.1,
        #                       bottleneck=64,
        #                       init_option="lora",
        #                       adapter_scalar="0.1",
        #                       adapter_layernorm_option=None)


    def forward_train(self, image_features, text_tokens=None):
        logits_dim = text_tokens.shape[0]
        '''process text'''
        text_tokens = text_tokens.reshape((-1, 77))  # train: [logits_dim, 77] or [logits_dim*prompt_num,77]  test:[bs*logits_dim, 77] or [bs*logits_dim*prompt_num,77]
        text_features = self.backbone.encode_text(text_tokens, self.text_adapter_list, self.prompt_param)  # train: [logits_dim, 512] or [logits_dim*prompt_num, 512]  test: [bs*logits_dim, 512] or [bs*logits_dim*prompt_num, 512]
        if self.text_final_adapter is not None:
            text_features = self.text_final_adapter(text_features)
        elif self.text_mlp is not None:
            text_features = self.text_mlp(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.use_desc:
            text_features = text_features.reshape((logits_dim, self.desc_num, -1)).mean(dim=-2)  # [logits_dim, 512]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.transpose(-1, -2))
        logits = similarity*self.backbone.logit_scale.exp()
        # logits = similarity*self.scale.exp()

        return {"logits": logits, "features": image_features.clone().squeeze()}

    def forward_test(self, image_features, text_tokens):
        bs = image_features.shape[0]
        '''process image'''
        image_features = image_features.reshape((bs, 1, -1))  # [bs, 1, 512]
        '''process text'''
        text_tokens = text_tokens.reshape((-1, 77))  # train: [logits_dim, 77] or [logits_dim*prompt_num,77]  test:[bs*logits_dim, 77] or [bs*logits_dim*prompt_num,77]
        text_features = self.backbone.encode_text(text_tokens, self.text_adapter_list, self.prompt_param)  # train: [logits_dim, 512] or [logits_dim*prompt_num, 512]  test: [bs*logits_dim, 512] or [bs*logits_dim*prompt_num, 512]
        if self.text_final_adapter is not None:
            text_features = self.text_final_adapter(text_features)
        elif self.text_mlp is not None:
            text_features = self.text_mlp(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if not self.use_desc:
            text_features = text_features.reshape((bs, self.topk, -1))  # [bs, logits_dim, 512]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = text_features.reshape((bs, self.topk, self.desc_num, -1)).mean(dim=-2)  # [bs, logits_dim, 512]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.transpose(-1, -2))*self.backbone.logit_scale.exp()
        # similarity = (image_features @ text_features.transpose(-1, -2))*self.scale.exp()
        logits = torch.squeeze(similarity, dim=-2)  # 最后一个item的bs为1，squeeze后，shape为[1]

        return {"logits": logits, "features": image_features.clone().squeeze()}


    def forward(self, images, text_tokens=None, train=False):
        image_features = self.backbone.encode_image(images, self.img_adapter_list)  # [bs, 512]
        if self.img_final_adapter is not None:
            image_features = self.img_final_adapter(image_features)  # [bs, 512]
        elif self.vis_mlp is not None:
            image_features = self.vis_mlp(image_features)  # [bs, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if train:
            out = self.forward_train(image_features, text_tokens)
        else:
            out = self.forward_test(image_features, text_tokens)

        return out

    # def forward_with_vectors(self, x, text_tokens):
    #     x = x.type(self.backbone.dtype)
    #     image_features = self.img_final_adapter(x)
    #     text_features = self.backbone.encode_text(text_tokens, self.text_adapter_list)
    #     image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
    #     # cosine similarity as logits
    #     logit_scale = self.backbone.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features_normed @ text_features_normed.t()
    #     logits_per_text = logits_per_image.t()
    #     return {"logits": logits_per_image, "features": image_features}

    def use_lora(self):
        for name, param in self.backbone.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in self.vis_mlp.parameters():
            param.requires_grad = True
        # for param in self.text_mlp.parameters():
        #     param.requires_grad = True

    def freeze_text(self):
        for name, param in self.backbone.named_parameters():
            if "visual" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


