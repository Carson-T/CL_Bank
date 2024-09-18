import open_clip
from open_clip.tokenizer import HFTokenizer
import numpy as np
import torch
import torch.nn as nn
from model.CLIP_Base_Net import CLIP_Base_Net
import copy
from model.backbone.clip.clip import load, tokenize
from model.backbone.MedCLIP.model import MedCLIPModel, MedCLIPVisionModelViT
from model.backbone.Adapter import Adapter
from utils.functions import *


class CLIP_MoE_Net(CLIP_Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.prompt_length = config.prompt_length
        self.topk = config.topk
        self.inc = config.increment_steps[0]
        self.img_adapter_list = nn.ModuleList([])
        self.img_router_list = nn.ParameterList([])
        self.img_noise_w_list = nn.ParameterList([])
        self.img_final_adapter = nn.ModuleList([])
        # self.attn_adapter_list = nn.ModuleList([])
        # self.attn_router_list = nn.ParameterList([])
        # self.attn_noise_w_list = nn.ParameterList([])

        # self.img_adapter_list = None
        # self.router_list = None

    def model_init(self):
        if self.config.backbone == "CLIP":
            self.backbone, _ = load(self.config.pretrained_path, jit=False)
        elif self.config.backbone == "OpenCLIP":
            # self.backbone, _ = open_clip.create_model_from_pretrained("ViT-B-16", pretrained=self.config.pretrained_path+"/open_clip_pytorch_model.bin")
            self.backbone = open_clip.create_model("ViT-B-16-SigLIP")
            self.backbone.load_state_dict(
                torch.load(self.config.pretrained_path+"/open_clip_pytorch_model.bin")
                )
            self.backbone.float()
        elif self.config.backbone == "MedCLIP":
            self.backbone = MedCLIPModel(MedCLIPVisionModelViT)
            self.backbone.from_pretrained(self.config.pretrained_path)
        self.output_dim = self.backbone.output_dim
        # self.output_dim = 512
        self.logger.info("model loaded!")

        # for j in range(2):
        #     img_adapter = nn.ModuleList([])
        #     for i in range(self.backbone.vision_layers):  # self.backbone.vision_layers
        #             img_adapter.append(Adapter(d_model=self.backbone.vision_width,
        #                                   dropout=0.1,
        #                                   bottleneck=64,
        #                                   is_ss=False,
        #                                   init_option="lora",
        #                                   adapter_scalar="0.1",
        #                                   adapter_layernorm_option=None))
        #     self.img_adapter_list.append(img_adapter)

        for i in range(self.backbone.vision_layers):  # self.backbone.vision_layers
            img_adapter = nn.ModuleList([])
            for j in range(self.config.adapter_num):
                img_adapter.append(Adapter(d_model=self.backbone.vision_width,
                                           dropout=0.1,
                                           bottleneck=64,
                                           is_ss=False,
                                           init_option="lora",
                                           adapter_scalar="0.1",
                                           adapter_layernorm_option=None))
            self.img_adapter_list.append(img_adapter)
            self.img_router_list.append(nn.Parameter(torch.zeros(768, self.config.adapter_num)))
            self.img_noise_w_list.append(nn.Parameter(torch.zeros(768, self.config.adapter_num)))

        for i in range(2):
            self.img_final_adapter.append(Adapter(d_model=512,
                                       dropout=0.1,
                                       bottleneck=512*4,
                                       is_ss=False,
                                       init_option="lora",
                                       adapter_scalar="0.1",
                                       adapter_layernorm_option=None))
        self.img_router = nn.Parameter(torch.zeros(512, 2))
        self.img_noise_w = nn.Parameter(torch.zeros(512, 2))

        # self.local_projector = nn.Linear(512, 512)
        # self.local_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if self.prompt_length > 0:
            self.text_prompts = tensor_prompt(sum(self.config.increment_steps), self.prompt_length, 512, ortho=True)
        # self.desc_prompts = tensor_prompt(sum(self.config.increment_steps), self.prompt_length, 512, ortho=True)


    def update_model(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes

        # if self.prompt_length > 0:
            # if self.text_prompts is not None:
            #     new_text_prompts = tensor_prompt(cur_classes, self.prompt_length, 512)
            #     new_text_prompts.requires_grad = False
            #     new_text_prompts[0: known_classes] = copy.deepcopy(self.text_prompts.data)
            #     del self.text_prompts
            #     new_text_prompts.requires_grad = True
            #     self.text_prompts = new_text_prompts
            # else:
            #     self.text_prompts = tensor_prompt(cur_classes, self.prompt_length, 512)

        # if task_id > 0:
        #     self.img_adapter_list.append(copy.deepcopy(self.cur_img_adapter))
        #     for param in self.img_adapter_list.parameters():
        #         param.requires_grad = False
        #     self.cur_img_adapter = nn.ModuleList([])
        #     for i in range(self.backbone.vision_layers):  # self.backbone.vision_layers
        #         img_adapter = Adapter(d_model=self.backbone.vision_width,
        #                               dropout=0.1,
        #                               bottleneck=64,
        #                               init_option="lora",
        #                               adapter_scalar="0.1",
        #                               adapter_layernorm_option=None)
        #         self.cur_img_adapter.append(img_adapter)
        # else:
        #     self.cur_img_adapter = nn.ModuleList([])
        #     for i in range(self.backbone.vision_layers):  #self.backbone.vision_layers
        #         img_adapter = Adapter(d_model=self.backbone.vision_width,
        #                           dropout=0.1,
        #                           bottleneck=64,
        #                           init_option="lora",
        #                           adapter_scalar="0.1",
        #                           adapter_layernorm_option=None)
        #         self.cur_img_adapter.append(img_adapter)

    def get_task_fe(self, image_features, train):
        q = F.normalize(image_features, dim=1)
        # if train:
        #     k = F.normalize(self.task_key_list[-1], dim=1)
        #     qk_loss = (1.0 - (q.detach() @ k.t())).sum()
        #     image_features = self.projector_list[-1](image_features)
        # else:
        #     k = F.normalize(self.task_key_list, dim=1)
        #     qk_sim = q @ k.t()
        #     top1_sim, top1 = torch.topk(qk_sim, k=1, dim=-1)
        #     qk_loss = (1.0 - top1_sim).sum()
        #     all_image_features = torch.stack([proj(image_features) for proj in self.projector_list], dim=1)
        #     image_features = torch.gather(all_image_features, dim=1,
        #                                   index=top1.unsqueeze(-1).expand(-1, -1, all_image_features.shape[-1])).squeeze()

        k = F.normalize(self.task_key_list, dim=1)
        qk_sim = q @ k.t()
        # top1_sim, top1 = torch.topk(qk_sim, k=1, dim=-1)
        qk_loss = (1.0 - qk_sim[:, -1]).sum()
        all_image_features = torch.stack([proj(image_features) for proj in self.projector_list], dim=1)
        image_features = torch.einsum("bk,bkd->bd", qk_sim, all_image_features)

        return image_features, qk_loss
    def moe_adapter(self, x, train=True):
        clean_logits = x @ self.img_router
        if train:
            raw_noise_stddev = x @ self.img_noise_w
            noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits
        mask = F.softmax(logits, dim=-1)
        all_out = torch.stack([adapter(x, add_residual=False) for adapter in self.img_final_adapter], dim=1)
        adapt_out = (all_out * mask.reshape(x.shape[0], -1, 1)).sum(dim=1)
        out = adapt_out+x

        return out
    # def forward_train(self, image, text_tokens=None, desc_tokens=None, task_id=None, targets=None):
    #     B = image.shape[0]
    #     class_num = text_tokens.shape[0]
    #     all_logits_global = torch.tensor([]).cuda()
    #     for i in range(len(self.img_adapter_list)):
    #         image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list[i])
    #         text_features = self.backbone.encode_text(text_tokens)
    #         image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
    #         text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
    #         logits = image_features_normed @ text_features_normed.t()
    #         logits_global = self.backbone.logit_scale.exp()*logits
    #         all_logits_global = torch.cat([all_logits_global, logits_global.unsqueeze(1)], dim=1)
    #     idx = torch.max(torch.gather(all_logits_global, dim=-1, index=targets.long().reshape(B, 1).unsqueeze(-1).expand(B, len(self.img_adapter_list), -1)), dim=1)[1]
    #     out_logits = torch.gather(all_logits_global, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, class_num)).squeeze(1)
    #     out_logits = out_logits.softmax(dim=-1)
    #     return {"logits": out_logits, "logits_local": None, "text_loss": None, "features": image_features_normed}
    #
    # def forward_test(self, image, text_tokens=None, desc_tokens=None, task_id=None):
    #     B = image.shape[1]
    #     all_logits_global = torch.tensor([]).cuda()
    #     all_features_global = torch.tensor([]).cuda()
    #     for i in range(len(self.img_adapter_list)):
    #         image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list[i])
    #         text_features = self.backbone.encode_text(text_tokens)
    #         image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
    #         text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
    #         logits = image_features_normed @ text_features_normed.t()
    #         logits_global = self.backbone.logit_scale.exp()*logits
    #         all_logits_global = torch.cat([all_logits_global, logits_global.unsqueeze(1)], dim=1)
    #         all_features_global = torch.cat([all_features_global, image_features_normed.unsqueeze(1)], dim=1)
    #
    #     out_logits = torch.max(all_logits_global, dim=1)[0]
    #     features = torch.mean(all_features_global, dim=1)
    #
    #     return {"logits": out_logits, "logits_local": None, "text_loss": None, "features": features}


    def forward_train(self, image, text_tokens=None, desc_tokens=None, task_id=None, targets=None):
        class_num = text_tokens.shape[0]
        B = image.shape[0]
        image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list, router_list=[self.img_router_list, self.img_noise_w_list])
        # image_features = self.backbone.encode_image(image)
        image_features = self.moe_adapter(image_features, train=True)

        image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.backbone.encode_text(text_tokens,
                                                  prompt=self.text_prompts[:(task_id+1)*self.inc] if self.text_prompts is not None else None)
        text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)

        logits_global = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()
        logits_global = logits_global.softmax(dim=-1)
        if self.config.alpha > 0:
            text_loss = torch.pdist(text_features_normed, p=2).pow(2.0).mul(-2.0).exp().mean()
        else:
            text_loss = None

        return {"logits": logits_global, "logits_local": None, "text_loss": text_loss, "features": image_features_normed}

    def forward_test(self, image, text_tokens=None, desc_tokens=None, task_id=None):
        B = image.shape[0]
        class_num = text_tokens.shape[0]
        image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list, router_list=[self.img_router_list, [None for _ in range(len(self.img_router_list))]])
        # image_features = self.backbone.encode_image(image)
        image_features = self.moe_adapter(image_features, train=False)

        image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.backbone.encode_text(text_tokens,
                                                  prompt=self.text_prompts[:(task_id+1)*self.inc] if self.text_prompts is not None else None)
        text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
        logits_global = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()
        logits_global = logits_global.softmax(dim=-1)
        return {"logits": logits_global, "logits_local": None, "features": image_features_normed}


    def forward(self, image, text_tokens=None, desc_tokens=None, train=False, task_id=None, targets=None):
        # if text_tokens is None:
        #     all_features_global = torch.tensor([]).cuda()
        #     for i in range(len(self.img_adapter_list)):
        #         image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list[i])
        #         image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
        #         all_features_global = torch.cat([all_features_global, image_features_normed.unsqueeze(1)], dim=1)
        #
        #     features = torch.mean(all_features_global, dim=1)
        #
        #     return {"features": features}
        if text_tokens is None:
            image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list, router_list=[self.img_router_list, [None for _ in range(len(self.img_router_list))]])
            # image_features = self.backbone.encode_image(image)
            # image_features = self.moe_adapter(image_features, train=False)
            image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)

            return {"features": image_features_normed}
        else:
            if train:
                out = self.forward_train(image, text_tokens, desc_tokens=desc_tokens, task_id=task_id, targets=targets)
            else:
                out = self.forward_test(image, text_tokens, desc_tokens=desc_tokens, task_id=task_id)

            return out
    def forward_with_vectors(self, x, text_tokens=None):
        x = x.type(self.backbone.dtype)
        image_features = self.moe_adapter(x, train=True)
        image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.backbone.encode_text(text_tokens)
        text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()
        logits = logits.softmax(dim=-1)
        return {"logits": logits}
        
    def freeze_adapter(self):
        for name, param in self.named_parameters():
            if "img_adapter_list" in name or "img_router_list" in name or "img_noise_w_list" in name:
                param.requires_grad = False

    def use_lora(self):
        for name, param in self.backbone.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
