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


class CLIP_local_fe_Net(CLIP_Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.prompt_length = config.prompt_length
        self.topk = config.topk
        self.inc = config.increment_steps[0]
        self.img_adapter_list = nn.ModuleList([])
        self.local_proj_list = nn.ModuleList([])
        self.cur_local_proj = None
        self.local_prompts = None

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

        for i in range(self.backbone.vision_layers):  # self.backbone.vision_layers
            if i < 0:
                img_adapter = None
            else:
                img_adapter = Adapter(d_model=self.backbone.vision_width,
                                      dropout=0.1,
                                      bottleneck=64,
                                      is_ss=False,
                                      init_option="lora",
                                      adapter_scalar="0.1",
                                      adapter_layernorm_option=None)
            self.img_adapter_list.append(img_adapter)

        self.local_projector = nn.Linear(512, 512)
        # self.local_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if self.prompt_length > 0:
            self.local_prompts = tensor_prompt(4, self.prompt_length, 512, ortho=True)

    def update_model(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes

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
        #
        # if task_id > 0:
        #     self.local_proj_list.append(copy.deepcopy(self.cur_local_proj))
        #     for param in self.local_proj_list.parameters():
        #         param.requires_grad = False
        #     self.cur_local_proj = nn.Linear(512, 512)
        # else:
        #     self.cur_local_proj = nn.Linear(512, 512)

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
        image_features = torch.einsum("bk,bkd->bd",qk_sim, all_image_features)

        return image_features, qk_loss

    def forward_train(self, image, text_tokens=None, desc_tokens=None, task_id=None):
        class_num = text_tokens.shape[0]
        B = image.shape[0]
        image_features, patch_tokens = self.backbone.encode_image(image, adapter_list=self.img_adapter_list,
                                                                  ret_all=True)
        # image_features, patch_tokens = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter,
        #                                                           ret_all=True)

        image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.backbone.encode_text(text_tokens, prompt=self.text_prompts if self.text_prompts is not None else None)
        text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)

        logits_global = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()
        # logits_global = logits_global.softmax(dim=-1)
        logits_local = None
        if desc_tokens is not None:
            patch_tokens = self.local_projector(patch_tokens)
            patch_tokens_normed = patch_tokens / patch_tokens.norm(dim=1, keepdim=True)
            logits_local = torch.FloatTensor([]).cuda()
            for l in range(self.local_prompts.shape[0]):
                desc_features = self.backbone.encode_text(desc_tokens, prompt=self.local_prompts[l].expand(class_num, -1, -1))
                # desc_features = self.backbone.encode_text(desc_tokens, prompt=self.desc_prompts[task_id*self.inc:(task_id+1)*self.inc])
                desc_features_normed = desc_features / desc_features.norm(dim=1, keepdim=True)

                local_sims = patch_tokens_normed @ desc_features_normed.expand(B, -1, -1).transpose(-1, -2) # B,196,cur_class
                values, indices = torch.topk(local_sims, k=self.topk, dim=1)
                logits_local = torch.cat([logits_local, (self.backbone.logit_scale.exp() * torch.mean(values, dim=1)).unsqueeze(1)], dim=1)
            logits_local = logits_local.mean(dim=1)
            # logits_local = logits_local.softmax(dim=-1)

        if self.config.alpha > 0:
            text_loss = torch.pdist(text_features_normed, p=2).pow(2.0).mul(-2.0).exp().mean()
        else:
            text_loss = None

        return {"logits": logits_global, "logits_local": logits_local, "text_loss": text_loss, "features": image_features}

    def forward_test(self, image, text_tokens=None, desc_tokens=None, task_id=None):
        B = image.shape[0]
        class_num = text_tokens.shape[0]
        image_features, patch_tokens = self.backbone.encode_image(image, adapter_list=self.img_adapter_list,
                                                                  ret_all=True)

        image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.backbone.encode_text(text_tokens, prompt=self.text_prompts if self.text_prompts is not None else None)
        text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
        logits_global = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()
        # logits_global = logits_global.softmax(dim=-1)
        logits_local = None
        if desc_tokens is not None:
            patch_tokens = self.local_projector(patch_tokens)
            patch_tokens_normed = patch_tokens / patch_tokens.norm(dim=1, keepdim=True)
            logits_local = torch.FloatTensor([]).cuda()
            for l in range(self.local_prompts.shape[0]):
                desc_features = self.backbone.encode_text(desc_tokens, prompt=self.local_prompts[l].expand(class_num, -1, -1))
                # desc_features = self.backbone.encode_text(desc_tokens, prompt=self.desc_prompts[task_id*self.inc:(task_id+1)*self.inc])
                desc_features_normed = desc_features / desc_features.norm(dim=1, keepdim=True)

                local_sims = patch_tokens_normed @ desc_features_normed.expand(B, -1, -1).transpose(-1, -2)  # B,196,cur_class
                values, indices = torch.topk(local_sims, k=self.topk, dim=1)
                logits_local = torch.cat([logits_local, (self.backbone.logit_scale.exp() * torch.mean(values, dim=1)).unsqueeze(1)],
                                         dim=1)
            logits_local = logits_local.mean(dim=1)
            # logits_local = logits_local.softmax(dim=-1)

        return {"logits": logits_global, "logits_local": logits_local, "features": image_features_normed}

    # def forward_test(self, image, text_tokens=None, desc_tokens=None, task_id=None):
    #     B = image.shape[0]
    #     all_logits_global = torch.tensor([]).cuda()
    #     all_logits_local = torch.tensor([]).cuda()
    #     for i in range(task_id):
    #         image_features, patch_tokens = self.backbone.encode_image(image, adapter_list=self.img_adapter_list[i], ret_all=True)
    #         text_features = self.backbone.encode_text(text_tokens[i*self.inc:(i+1)*self.inc], prompt=self.text_prompts if self.text_prompts is not None else None)
    #         image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
    #         text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
    #         logits_global = self.backbone.logit_scale.exp()*image_features_normed @ text_features_normed.t()
    #         all_logits_global = torch.cat([all_logits_global, logits_global], dim=1)
    #
    #         if desc_tokens is not None:
    #             desc_features = self.backbone.encode_text(desc_tokens[i*self.inc:(i+1)*self.inc], prompt=self.desc_prompts[i*self.inc:(i+1)*self.inc])
    #             # desc_features = desc_features.reshape((class_num, -1))   #  5, cls, 512
    #             # patch_tokens = self.local_projector(patch_tokens)
    #             patch_tokens_normed = patch_tokens / patch_tokens.norm(dim=1, keepdim=True)
    #             desc_features_normed = desc_features / desc_features.norm(dim=1, keepdim=True)
    #
    #             local_sims = patch_tokens_normed @ desc_features_normed.expand(B, -1, -1).transpose(-1, -2)
    #             values, indices = torch.topk(local_sims, k=self.topk, dim=1)
    #             logits_local = self.local_logit_scale.exp() * torch.mean(values, dim=1)
    #             all_logits_local = torch.cat([all_logits_local, logits_local], dim=1)
    #
    #     image_features, patch_tokens = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter, ret_all=True)
    #     text_features = self.backbone.encode_text(text_tokens[task_id*self.inc:(task_id+1)*self.inc], prompt=self.text_prompts if self.text_prompts is not None else None)
    #     image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
    #     logits_global = self.backbone.logit_scale.exp()*image_features_normed @ text_features_normed.t()
    #     all_logits_global = torch.cat([all_logits_global, logits_global], dim=1)
    #
    #     if desc_tokens is not None:
    #         desc_features = self.backbone.encode_text(desc_tokens[task_id*self.inc:(task_id+1)*self.inc], prompt=self.desc_prompts[task_id*self.inc:(task_id+1)*self.inc])
    #         # desc_features = desc_features.reshape((class_num, -1))   #  5, cls, 512
    #         # patch_tokens = self.local_projector(patch_tokens)
    #         patch_tokens_normed = patch_tokens / patch_tokens.norm(dim=1, keepdim=True)
    #         desc_features_normed = desc_features / desc_features.norm(dim=1, keepdim=True)
    #
    #         local_sims = patch_tokens_normed @ desc_features_normed.expand(B, -1, -1).transpose(-1, -2)
    #         values, indices = torch.topk(local_sims, k=self.topk, dim=1)
    #         logits_local = self.local_logit_scale.exp() * torch.mean(values, dim=1)
    #         all_logits_local = torch.cat([all_logits_local, logits_local], dim=1)
    #     logits = (all_logits_global+all_logits_local)
    #
    #     return {"logits": logits, "logits_local": all_logits_local, "features": image_features_normed}

    def forward(self, image, text_tokens=None, desc_tokens=None, train=False, task_id=None):
        if text_tokens is None:
            image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list)
            # image_features = self.projector(image_features)
            image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
            return {"features": image_features_normed}
        else:
            if train:
                out = self.forward_train(image, text_tokens, desc_tokens, task_id)
            else:
                out = self.forward_test(image, text_tokens, desc_tokens, task_id)

            return out
