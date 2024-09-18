import open_clip
from open_clip.tokenizer import HFTokenizer
import numpy as np
import torch
import torch.nn as nn
from model.Base_Net import Base_Net
import copy
from model.backbone.clip.clip import load, tokenize
from model.backbone.MedCLIP.model import MedCLIPModel, MedCLIPVisionModelViT
from model.backbone.Adapter import Adapter
from utils.functions import *


class CLIP_Base_Net(Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.desc_prompts = None
        self.text_prompts = None

        self.img_adapter_list = None
        self.cur_img_adapter = None
        self.text_adapter_list = None
        self.img_final_adapter = None

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
        self.img_final_adapter = nn.Linear(512, 512)
        # self.img_adapter_list = nn.ModuleList([])
        # for i in range(self.backbone.vision_layers):  # self.backbone.vision_layers
        #     img_adapter = Adapter(d_model=self.backbone.vision_width,
        #                           dropout=0.1,
        #                           bottleneck=64,
        #                           init_option="lora",
        #                           adapter_scalar="0.1",
        #                           adapter_layernorm_option=None)
        #     self.img_adapter_list.append(img_adapter)

    def update_model(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes

    def text_tokenize(self, class_names, prompt_template, descs=None):
        if self.config.backbone == "CLIP":
            text = [prompt_template.format(c) for c in class_names]
            if descs is not None:
                assert len(class_names) == len(descs)
                for i in range(len(class_names)):
                    text[i] += descs[i]
            text_tokens = tokenize(text)
        elif self.config.backbone == "OpenCLIP":
            # tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
            tokenizer = HFTokenizer(self.config.pretrained_path)
            text_tokens = tokenizer([prompt_template.format(c) for c in class_names], context_length=self.backbone.context_length)
        elif self.config.backbone == "MedCLIP":
            text_tokens = self.backbone.tokenize([prompt_template.format(c) for c in class_names])
            del text_tokens["token_type_ids"]

        return text_tokens

    def forward(self, image, text_tokens=None, img_proto=None):
        if text_tokens is None:
            image_features = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter)
            image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
            return {"features": image_features_normed}
        else:
            image_features = self.backbone.encode_image(image)
            image_features = self.img_final_adapter(image_features)
            text_features = self.backbone.encode_text(text_tokens)
            image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
            text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)

            logits = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()

            return {"logits": logits, "features": image_features_normed}


    def save_old_param(self):
        self.old_adapter_state_dict = copy.deepcopy(self.img_final_adapter.state_dict())

    def param_retention(self):
        self.img_final_adapter.weight.requires_grad = False
        self.img_final_adapter.bias.requires_grad = False

        delta_w = abs(self.img_final_adapter.weight - self.old_adapter_state_dict["weight"]).view(-1)
        delta_b = abs(self.img_final_adapter.bias - self.old_adapter_state_dict["bias"])
        w_indices = torch.argsort(delta_w, descending=False)
        w_indices = w_indices[:int(len(w_indices)*self.config.ret_ratio)]
        for index in w_indices:
            i = index.item() // self.output_dim
            j = index.item() % self.output_dim
            self.img_final_adapter.weight[i, j] = self.old_adapter_state_dict["weight"][i, j]
        b_indices = torch.argsort(delta_b, descending=False)
        b_indices = b_indices[:int(len(b_indices)*self.config.ret_ratio)]
        self.img_final_adapter.bias[b_indices] = self.old_adapter_state_dict["bias"][b_indices]
        self.logger.info("adapter param retention finished!")
        self.img_final_adapter.weight.requires_grad = True
        self.img_final_adapter.bias.requires_grad = True

