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


class CLIP_task_adapter_Net(CLIP_Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.prompt_length = config.prompt_length
        self.inc = config.increment_steps[0]
        self.img_adapter_list = nn.ModuleList([])

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

    def update_model(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes
        if self.prompt_length > 0:
            if self.text_prompts is not None:
                new_text_prompts = tensor_prompt(cur_classes, self.prompt_length, 512)
                new_text_prompts.requires_grad = False
                new_text_prompts[0: known_classes] = copy.deepcopy(self.text_prompts.data)
                del self.text_prompts
                new_text_prompts.requires_grad = True
                self.text_prompts = new_text_prompts
            else:
                self.text_prompts = tensor_prompt(cur_classes, self.prompt_length, 512)
            # for param in self.text_prompts.parameters():
            #     param.requires_grad = False
            # self.text_prompts.append(tensor_prompt(new_classes, self.prompt_length, 512))


        if task_id > 0:
            self.img_adapter_list.append(copy.deepcopy(self.cur_img_adapter))
            for param in self.img_adapter_list.parameters():
                param.requires_grad = False
            self.cur_img_adapter = nn.ModuleList([])
            for i in range(self.backbone.vision_layers):  # self.backbone.vision_layers
                img_adapter = Adapter(d_model=self.backbone.vision_width,
                                      dropout=0.1,
                                      bottleneck=64,
                                      init_option="lora",
                                      adapter_scalar="0.1",
                                      adapter_layernorm_option=None)
                self.cur_img_adapter.append(img_adapter)
        else:
            self.cur_img_adapter = nn.ModuleList([])
            for i in range(self.backbone.vision_layers):  #self.backbone.vision_layers
                img_adapter = Adapter(d_model=self.backbone.vision_width,
                                  dropout=0.1,
                                  bottleneck=64,
                                  init_option="lora",
                                  adapter_scalar="0.1",
                                  adapter_layernorm_option=None)
                self.cur_img_adapter.append(img_adapter)

        #     for param in self.conv_list.parameters():
        #         param.requires_grad = False
        #     for param in self.conv_scaler_list.parameters():
        #         param.requires_grad = False
        #     self.conv_list.append(nn.Conv1d(196, self.desc_num, kernel_size=1))
        #     self.conv_scaler_list.append(nn.Parameter(torch.ones([])))

    def forward_train(self, image, text_tokens):
        class_num = text_tokens.shape[0]
        image_features = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter)
        text_features = self.backbone.encode_text(text_tokens, prompt=self.text_prompts[-class_num:])
        image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
        text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)

        logits_global = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()

        return {"logits": logits_global, "text_features": text_features_normed, "features": image_features}

    def forward_test(self, image, text_tokens=None, task_id=None, img_proto=None):
        all_logits_global = torch.tensor([]).cuda()
        for i in range(task_id):
            image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list[i])
            text_features = self.backbone.encode_text(text_tokens[i*self.inc:(i+1)*self.inc], prompt=self.text_prompts[i * self.inc:(i + 1) * self.inc])
            image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
            text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
            if img_proto is not None:
                image_prototypes = img_proto[i*self.inc:(i+1)*self.inc]
                logits = image_features_normed @ text_features_normed.t() + image_features_normed @ image_prototypes.t()
            else:
                logits = image_features_normed @ text_features_normed.t()
            logits_global = logits
            all_logits_global = torch.cat([all_logits_global, logits_global], dim=1)

        image_features = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter)
        text_features = self.backbone.encode_text(text_tokens[-self.inc:], prompt=self.text_prompts[-self.inc:])
        image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
        text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)

        if img_proto is not None:
            image_prototypes = img_proto[-self.inc:]
            logits = image_features_normed @ text_features_normed.t() + image_features_normed @ image_prototypes.t()
        else:
            logits = image_features_normed @ text_features_normed.t()

        logits_global = logits
        all_logits_global = torch.cat([all_logits_global, logits_global], dim=1)

        text_features = None

        return {"logits": all_logits_global, "text_features": text_features, "features": image_features}

    def forward(self, image, text_tokens=None, train=False, task_id=None, img_proto=None):
        if text_tokens is None:
            image_features = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter)
            image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
            return {"features": image_features_normed}
        else:
            if train:
                out = self.forward_train(image, text_tokens)
            else:
                out = self.forward_test(image, text_tokens, task_id, img_proto=img_proto)

            return out
