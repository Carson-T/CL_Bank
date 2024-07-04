from model.backbone.clip.clip import load, tokenize
from model.backbone.Adapter import Adapter
import torch
import torch.nn as nn
from model.Base_Net import Base_Net
import copy


class CLIP_Adapter_Net(Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        assert config.backbone == "CLIP"
        # self.img_adapter_list = nn.ModuleList([])
        self.img_adapter_list = None
        # self.text_adapter_list = nn.ModuleList([])
        self.text_adapter_list = None
        self.img_final_adapter = None

    def model_init(self):
        self.backbone, _ = load(self.config.pretrained_path, jit=False)
        self.output_dim = self.backbone.output_dim
        self.logger.info("model loaded!")
        # for i in range(self.backbone.vision_layers):  #self.backbone.vision_layers
        #     img_adapter = Adapter(d_model=self.backbone.vision_width,
        #                       dropout=0.1,
        #                       bottleneck=64,
        #                       init_option="lora",
        #                       adapter_scalar="0.1",
        #                       adapter_layernorm_option=None)
        #     self.img_adapter_list.append(img_adapter)
        # self.img_final_adapter = Adapter(d_model=self.output_dim,
        #                       dropout=0.1,
        #                       bottleneck=64,
        #                       init_option="lora",
        #                       adapter_scalar="0.1",
        #                       adapter_layernorm_option=None)
        self.img_final_adapter = nn.Linear(self.output_dim, self.output_dim)
        # for j in range(self.backbone.transformer_layers):
        #     text_adapter = Adapter(d_model=self.backbone.transformer_width,
        #                       dropout=0.1,
        #                       bottleneck=64,
        #                       init_option="lora",
        #                       adapter_scalar="0.1",
        #                       adapter_layernorm_option=None)
        #     self.text_adapter_list.append(text_adapter.half())

    def text_tokenize(self, cur_class_names, prompt_template):
        text_tokens = tokenize([prompt_template.format(c) for c in cur_class_names])
        return text_tokens

    def forward(self, image, text_tokens=None):
        if text_tokens is None:
            image_features = self.backbone.encode_image(image, self.img_adapter_list)
            return {"features": image_features}
        else:
            image_features = self.backbone.encode_image(image, self.img_adapter_list)
            # print(image_features.shape)
            image_features = self.img_final_adapter(image_features)
            text_features = self.backbone.encode_text(text_tokens, self.text_adapter_list)
            image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
            text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            logit_scale = self.backbone.logit_scale.exp()
            logits_per_image = logit_scale * image_features_normed @ text_features_normed.t()
            logits_per_text = logits_per_image.t()
            return {"logits": logits_per_image, "features": image_features}

    # def forward(self, image, text_tokens):
    #     logits_per_image, logits_per_text, image_features, text_features = self.backbone(image, text_tokens, self.img_adapter_list, self.text_adapter_list)
    #     logits_per_image = logits_per_image.softmax(dim=-1)
    #     return {"logits": logits_per_image, "features": image_features}

    def forward_with_vectors(self, x, text_tokens):
        x = x.type(self.backbone.dtype)
        image_features = self.img_final_adapter(x)
        text_features = self.backbone.encode_text(text_tokens, self.text_adapter_list)
        image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
        text_features_normed = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.backbone.logit_scale.exp()
        logits_per_image = logit_scale * image_features_normed @ text_features_normed.t()
        logits_per_text = logits_per_image.t()
        return {"logits": logits_per_image, "features": image_features}

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

    def freeze_adapter(self):
        for name, param in self.named_parameters():
            if "img_final_adapter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
