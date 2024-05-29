from .clip import load, tokenize
import torch
import torch.nn as nn


class MoE_Adapter_Net(nn.Module):
    def __init__(self, config, logger, jit=False):
        super().__init__()
        assert config.backbone == "CLIP"
        self.backbone, _, _ = load(config.pretrained_path, jit=jit)
        logger.info("model loaded!")

    def text_tokenize(self, cur_class_names, prompt_template):
        text_tokens = tokenize([prompt_template.format(c) for c in cur_class_names])
        return text_tokens

    def forward(self, image, text_tokens, train=False):
        if train:
            logits_per_image, logits_per_text, image_features, text_features = self.backbone(image, text_tokens, 0, is_train=train)
            logits_per_image = logits_per_image.softmax(dim=-1)
            return {"logits": logits_per_image, "features": image_features}

    def freeze_fe(self):
        for name, param in self.backbone.named_parameters():
            if "adaptmlp" not in name and "router" not in name and "noise" not in name:
                param.requires_grad = False