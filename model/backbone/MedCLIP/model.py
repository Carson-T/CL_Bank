import pdb
import os
import copy
from collections import defaultdict
import requests

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
import torchvision

WEIGHTS_NAME = "pytorch_model.bin"
class MedCLIPTextModel(nn.Module):
    def __init__(self,
                 proj_dim=512,
                 proj_bias=False) -> None:
        super().__init__()
        self.last_n_layer = 4
        config = AutoConfig.from_pretrained('/data/jiantao/pretrained_model/Bio_ClinicalBERT')   #   /data/jiantao/pretrained_model/Bio_ClinicalBERT   emilyalsentzer/Bio_ClinicalBERT
        self.model = AutoModel.from_config(config)
        # this tokenizer is actually not used
        self.tokenizer = AutoTokenizer.from_pretrained("/data/jiantao/pretrained_model/Bio_ClinicalBERT")
        self.tokenizer.model_max_length = 77
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # take the average of last four layers
        # last_hidden_states = torch.stack(output['hidden_states'][-self.last_n_layer:]) # n_layer, batch, seqlen, emb_dim
        # embed = last_hidden_states.permute(1,0,2,3)
        # embed = embed.mean(1).mean(1) # pooling

        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                          output['hidden_states'][-1]])  # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)  # pooling

        # let's take only the last hidden layer
        # embed = output['pooler_output']

        embed = self.projection_head(embed)
        return embed


class MedCLIPVisionModel(nn.Module):
    '''
    take resnet50 as backbone.
    '''

    def __init__(self, checkpoint=None, medclip_checkpoint=None):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias=False)  # projection head
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.', '')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

    def forward(self, pixel_values, **kwargs):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1, 3, 1, 1))
        img_embeds = self.model(pixel_values)
        return img_embeds


class MedCLIPVisionModelViT(nn.Module):
    '''take an VIT model as the backbone.
    '''

    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        config = AutoConfig.from_pretrained("/data/jiantao/pretrained_model/swin-tiny-patch4-window7-224")
        self.model = AutoModel.from_config(config)
        self.projection_head = nn.Linear(768, 512, bias=False)
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.', '')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

    def forward(self, pixel_values, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1, 3, 1, 1))
        output = self.model(pixel_values)
        img_embeds = output['pooler_output']
        if project:
            img_embeds = self.projection_head(img_embeds)
        return img_embeds


class MedCLIPModel(nn.Module):
    def __init__(self,
                 vision_cls=MedCLIPVisionModel,
                 checkpoint=None,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 ) -> None:
        super().__init__()
        text_proj_bias = False
        assert vision_cls in [MedCLIPVisionModel,
                              MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint)
        self.text_model = MedCLIPTextModel(proj_bias=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / logit_scale_init_value)))

        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME))
            self.load_state_dict(state_dict)
            print('load model weight from:', checkpoint)

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        state_dict = torch.load(os.path.join(input_dir, WEIGHTS_NAME))
        miss, unexp = self.load_state_dict(state_dict, strict=False)
        print("missing:", miss)
        print("unexpected:", unexp)
        print('load model weight from:', input_dir)

    def tokenize(self, text):
        out = self.text_model.tokenizer(text, return_tensors="pt", padding=True)
        return out

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds = self.text_model(input_ids, attention_mask)

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        return text_embeds

    def encode_image(self, pixel_values=None):
        # image encoder
        img_embeds = self.vision_model(pixel_values=pixel_values)

        return img_embeds

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                return_loss=None,
                **kwargs,
                ):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logits_per_image = self.compute_logits(img_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        if return_loss:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = None

        return {'img_embeds': img_embeds, 'text_embeds': text_embeds,
                'logits': logits_per_image, 'loss_value': loss, 'logits_per_text': logits_per_text}

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
