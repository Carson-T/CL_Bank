import torch
import torch.nn as nn
import copy
import timm
from safetensors.torch import load_file
from model.Base_Net import Base_Net
from model.backbone import *


class DA_Net(Base_Net):
    def __init__(self, config, logger):
        super(DA_Net, self).__init__(config, logger)
        self.old_fc_state_dict = None
        self.old_adapter = None
        self.projector = None
        self.adapter_list = nn.ModuleList([])
        self.dm = config.dm
        # self.aux_fc = None
        # self.feature_dim_list = []

    def model_init(self):
        self.backbone = timm.create_model(model_name=self.config.backbone,
                                          drop_rate=self.config.drop_rate,
                                          drop_path_rate=self.config.drop_path_rate
                                          )
        # if self.config.pretrained_path:
        #     self.backbone.load_state_dict(load_file(self.config.pretrained_path))
        if self.config.pretrained_path:
            state_dict = load_file(self.config.pretrained_path)
            for key in list(state_dict.keys()):
                if 'qkv.weight' in key:
                    qkv_weight = state_dict.pop(key)
                    q_weight = qkv_weight[:768]
                    k_weight = qkv_weight[768:768 * 2]
                    v_weight = qkv_weight[768 * 2:]
                    state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
                    state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
                    state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
                elif 'qkv.bias' in key:
                    qkv_bias = state_dict.pop(key)
                    q_bias = qkv_bias[:768]
                    k_bias = qkv_bias[768:768 * 2]
                    v_bias = qkv_bias[768 * 2:]
                    state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
                    state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
                    state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
            del state_dict['head.weight']
            del state_dict['head.bias']
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            self.logger.info("missing keys:{}".format(missing))
            self.logger.info("unexpected keys:{}".format(unexpected))
        self.feature_dim = self.backbone.num_features

        for i in range(len(self.backbone.blocks)):
            adapter = Adapter(d_model=768,
                              dropout=0.1,
                              bottleneck=64,
                              init_option="lora",
                              adapter_scalar="0.1",
                              adapter_layernorm_option=None,
                              )
            self.adapter_list.append(adapter)

    def update_model(self, task_id):
        self.update_fc(task_id)
        if self.dm and task_id > 0:
            del self.projector
            self.projector = nn.Linear(self.feature_dim, self.feature_dim)
            del self.old_adapter
            self.old_adapter = copy.deepcopy(self.adapter_list)
            for param in self.old_adapter.parameters():
                param.requires_grad = False

    def fc_backup(self):
        self.old_fc_state_dict = copy.deepcopy(self.fc.state_dict())

    def fc_recall(self):
        if self.old_fc_state_dict is not None:
            self.fc.load_state_dict(self.old_fc_state_dict)

    def map_proto(self, proto):
        with torch.no_grad():
            mapped_proto = self.projector(proto)
        return mapped_proto

    def forward(self, x, proto=None, fc_only=False):
        if fc_only:
            logits = self.fc(x)
            return {"logits": logits}
        else:
            if self.old_adapter is not None:
                features = self.backbone(x, cur_adapter=self.adapter_list)
                old_features = self.backbone(x, cur_adapter=self.old_adapter)
                old_features = self.projector(old_features)
                if proto is not None:
                    proto = self.projector(proto)
                # both_features = torch.cat([old_features.unsqueeze(1), features.unsqueeze(1)], dim=1)
                logits = self.fc(features)

                return {"logits": logits, "features": features, "old_features": old_features, "proto": proto}
            else:
                features = self.backbone(x, cur_adapter=self.adapter_list)
                logits = self.fc(features)

                return {"logits": logits, "features": features}

