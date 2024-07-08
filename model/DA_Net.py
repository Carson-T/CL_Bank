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
        self.old_fe = None
        self.projector = None
        # self.aux_fc = None
        # self.feature_dim_list = []

    def model_init(self):
        self.backbone = timm.create_model(model_name=self.config.backbone,
                                          drop_rate=self.config.drop_rate,
                                          drop_path_rate=self.config.drop_path_rate
                                          )
        if self.config.pretrained_path:
            self.backbone.load_state_dict(load_file(self.config.pretrained_path))
        self.feature_dim = self.backbone.num_features

        self.projector = nn.Linear(self.feature_dim, self.feature_dim)

    def save_old_fe(self):
        self.old_fe = copy.deepcopy(self.backbone)
        for param in self.old_fe.parameters():
            param.requires_grad = False

    def fc_backup(self):
        self.old_fc_state_dict = copy.deepcopy(self.fc.state_dict())

    def fc_recall(self):
        self.fc.load_state_dict(self.old_fc_state_dict)

    def map_proto(self, proto):
        mapped_proto = self.projector(proto)
        return mapped_proto

    def forward(self, x, fc_only=False):
        if fc_only:
            logits = self.fc(x)
            return {"logits": logits}
        else:
            if self.old_fe is not None:
                if hasattr(self.backbone, "forward_features"):
                    features = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
                    old_features = self.old_fe.forward_head(self.back.forward_features(x), pre_logits=True)
                else:
                    features = self.backbone(x)
                    old_features = self.old_fe(x)
                old_features = self.projector(old_features)
                both_features = torch.cat([old_features.unsqueeze(1), features.unsqueeze(1)], dim=1)
                logits = self.fc(features)

                return {"logits": logits, "features": features, "both_features": both_features}
            else:
                if hasattr(self.backbone, "forward_features"):
                    features = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
                else:
                    features = self.backbone(x)
                logits = self.fc(features)

                return {"logits": logits, "features": features}


