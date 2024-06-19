import torch
import torch.nn as nn
import copy
import timm
from safetensors.torch import load_file
from model.Base_Net import Base_Net
from model.backbone import *


class Inc_Net(Base_Net):
    def __init__(self, config, logger):
        super(Inc_Net, self).__init__(config, logger)
        self.old_fc_state_dict = None
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

    def update_fc(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes

        new_fc = nn.Linear(self.feature_dim, cur_classes)
        # self.reset_fc_parameters(new_fc)

        if self.fc is not None:
            new_fc.weight.data[:known_classes, :] = copy.deepcopy(self.fc.weight.data)
            new_fc.bias.data[:known_classes] = copy.deepcopy(self.fc.bias.data)
            self.logger.info('Updated classifier head output dim from {} to {}'.format(known_classes, cur_classes))
        else:
            self.logger.info('Created classifier head with output dim {}'.format(cur_classes))
        del self.fc
        self.fc = new_fc

    def create_all_class_fc(self, num_classes):
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.logger.info('Created classifier head with output dim {}'.format(num_classes))

    def fc_backup(self):
        self.old_fc_state_dict = copy.deepcopy(self.fc.state_dict())

    def fc_recall(self):
        self.fc.load_state_dict(self.old_fc_state_dict)

    def forward(self, x, train=True, task_id=None, fc_only=False):
        if fc_only:
            logits = self.fc(x)
            return {"logits": logits}
        else:
            if hasattr(self.backbone, "forward_features"):
                features = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
            else:
                features = self.backbone(x)
            logits = self.fc(features)

            return {"logits": logits, "features": features}

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma
