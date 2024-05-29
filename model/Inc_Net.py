import torch
import torch.nn as nn
import copy
import timm
from model.backbone import *


class Inc_Net(nn.Module):
    def __init__(self, logger):
        super(Inc_Net, self).__init__()
        # self.feature_extractor_list = nn.ModuleList()
        self.logger = logger
        self.feature_dim = None
        self.fc = None
        self.backbone = None
        # self.aux_fc = None
        # self.feature_dim_list = []

    def model_init(self, config):
        self.backbone = timm.create_model(model_name=config.backbone,
                                          drop_rate=config.drop_rate,
                                          drop_path_rate=config.drop_path_rate
                                          )
        if config.pretrained_path:
            self.backbone.load_state_dict(load_file(config.pretrained_path))
        self.feature_dim = self.backbone.num_features

    def update_fc(self, config, task_id):
        new_classes = config.increment_steps[task_id]
        known_classes = sum(config.increment_steps[:task_id])
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


    def forward(self, x):
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
        else:
            features = self.backbone(x)
        logits = self.fc(features)

        return {"logits": logits, "features": features}

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def reset_fc_parameters(self, fc):
        nn.init.kaiming_uniform_(fc.weight, nonlinearity='linear')
        nn.init.constant_(fc.bias, 0)

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma