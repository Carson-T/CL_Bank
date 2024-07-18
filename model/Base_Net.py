import torch
import torch.nn as nn
import copy
import timm
from safetensors.torch import load_file
from abc import abstractmethod
from model.backbone import *


class Base_Net(nn.Module):
    def __init__(self, config, logger):
        super(Base_Net, self).__init__()
        self.config = config
        self.logger = logger
        self.feature_dim = None
        self.backbone = None
        self.fc = None

    @abstractmethod
    def model_init(self):
        pass

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

    @abstractmethod
    def forward(self, x):
        pass

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        return self

    def freeze_fe(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def reset_fc_parameters(self, fc):
        nn.init.kaiming_uniform_(fc.weight, nonlinearity='linear')
        nn.init.constant_(fc.bias, 0)

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.logger.info("{} {}".format(name, param.numel()))