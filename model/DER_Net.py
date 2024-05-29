import torch
import torch.nn as nn
import copy
import timm
from model.backbone import *


class DERNet(nn.Module):
    def __init__(self):
        super(DERNet, self).__init__()
        self.feature_extractor_list = nn.ModuleList()
        self.feature_dim = None
        self.fc = None
        self.aux_fc = None
        self.feature_dim_list = []

    def update(self, config, task_id):
        backbone = timm.create_model(model_name=config.backbone,
                                     drop_rate=config.drop_rate,
                                     drop_path_rate=config.drop_path_rate
                                     )
        if len(self.feature_extractor_list) == 0:
            if config.pretrained_path:
                backbone.load_state_dict(load_file(config.pretrained_path))
        else:
            backbone.load_state_dict(self.feature_extractor_list[-1].state_dict())
        self.feature_extractor_list.append(backbone)
        self.feature_dim_list.append(backbone.num_features)

        new_classes = config.increment_steps[task_id]
        known_classes = sum(config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes

        new_fc = nn.Linear(sum(self.feature_dim_list), cur_classes)
        # self.reset_fc_parameters(new_fc)
        self.aux_fc = nn.Linear(self.feature_dim_list[-1], new_classes + 1)

        if self.fc is not None:
            new_fc.weight.data[:known_classes, :-self.feature_dim_list[-1]] = copy.deepcopy(self.fc.weight.data)
            new_fc.bias.data[:known_classes] = copy.deepcopy(self.fc.bias.data)
            self.logger.info('Updated classifier head output dim from {} to {}'.format(known_classes, cur_classes))
        else:
            self.logger.info('Created classifier head with output dim {}'.format(cur_classes))
        del self.fc
        self.fc = new_fc

    def forward(self, x):
        features = [fe.forward_head(fe.forward_features(x), pre_logits=True) for fe in self.feature_extractor_list]
        all_features = torch.cat(features, 1)

        logits = self.fc(all_features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[-1])

        return {"logits": logits, "aux_logits": aux_logits, "features": all_features}

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def change_new_feature_extractors_grad(self, requires_grad):
        for param in self.feature_extractor_list[-1].parameters():
            param.requires_grad = requires_grad
        if requires_grad:
            self.feature_extractor_list[-1].train()
        else:
            self.feature_extractor_list[-1].eval()

    def freeze_old_feature_extractors(self, task_id):
        for i in range(task_id):
            for p in self.feature_extractor_list[i].parameters():
                p.requires_grad = False
            self.feature_extractor_list[i].eval()

    def reset_fc_parameters(self, fc):
        nn.init.kaiming_uniform_(fc.weight, nonlinearity='linear')
        nn.init.constant_(fc.bias, 0)
