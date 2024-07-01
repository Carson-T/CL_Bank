import torch
import torch.nn as nn
import copy
import timm
from model.backbone import *
from model.Base_Net import Base_Net
from safetensors.torch import load_file


class DERNet(Base_Net):
    def __init__(self, config, logger):
        super(DERNet, self).__init__(config, logger)
        self.feature_extractor_list = nn.ModuleList()
        self.aux_fc = None
        self.feature_dim_list = []

    def model_init(self):
        pass

    def update(self, task_id):
        backbone = timm.create_model(model_name=self.config.backbone,
                                     drop_rate=self.config.drop_rate,
                                     drop_path_rate=self.config.drop_path_rate
                                     )
        if len(self.feature_extractor_list) == 0:
            if self.config.pretrained_path:
                backbone.load_state_dict(load_file(self.config.pretrained_path))
        else:
            backbone.load_state_dict(self.feature_extractor_list[-1].state_dict())
        self.feature_extractor_list.append(backbone)
        self.feature_dim_list.append(backbone.num_features)

        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
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

    def forward(self, x, train=False, task_id=None):
        if hasattr(self.feature_extractor_list[0], "forward_features"):
            features = [fe.forward_head(fe.forward_features(x), pre_logits=True) for fe in self.feature_extractor_list]
            all_features = torch.cat(features, 1)
        else:
            features = [fe(x) for fe in self.feature_extractor_list]
            all_features = torch.cat(features, 1)
        logits = self.fc(all_features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[-1])

        return {"logits": logits, "aux_logits": aux_logits, "features": all_features}

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

