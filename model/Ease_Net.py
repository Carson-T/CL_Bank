import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from model.Base_Net import Base_Net
from model.backbone import *


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


class EaseCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(EaseCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out
        return out

    def forward_reweight(self, input, task_id, increment_steps, alpha=0.1, beta=0.0, out_dim=768,
                         use_init_ptm=False):
        for i in range(task_id + 1):
            if i == 0:
                start_cls = 0
                end_cls = increment_steps[i]
            else:
                start_cls = sum(increment_steps[:i])
                end_cls = start_cls + increment_steps[i]

            out = 0
            for j in range((self.in_features // out_dim)):
                # PTM feature
                if use_init_ptm and j == 0:
                    input_ptm = F.normalize(input[:, 0:out_dim], p=2, dim=1)
                    weight_ptm = F.normalize(self.weight[start_cls:end_cls, 0:out_dim], p=2, dim=1)
                    out_ptm = beta * F.linear(input_ptm, weight_ptm)
                    out += out_ptm
                    continue

                input1 = F.normalize(input[:, j * out_dim:(j + 1) * out_dim], p=2, dim=1)
                weight1 = F.normalize(self.weight[start_cls:end_cls, j * out_dim:(j + 1) * out_dim], p=2, dim=1)
                if use_init_ptm:
                    if j != (i + 1):
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= task_id
                    else:
                        out1 = F.linear(input1, weight1)
                else:
                    if j != i:
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= task_id
                    else:
                        out1 = F.linear(input1, weight1)

                out += out1

            if i == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all, out), dim=1)

        if self.to_reduce:
            # Reduce_proxy
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all

        return out_all

class EaseNet(Base_Net):

    def __init__(self, config, logger):
        super(EaseNet, self).__init__(config, logger)

        self.adapter_list = nn.ModuleList()
        self.cur_adapter = None
        self.proxy_fc = None

        self.use_init_ptm = config.use_init_ptm
        self.alpha = config.alpha
        self.beta = config.beta
        self.embed_dim = 768

    def model_init(self):
        backbone = timm.create_model(model_name=self.config.backbone,
                                     img_size=self.config.img_size,
                                     patch_size=16,
                                     embed_dim=self.embed_dim,
                                     depth=12,
                                     num_heads=12,
                                     )
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
            backbone.load_state_dict(state_dict, strict=False)
        self.backbone = backbone
        self.feature_dim = self.backbone.num_features

    def update_model(self, task_id):
        self.update_adapter()
        self.update_fc(task_id)

    def update_adapter(self):
        if self.cur_adapter is not None:
            self.adapter_list.append(copy.deepcopy(self.cur_adapter))
        self.cur_adapter = nn.ModuleList()
        for i in range(len(self.backbone.blocks)):
            adapter = Adapter(d_model=768,
                              dropout=0.1,
                              bottleneck=64,
                              init_option="lora",
                              adapter_scalar="0.1",
                              adapter_layernorm_option=None,
                              )
            self.cur_adapter.append(adapter)
        self.logger.info("old adapter num {}".format(len(self.adapter_list)))

    # (proxy_fc = cls * dim)
    def update_fc(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes
        if self.use_init_ptm:
            self.feature_dim = self.num_features * (task_id+2)
        else:
            self.feature_dim = self.num_features * (task_id+1)
        if self.proxy_fc is not None:
            del self.proxy_fc
        self.proxy_fc = self.generate_fc(self.num_features, new_classes).cuda()

        fc = self.generate_fc(self.feature_dim, cur_classes).cuda()
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = copy.deepcopy(self.fc.sigma.data)
            assert self.fc.out_features == known_classes
            fc.weight.data[:known_classes, : -self.num_features] = nn.Parameter(weight)
            self.logger.info('Updated classifier head output dim from {} to {}'.format(known_classes, cur_classes))
        else:
            self.logger.info('Created classifier head with output dim {}'.format(cur_classes))
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = EaseCosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x, train=False, task_id=None):
        if train:
            features = self.backbone.forward(x, self.adapter_list, self.cur_adapter, use_old=False)
            logits = self.proxy_fc(features)    # B new_class_num
        else:
            features = self.backbone.forward(x, self.adapter_list, self.cur_adapter, use_old=True, use_init_ptm=self.use_init_ptm)
            if not self.config.use_reweight:
                logits = self.fc(features)      # B cur_class_num
            else:
                logits = self.fc.forward_reweight(features,
                                                  task_id=task_id,
                                                  increment_steps=self.config.increment_steps,
                                                  alpha=self.alpha,
                                                  beta=self.beta,
                                                  use_init_ptm=self.use_init_ptm)

        return {"logits": logits, "features": features}

    def freeze_fe(self):
        for name, param in self.named_parameters():
            if "cur_adapter" in name:
                param.requires_grad = True
            elif "adapter_list" in name:
                param.requires_grad = False
            elif "backbone" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.logger.info("backbone and old adapter freeze, current adapter unfreeze!")

    # def show_trainable_params(self):
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             print(name, param.numel())