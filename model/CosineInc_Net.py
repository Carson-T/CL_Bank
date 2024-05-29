import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
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


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
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
                start_cls = increment_steps[i-1]
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


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1, out2), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return out, {
                'old_scores': reduce_proxies(out1, self.nb_proxy),
                'new_scores': reduce_proxies(out2, self.nb_proxy)}

class CosineIncrementalNet(nn.Module):
    def __init__(self, config, logger):
        super(CosineIncrementalNet, self).__init__()
        self.config = config
        self.logger = logger
        self.feature_dim = None
        self.fc = None
        self.backbone = None

    def model_init(self):
        self.backbone = timm.create_model(model_name=self.config.backbone,
                                          drop_rate=self.config.drop_rate,
                                          drop_path_rate=self.config.drop_path_rate
                                          )
        if self.config.pretrained_path:
            self.backbone.load_state_dict(load_file(self.config.pretrained_path))
        self.feature_dim = self.backbone.num_features

    def update_fc(self, nb_classes, task_id):
        fc = self.generate_fc(self.feature_dim, nb_classes, nb_proxy=self.config.nb_proxy)
        if self.fc is not None:
            if task_id == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim, nb_proxy=1):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, nb_proxy, to_reduce=True)
            self.logger.info('Created CosineLinear with output dim {}'.format(out_dim))
        else:
            prev_out_features = self.fc.out_features // nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, nb_proxy)
            self.logger.info('Updated CosineLinear output dim from {} to {}'.format(prev_out_features, out_dim))
        return fc

    def forward(self, x):
        ret = {}
        features = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
        ret["features"] = features
        out = self.fc(features)
        if isinstance(out, tuple):
            ret["logits"] = out[0]
            ret.update(out[1])
        else:
            ret["logits"] = out

        return ret

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self