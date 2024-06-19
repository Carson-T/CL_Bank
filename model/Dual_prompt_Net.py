import copy
import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file
from utils.functions import *
from model.backbone import *
from model.Base_Net import Base_Net


class Dual_prompt_module(nn.Module):
    def __init__(self, emb_dim, e_prompt_pool_size, g_prompt_length, e_prompt_length, pt_type="prefix_t", key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_dim
        self.key_d = key_dim
        self.pt_type = pt_type

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        # prompt pool size
        self.g_p_length = g_prompt_length
        self.e_p_length = e_prompt_length
        self.e_pool_size = e_prompt_pool_size

        # prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_dim)
            setattr(self, f'g_p_{g}', p)

            # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_dim)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def forward(self, query, train, task_id=None):
        # prompts
        e_qk_loss = 0.
        all_p = {}
        all_qk_loss = {}
        B, C = query.shape
        for l in set(self.e_layers+self.g_layers):
            e_valid = False
            if l in self.e_layers:
                e_valid = True
                K = getattr(self, f'e_k_{l}')  # 0 based indexing here
                p = getattr(self, f'e_p_{l}')  # 0 based indexing here

                # cosine similarity to match keys/querries
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(query, dim=1).detach()
                cos_sim = torch.einsum('bj,kj->bk', q, n_K)
                # print(cos_sim.shape)
                if train:
                    assert task_id is not None, "task_id is missed!"
                    e_qk_loss = (1.0 - cos_sim[:, task_id]).mean()
                    selected_P = p[task_id].expand(B, -1, -1)
                else:
                    _, top_1 = torch.topk(cos_sim, 1, dim=1)
                    e_qk_loss = (1.0 - cos_sim[:, top_1]).sum(dim=-1).mean()
                    selected_P = p[top_1].squeeze(dim=1)

                # select prompts
                if self.pt_type == "prefix_t":
                    i = int(self.e_p_length / 2)
                    e_prefix_k = selected_P[:, :i, :].reshape((B, -1, self.emb_d))
                    e_prefix_v = selected_P[:, i:, :].reshape((B, -1, self.emb_d))
                else:
                    raise ValueError("Unknown prompt tuning type!")

            g_valid = False
            if l in self.g_layers:
                g_valid = True
                p = getattr(self, f'g_p_{l}')  # 0 based indexing here
                G_P = p.expand(B, -1, -1)
                if self.pt_type == "prefix_t":
                    j = int(self.g_p_length / 2)
                    g_prefix_k = G_P[:, :j, :]
                    g_prefix_v = G_P[:, j:, :]
                else:
                    raise ValueError("Unknown prompt tuning type!")

            if e_valid and g_valid:
                Pk = torch.cat((e_prefix_k, g_prefix_k), dim=1)
                Pv = torch.cat((e_prefix_v, g_prefix_v), dim=1)
                p_return = [Pk, Pv]
            elif e_valid:
                p_return = [e_prefix_k, e_prefix_v]
            elif g_valid:
                p_return = [g_prefix_k, g_prefix_v]

            all_p[str(l)] = p_return
            all_qk_loss[str(l)] = e_qk_loss
            # print(qk_loss)

        # return
        return all_p, all_qk_loss


class Dual_prompt_Net(Base_Net):
    def __init__(self, config, logger):
        super(Dual_prompt_Net, self).__init__(config, logger)
        self.prompt_pool = None
        self.embed_dim = 768

    def model_init(self):
        assert self.config.e_prompt_pool_size == len(self.config.increment_steps), "special prompt num should be the same with the task num"
        self.prompt_pool = Dual_prompt_module(self.embed_dim, self.config.e_prompt_pool_size, self.config.g_prompt_length, self.config.e_prompt_length, pt_type=self.config.pt_type)
        # get feature encoder
        backbone = timm.create_model(model_name=self.config.backbone,
                                     pt_type=self.config.pt_type,
                                     img_size=self.config.img_size,
                                     patch_size=16,
                                     embed_dim=self.embed_dim,
                                     depth=12,
                                     num_heads=12,
                                     )
        if self.config.pretrained_path:
            state_dict = load_file(self.config.pretrained_path)
            del state_dict['head.weight']
            del state_dict['head.bias']
            backbone.load_state_dict(state_dict, strict=True)
        self.backbone = backbone

    def forward(self, x, train=False, task_id=None):
        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        if self.prompt_pool is not None:
            with torch.no_grad():
                q = self.backbone(x)
                q = q[:, 0, :]
            all_prompts, all_qk_loss = self.prompt_pool(q, train, task_id)
            out = self.backbone(x, all_prompts=all_prompts)
            out = out[:, 0, :]
            for loss in all_qk_loss.values():
                # print(loss)
                prompt_loss += loss
        else:
            out = self.backbone(x)
            out = out[:, 0, :]
        features = out.view(out.size(0), -1)
        out = self.fc(features)
        return {"logits": out, "features": features, "prompt_loss": prompt_loss}

    def freeze_fe(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.logger.info('Freezing feature extractor(requires_grad=False) ...')
        return self

    def update_fc(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes
        new_fc = nn.Linear(self.embed_dim, cur_classes)
        if self.fc is not None:
            new_fc.weight.data[:known_classes, :] = copy.deepcopy(self.fc.weight.data)
            new_fc.bias.data[:known_classes] = copy.deepcopy(self.fc.bias.data)
            self.logger.info('Updated classifier head output dim from {} to {}'.format(known_classes, cur_classes))
        else:
            self.logger.info('Created classifier head with output dim {}'.format(cur_classes))
        del self.fc
        self.fc = new_fc
