import copy
import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file
from utils.functions import *
from model.backbone import *
from model.Base_Net import Base_Net

class Prompt_Pool(nn.Module):
    def __init__(self, emb_dim, prompt_pool_size, prompt_length, top_k, shallow_or_deep, pt_type, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_dim
        self.key_d = key_dim
        self.top_k = top_k
        self.pt_type = pt_type
        self.p_length = int(prompt_length)
        self.p_pool_size = int(prompt_pool_size)

        if shallow_or_deep:  # true for shallow
            self.p_layers = [0]
        else:  # false for deep
            self.p_layers = [0, 1, 2, 3, 4]

        # prompt pool size

        # prompt init
        for i in self.p_layers:
            p = tensor_prompt(self.p_pool_size, self.p_length, self.emb_d)
            k = tensor_prompt(self.p_pool_size, self.key_d)
            setattr(self, f'p_{i}', p)
            setattr(self, f'k_{i}', k)

    def forward(self, query, train=False, task_id=None):
        # prompts
        # all_K = []
        all_p = {}
        all_qk_loss = {}
        B, C = query.shape
        for l in self.p_layers:
            K = getattr(self, f'k_{l}')  # 0 based indexing here
            p = getattr(self, f'p_{l}')  # 0 based indexing here

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(query, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            # print(cos_sim.shape)
            topk_sim, topk = torch.topk(cos_sim, self.top_k, dim=1)
            qk_loss = (1.0 - topk_sim).sum(dim=-1).mean()
            selected_P = p[topk]

            # select prompts
            if self.pt_type == "prefix_t":
                i = int(self.p_length / 2)
                prefix_k = selected_P[:, :, :i, :].reshape((B, -1, self.emb_d))
                prefix_v = selected_P[:, :, i:, :].reshape((B, -1, self.emb_d))
                p_return = [prefix_k, prefix_v]
            elif self.pt_type == "prompt_t":
                p_return = selected_P.reshape((B, -1, self.emb_d))
            else:
                raise ValueError("Unknown prompt tuning type!")
            all_p[str(l)] = p_return
            all_qk_loss[str(l)] = qk_loss
            # print(qk_loss)

        # return
        return all_p, all_qk_loss


class L2P_Net(Base_Net):
    def __init__(self, config, logger):
        super(L2P_Net, self).__init__(config, logger)

        self.prompt_pool = None
        self.embed_dim = 768

    def model_init(self):
        self.prompt_pool = Prompt_Pool(self.embed_dim, self.config.prompt_pool_size, self.config.prompt_length, self.config.top_k,
                                       self.config.shallow_or_deep, pt_type=self.config.pt_type)
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

    def forward(self, x):
        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        if self.prompt_pool is not None:
            with torch.no_grad():
                q = self.backbone(x)
                q = q[:, 0, :]
            all_prompts, all_qk_loss = self.prompt_pool(q)
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
            # if name != "pos_embed":
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
