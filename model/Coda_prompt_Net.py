import copy
import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file
from utils.functions import *
from utils.train_utils import ortho_penalty
from model.backbone import *
from model.Base_Net import Base_Net

class Coda_prompt_module(nn.Module):
    def __init__(self, emb_dim, prompt_pool_size, prompt_length, orth_mu, task_num, pt_type="prefix_t", key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_dim
        self.key_d = key_dim
        self.pt_type = pt_type
        # prompt locations
        self.p_layers = [0, 1, 2, 3, 4]
        self.p_length = prompt_length
        self.p_pool_size = prompt_pool_size
        self.p_per_task = int(self.p_pool_size / task_num)
        self.orth_mu = orth_mu
        # e prompt init
        for l in self.p_layers:
            p = tensor_prompt(self.p_pool_size, self.p_length, emb_dim)
            k = tensor_prompt(self.p_pool_size, self.key_d)
            a = tensor_prompt(self.p_pool_size, self.key_d)
            setattr(self, f'p_{l}', p)
            setattr(self, f'k_{l}', k)
            setattr(self, f'a_{l}', a)

    def prompt_init_orth(self, task_id):
        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more
        # fair in the spirit of continual learning and has little affect on performance
        #
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for l in self.p_layers:
            k = getattr(self, f'k_{l}')
            a = getattr(self, f'a_{l}')
            p = getattr(self, f'p_{l}')
            k = self.gram_schmidt(k, task_id)
            a = self.gram_schmidt(a, task_id)
            p = self.gram_schmidt(p, task_id)
            setattr(self, f'p_{l}', p)
            setattr(self, f'k_{l}', k)
            setattr(self, f'a_{l}', a)

    '''Schmidt orthogonalization '''
    def gram_schmidt(self, A0, task_id):
        def projection(u, v):
            denominator = (u * u).sum()
            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(A0.shape) == 3  # only prompt matrix
        if is_3d:
            shape_2d = copy.deepcopy(A0.shape)
            A0 = A0.view(A0.shape[0], -1)

        A0 = A0.T
        A1 = torch.zeros_like(A0, device=A0.device)

        # get starting point
        old_component = int(task_id * self.p_per_task)
        cur_component = int((task_id + 1) * self.p_per_task)
        if old_component > 0:
            A1[:, 0:old_component] = A0[:, 0:old_component].clone()
        for i in range(old_component, cur_component):
            redo = True
            while redo:
                redo = False
                alpha_i = torch.randn_like(A0[:, i]).to(A0.device)
                offset = 0
                for j in range(0, i):
                    if not redo:
                        beta_j = A1[:, j].clone()
                        proj = projection(beta_j, alpha_i)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            offset = offset + proj
                if not redo:
                    A1[:, i] = alpha_i - offset
        # 单位化
        for i in range(old_component, cur_component):
            A1_new = A1[:, i].clone()
            A1[:, i] = A1_new / (A1_new.norm())

        A1 = A1.T
        if is_3d:
            A1 = A1.view(shape_2d)

        return torch.nn.Parameter(A1)

    def forward(self, query, train, task_id=None):
        # prompts
        all_p = {}
        all_orth_loss = {}
        B, C = query.shape
        for l in self.p_layers:
            K = getattr(self, f'k_{l}')
            A = getattr(self, f'a_{l}')
            P = getattr(self, f'p_{l}')

            old_component = int(task_id * self.p_per_task)
            cur_component = int((task_id + 1) * self.p_per_task)

            # cosine similarity to match keys/querries
            if train:
                if task_id > 0:
                    K = torch.cat((K[:old_component].detach().clone(), K[old_component:cur_component]), dim=0)
                    A = torch.cat((A[:old_component].detach().clone(), A[old_component:cur_component]), dim=0)
                    P = torch.cat((P[:old_component].detach().clone(), P[old_component:cur_component]), dim=0)
                else:
                    K = K[0:cur_component]
                    A = A[0:cur_component]
                    P = P[0:cur_component]
            else:
                K = K[0:cur_component]
                A = A[0:cur_component]
                P = P[0:cur_component]

            attn_query = torch.einsum('bd,kd->bkd', query, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(attn_query, dim=2)
            alpha = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            weighted_P = torch.einsum('bk,kld->bld', alpha, P)

            # select prompts
            if self.pt_type == "prefix_t":
                i = int(self.p_length / 2)
                prefix_k = weighted_P[:, :i, :]
                prefix_v = weighted_P[:, i:, :]
                p_return = [prefix_k, prefix_v]
            else:
                raise ValueError("Unknown prompt tuning type!")

            if self.orth_mu > 0:
                orth_loss = ortho_penalty(K) * self.orth_mu
                orth_loss += ortho_penalty(A) * self.orth_mu
                orth_loss += ortho_penalty(P.view(P.shape[0], -1)) * self.orth_mu
            else:
                orth_loss = 0.

            all_p[str(l)] = p_return
            all_orth_loss[str(l)] = orth_loss

        # return
        return all_p, all_orth_loss


class Coda_prompt_Net(Base_Net):
    def __init__(self, config, logger):
        super(Coda_prompt_Net, self).__init__(config, logger)
        self.prompt_pool = None
        self.embed_dim = 768

    def model_init(self):
        self.prompt_pool = Coda_prompt_module(self.embed_dim, self.config.prompt_pool_size, self.config.prompt_length,
                                              self.config.orth_mu, len(self.config.increment_steps), pt_type=self.config.pt_type)
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
            all_prompts, all_prompt_loss = self.prompt_pool(q, train, task_id)
            out = self.backbone(x, all_prompts=all_prompts)
            out = out[:, 0, :]
            for loss in all_prompt_loss.values():
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
        if self.prompt_pool is not None:
            self.prompt_pool.prompt_init_orth(task_id)
            self.logger.info('prompt init orth finished!')
        new_fc = nn.Linear(self.embed_dim, cur_classes)
        if self.fc is not None:
            new_fc.weight.data[:known_classes, :] = copy.deepcopy(self.fc.weight.data)
            new_fc.bias.data[:known_classes] = copy.deepcopy(self.fc.bias.data)
            self.logger.info('Updated classifier head output dim from {} to {}'.format(known_classes, cur_classes))
        else:
            self.logger.info('Created classifier head with output dim {}'.format(cur_classes))
        del self.fc
        self.fc = new_fc
