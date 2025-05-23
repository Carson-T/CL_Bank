import numpy as np
import torch
from torch import nn
from torch import optim


def get_optimizer(params, config):
    optimizer = None
    if config.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)

    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    return optimizer


def get_scheduler(optimizer, config):
    scheduler = None
    if config.scheduler == 'multi_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.milestones,
                                                   gamma=config.gamma)

    elif config.scheduler == 'Warm-up-Cosine-Annealing':
        init_ratio, warm_up_steps, min_lr_ratio, max_steps = config.init_ratio, config.warm_up_steps, config.min_lr_ratio, config.epochs
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (1 - init_ratio) / (
                warm_up_steps - 1) * step + init_ratio
        if step < warm_up_steps - 1 else (1 - min_lr_ratio) * 0.5 * (np.cos((step - (warm_up_steps - 1)) /
                                                                            (max_steps - (
                                                                                    warm_up_steps - 1)) * np.pi) + 1) + min_lr_ratio)
    elif config.scheduler == "Cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.eta_min)

    elif config.scheduler is None:
        scheduler = None

    return scheduler


def get_loss_func(config, weight=None):
    loss_func = None
    if config.loss_func == "CEloss":
        loss_func = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing if config.label_smoothing is not None else 0, weight=weight)

    return loss_func


def KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda())**2).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_feature = anchor_feature / anchor_feature.norm(dim=-1, keepdim=True)
        contrast_feature = contrast_feature / contrast_feature.norm(dim=-1, keepdim=True)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss2(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, old_features, labels, proto=None, proto_label=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        labels = torch.cat([labels, labels], dim=0)
        if proto_label is not None:
            labels = torch.cat([labels, proto_label], dim=0)
        labels = labels.contiguous().view(-1, 1)
        num = len(labels)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat([features, old_features], dim=0)
        if proto is not None:
            contrast_feature = torch.cat([contrast_feature, proto], dim=0)
        anchor_feature = contrast_feature

        anchor_feature = anchor_feature / anchor_feature.norm(dim=-1, keepdim=True)
        contrast_feature = contrast_feature / contrast_feature.norm(dim=-1, keepdim=True)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        mean_log_prob_pos = mean_log_prob_pos[:len(labels)*2]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss