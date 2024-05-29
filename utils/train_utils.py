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


def get_loss_func(config):
    loss_func = None
    if config.loss_func == "CEloss":
        loss_func = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing if config.label_smoothing is not None else 0)

    return loss_func


def KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda())**2).mean()
