import os
import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.CosineInc_Net import CosineIncrementalNet
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class UCIR(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.memory_bank = ReplayBank(config, logger)

        self.lambda_base = config.lambda_base
        self.K = config.K
        self.margin = config.margin
        self.nb_proxy = config.nb_proxy

        if config.increment_type != 'CIL':
            raise ValueError('UCIR is a class incremental method!')

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = CosineIncrementalNet(self.config, self.logger)
            self.model.model_init()
        self.model.update_fc(task_id)
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        for name, param in self.model.fc.named_parameters():
            if 'fc1' in name:
                param.requires_grad = False
                self.logger.info('{} requires_grad=False'.format(name))
        self.logger.info('Freezing SplitCosineLinear fc1 ...')
        # self.logger.info('All params: {}'.format(count_parameters(self._network)))
        # self.logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self.model = self.model.cuda()

        if self.old_model is not None:
            self.old_model = self.old_model.cuda()

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")

        if task_id == 0:
            self.lamda = 0
        else:
            self.lamda = self.lambda_base * math.sqrt(self.known_classes / (self.cur_classes - self.known_classes))
        self.logger.info('Adaptive lambda: {}'.format(self.lamda))
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = F.cosine_embedding_loss
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)

        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = self.model.module

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses, lf_losses, is_losses = 0., 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs)
            logits = out["logits"]
            features = out["features"]
            outputs = F.softmax(logits, dim=-1)
            preds = torch.max(outputs[:, :self.cur_classes], dim=1)[1]
            assert logits.shape[1] == self.cur_classes, "epoch train error"

            # ce loss version implementation
            ce_loss = hard_loss(logits[:, :self.cur_classes], targets)
            # ce_loss = F.cross_entropy(logits[:, :self.cur_classes], targets)
            ce_losses += ce_loss.item()
            if task_id == 0:
                loss = ce_loss
            else:
                old_out = self.old_model(inputs)
                old_features = old_out["features"]
                lf_loss = soft_loss(features, old_features.detach(),
                                                  torch.ones(inputs.shape[0]).cuda()) * self.lamda
                lf_losses += lf_loss.item()
                loss = ce_loss + lf_loss

            old_classes_sample_mask = np.where(targets.cpu().numpy() < self.known_classes)[0]
            if len(old_classes_sample_mask) != 0:
                old_targets = targets[old_classes_sample_mask]
                new_scores = out['new_scores'][old_classes_sample_mask]
                old_scores = out['old_scores'][old_classes_sample_mask]
                old_bool_onehot = F.one_hot(old_targets, self.known_classes).type(torch.bool)
                anchor_positive = torch.masked_select(old_scores, old_bool_onehot)
                anchor_positive = anchor_positive.view(-1, 1).repeat(1, self.K)
                anchor_hard_negative = new_scores.topk(self.K, dim=1)[0]
                is_loss = F.margin_ranking_loss(anchor_positive, anchor_hard_negative,
                                                torch.ones_like(anchor_positive).cuda(), margin=self.margin)
                is_losses += is_loss.item()
                loss += is_loss

            if idx == 0:
                all_preds = preds
                all_targets = targets
            else:
                all_preds = torch.cat((all_preds, preds))
                all_targets = torch.cat((all_targets, targets))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

        train_loss = {'all_loss': losses / len(train_loader), 'loss_clf': ce_losses / len(train_loader),
                      'loss_lf': lf_losses / len(train_loader), 'loss_is': is_losses / len(train_loader)}

        return all_preds, all_targets, train_loss

    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        losses = 0.
        ce_losses, lf_losses, is_losses = 0., 0., 0.
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs)
                logits = out["logits"]
                features = out["features"]
                outputs = F.softmax(logits, dim=-1)
                preds = torch.max(outputs[:, :self.cur_classes], dim=1)[1]

                # ce loss
                ce_loss = hard_loss(logits[:, :self.cur_classes], targets)
                ce_losses += ce_loss.item()
                if task_id == 0:
                    loss = ce_loss
                else:
                    old_out = self.old_model(inputs)
                    old_features = old_out["features"]
                    lf_loss = soft_loss(features, old_features.detach(),
                                                      torch.ones(inputs.shape[0]).cuda()) * self.lamda
                    lf_losses += lf_loss.item()
                    loss = ce_loss + lf_loss

                old_classes_sample_mask = np.where(targets.cpu().numpy() < self.known_classes)[0]
                if len(old_classes_sample_mask) != 0:
                    old_targets = targets[old_classes_sample_mask]
                    new_scores = out['new_scores'][old_classes_sample_mask]
                    old_scores = out['old_scores'][old_classes_sample_mask]
                    old_bool_onehot = F.one_hot(old_targets, self.known_classes).type(torch.bool)
                    anchor_positive = torch.masked_select(old_scores, old_bool_onehot)
                    anchor_positive = anchor_positive.view(-1, 1).repeat(1, self.K)
                    anchor_hard_negative = new_scores.topk(self.K, dim=1)[0]
                    is_loss = F.margin_ranking_loss(anchor_positive, anchor_hard_negative,
                                                    torch.ones_like(anchor_positive).cuda(), margin=self.margin)
                    is_losses += is_loss.item()
                    loss += is_loss
                losses += loss.item()
                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))

            test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader),
                         'loss_lf': lf_losses / len(test_loader), 'loss_is': is_losses / len(test_loader)}
            return all_preds, all_targets, test_loss

    def after_task(self, task_id):
        super().after_task(task_id)
        self.old_model = copy.deepcopy(self.model).freeze()
