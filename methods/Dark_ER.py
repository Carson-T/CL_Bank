import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.Inc_Net import Inc_Net
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class Dark_ER(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.memory_bank = ReplayBank(config, logger)

        self.alpha = config.alpha
        self.beta = config.beta

        if config.increment_type != 'CIL':
            raise ValueError('Dark_ER is a class incremental method!')

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = Inc_Net(self.config, self.logger)
            self.model.model_init()
            self.model.create_all_class_fc(sum(self.config.increment_steps))
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        self.model = self.model.cuda()

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        # self.logger.info("new feature extractor requires_grad=True")
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = F.mse_loss
        if len(self.config.device_ids.split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)

        if len(self.config.device_ids.split(",")) > 1:
            self.model = self.model.module

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses, kl_losses, buf_ce_losses = 0., 0., 0.
        model.train()
        for idx, (inputs, targets, indexs) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs)
            logits = out["logits"]
            # outputs = F.softmax(logits, dim=-1)
            preds = torch.max(logits[:, :self.cur_classes], dim=1)[1]
            # assert logits.shape[1] == self.cur_classes, "epoch train error"

            # ce loss version implementation
            ce_loss = hard_loss(logits, targets)
            ce_losses += ce_loss.item()
            loss = ce_loss
            if len(self.memory_bank.samples_memory) != 0:
                buf_inputs, _, buf_soft_targets = self.memory_bank.get_memory_reservoir(self.config.replay_batch_size,
                                                                                         self.train_dataset.use_path,
                                                                                         self.train_dataset.transform)
                buf_inputs, buf_soft_targets = buf_inputs.cuda(), buf_soft_targets.cuda()

                kl_loss = soft_loss(model(buf_inputs)["logits"], buf_soft_targets) * self.config.alpha
                kl_losses += kl_loss.item()
                loss += kl_loss

                if self.config.beta != 0:
                    buf_inputs, buf_targets, _ = self.memory_bank.get_memory_reservoir(self.config.replay_batch_size,
                                                                                        self.train_dataset.use_path,
                                                                                        self.train_dataset.transform)
                    buf_inputs, buf_targets = buf_inputs.cuda(), buf_targets.cuda()

                    buf_ce_loss = hard_loss(model(buf_inputs)["logits"], buf_targets) * self.config.beta
                    buf_ce_losses += buf_ce_loss.item()
                    loss += buf_ce_loss

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

            self.memory_bank.store_samples_reservoir(self.train_dataset.samples[indexs], logits.detach().cpu().numpy(),
                                                      targets.cpu().numpy())

        train_loss = {'all_loss': losses / len(train_loader), 'loss_clf': ce_losses / len(train_loader),
                      'loss_kl': kl_losses / len(train_loader), "buf_ce_losses": buf_ce_losses / len(train_loader)}

        return all_preds, all_targets, train_loss

    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        losses = 0.
        ce_losses, kl_losses, buf_ce_losses = 0., 0., 0.
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs)
                logits = out["logits"]
                # outputs = F.softmax(logits, dim=-1)
                preds = torch.max(logits[:, :self.cur_classes], dim=1)[1]
                ce_loss = hard_loss(logits, targets)
                # ce_loss = F.cross_entropy(logits[:, :self.cur_classes], targets)
                ce_losses += ce_loss.item()
                losses += ce_loss.item()
                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))

        test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader),
                  'loss_kl': kl_losses / len(test_loader), "buf_ce_losses": buf_ce_losses / len(test_loader)}
        return all_preds, all_targets, test_loss

    def update_memory(self, data_manager):
        pass

