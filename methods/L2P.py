import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.L2P_Net import L2P_Net
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class L2P(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        if config.increment_type != 'CIL':
            raise ValueError('L2P is a class incremental method!')

    def prepare_model(self, task_id):
        if self.model is None:
            self.model = L2P_Net(self.logger)
            self.model.model_init(self.config)
        self.model.update_fc(self.config, task_id)
        if self.config.freeze_fe:
            self.model.freeze_FE()
        self.model = self.model.cuda()

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses, prompt_losses = 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs)
            logits = out["logits"]
            prompt_loss = out["prompt_loss"]
            # print(prompt_loss)
            logits[:, :self.known_classes] = -float('inf')
            # outputs = F.softmax(logits, dim=-1)
            preds = torch.max(logits[:, :self.cur_classes], dim=1)[1]
            assert logits.shape[1] == self.cur_classes, "epoch train error"
            ce_loss = hard_loss(logits, targets)
            ce_losses += ce_loss.item()
            prompt_loss = prompt_loss.mean()
            prompt_losses += prompt_loss.item()
            loss = prompt_loss+ce_loss
            # ce loss version implementation

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
                      'loss_prompt': prompt_losses / len(train_loader)}

        return all_preds, all_targets, train_loss

    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        losses = 0.
        ce_losses, prompt_losses = 0., 0.
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs)
                logits = out["logits"]
                prompt_loss = out["prompt_loss"]
                preds = torch.max(logits[:, :self.cur_classes], dim=1)[1]

                ce_loss = hard_loss(logits, targets)
                ce_losses += ce_loss.item()
                prompt_loss = prompt_loss.mean()
                prompt_losses += prompt_loss.item()
                loss = prompt_loss + ce_loss
                losses += loss.item()
                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))

            test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader),
                         'loss_prompt': prompt_losses / len(test_loader)}
            return all_preds, all_targets, test_loss
