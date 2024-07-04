import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.Dual_prompt_Net import Dual_prompt_Net
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class Dual_Prompt(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        if config.increment_type != 'CIL':
            raise ValueError('Dual_Prompt is a class incremental method!')

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = Dual_prompt_Net(self.config, self.logger)
            self.model.model_init()
        self.model.update_fc(task_id)
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        if self.config.freeze_fe:
            self.model.freeze_fe()
        self.model = self.model.cuda()

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses, prompt_losses = 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs, train=True, task_id=task_id)
            logits = out["logits"]
            prompt_loss = out["prompt_loss"]
            # print(prompt_loss)
            # logits[:, :self.known_classes] = -float('inf')
            # outputs = F.softmax(logits, dim=-1)
            assert logits.shape[1] == self.cur_classes, "epoch train error"
            preds = torch.max(logits[:, self.known_classes:self.cur_classes], dim=1)[1]+self.known_classes
            ce_loss = hard_loss(logits, targets-self.known_classes)
            ce_losses += ce_loss.item()
            prompt_loss = prompt_loss.mean()
            prompt_losses += prompt_loss.item()
            loss = prompt_loss + ce_loss
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
                out = model(inputs, train=False)
                logits = out["logits"]
                prompt_loss = out["prompt_loss"]
                # outputs = F.softmax(logits, dim=-1)
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

    def predict(self, model, test_loader, task_id):
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs, train=False)
                logits = out["logits"]
                # outputs = F.softmax(logits, dim=-1)
                scores, preds = torch.max(F.softmax(logits[:, :self.cur_classes], dim=-1), dim=1)

                if idx == 0:
                    all_preds = preds
                    all_scores = scores
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_scores = torch.cat((all_scores, scores))
                    all_targets = torch.cat((all_targets, targets))

            return all_preds, all_scores, all_targets


