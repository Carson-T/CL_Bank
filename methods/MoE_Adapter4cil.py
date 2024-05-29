import os
import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.CLIP_MoE_Adapter4cil.MoE_Adapter_Net import MoE_Adapter_Net
from model.CLIP_MoE_Adapter4cil import clip
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class MoE_Adapter4cil(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.class_to_idx = None
        self.current_class_names = []
        self.new_class_names = []
        self.cur_text_tokens = None
        self.new_text_tokens = None
        self.prompt_template = config.prompt_template if config.prompt_template is not None else "a bad photo of a {}."

        if config.increment_type != 'CIL':
            raise ValueError('MoE_Adapter is a class incremental method!')

    def prepare_task_data(self, data_manager, task_id):
        if task_id > 0 and self.memory_bank is not None:
            self.train_dataset = data_manager.get_dataset(source='train', mode='train',
                                                          indices=np.arange(self.known_classes, self.cur_classes),
                                                          appendent=self.memory_bank.get_memory())
        else:
            self.train_dataset = data_manager.get_dataset(source='train', mode='train',
                                                          indices=np.arange(self.known_classes, self.cur_classes))

        self.test_dataset = data_manager.get_dataset(source='test', mode='test', indices=np.arange(0, self.cur_classes))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                       num_workers=self.config.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                      num_workers=self.config.num_workers)
        if self.class_to_idx is None:
            self.class_to_idx = data_manager.class_to_idx
            self.idx_to_class = dict((value, key) for key, value in self.class_to_idx.items())
        self.new_class_names = [self.idx_to_class[i] for i in range(self.known_classes, self.cur_classes)]
        self.current_class_names = [self.idx_to_class[i] for i in range(0, self.cur_classes)]
        self.logger.info('Cur Task classnames: {}'.format(self.current_class_names))
        self.logger.info("train data num of task {}: {}".format(task_id + 1, len(self.train_dataset.samples)))
        self.logger.info("test data num of task {}: {}".format(task_id + 1, len(self.test_dataset.samples)))

    def prepare_model(self, task_id):
        if self.model is None:
            self.model = MoE_Adapter_Net(self.config, self.logger)
        self.model.freeze_fe()
        self.new_text_tokens = self.model.text_tokenize(self.new_class_names, self.prompt_template)
        self.cur_text_tokens = self.model.text_tokenize(self.current_class_names, self.prompt_template)
        self.model = self.model.cuda()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         self.logger.info('{} requires grad!'.format(name))

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses = 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs, text_tokens=self.new_text_tokens.cuda(), train=True)
            logits_per_image = out["logits"]
            # features = out["features"]
            assert logits_per_image.shape[1] == self.new_classes, "epoch train error"

            # ce loss version implementation
            ce_loss = hard_loss(logits_per_image, targets-self.known_classes)
            # ce_loss = F.cross_entropy(logits[:, :self.cur_classes], targets)
            ce_losses += ce_loss.item()
            loss = ce_loss
            preds = torch.max(logits_per_image, dim=1)[1]+self.known_classes

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

        train_loss = {'all_loss': losses / len(train_loader), 'loss_clf': ce_losses / len(train_loader)}

        return all_preds, all_targets, train_loss

    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        losses = 0.
        ce_losses = 0.
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs, text_tokens=self.cur_text_tokens.cuda(), train=True)
                logits_per_image = out["logits"]
                # features = out["features"]
                assert logits_per_image.shape[1] == self.cur_classes, "epoch train error"
                preds = torch.max(logits_per_image, dim=1)[1]

                # ce loss version implementation
                ce_loss = hard_loss(logits_per_image, targets)
                ce_losses += ce_loss.item()
                loss = ce_loss
                losses += loss.item()
                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))

            test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader)}
            return all_preds, all_targets, test_loss

