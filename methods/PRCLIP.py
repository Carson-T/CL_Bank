import os
import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.CLIP_Adapter_Net import CLIP_Adapter_Net
from model.backbone import clip
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class PRCLIP(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.memory_bank = ReplayBank(config, logger)
        self.class_covs = None
        self.class_to_idx = None
        self.current_class_names = []
        self.new_class_names = []
        self.cur_text_tokens = None
        self.new_text_tokens = None
        self.prompt_template = config.prompt_template if config.prompt_template is not None else "a bad photo of a {}."

        if config.increment_type != 'CIL':
            raise ValueError('PRCLIP is a class incremental method!')

    def prepare_task_data(self, data_manager, task_id, is_train=True):
        if is_train:
            if task_id > 0 and self.memory_bank is not None:
                self.train_dataset = data_manager.get_dataset(source='train', mode='train',
                                                              indices=np.arange(self.known_classes, self.cur_classes),
                                                              appendent=self.memory_bank.get_memory())
            else:
                self.train_dataset = data_manager.get_dataset(source='train', mode='train',
                                                              indices=np.arange(self.known_classes, self.cur_classes))
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.num_workers)
            self.logger.info("train data num of task {}: {}".format(task_id + 1, len(self.train_dataset.samples)))

        self.test_dataset = data_manager.get_dataset(source='test', mode='test', indices=np.arange(0, self.cur_classes))

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                      num_workers=self.config.num_workers)
        if self.class_to_idx is None:
            self.class_to_idx = data_manager.class_to_idx
            self.idx_to_class = dict((value, key) for key, value in self.class_to_idx.items())
        self.new_class_names = [self.idx_to_class[i] for i in range(self.known_classes, self.cur_classes)]
        self.current_class_names = [self.idx_to_class[i] for i in range(0, self.cur_classes)]
        self.logger.info('Cur Task classnames: {}'.format(self.current_class_names))
        self.logger.info("test data num of task {}: {}".format(task_id + 1, len(self.test_dataset.samples)))

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = CLIP_Adapter_Net(self.config, self.logger)
            self.model.model_init()
        self.model.freeze_fe()
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        self.model.show_trainable_params()
        self.new_text_tokens = self.model.text_tokenize(self.new_class_names, self.prompt_template)
        self.cur_text_tokens = self.model.text_tokenize(self.current_class_names, self.prompt_template)
        self.model = self.model.cuda()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         self.logger.info('{} requires grad!'.format(name))

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        # self.logger.info("new feature extractor requires_grad=True")
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        # optimizer = optim.SGD([{"params": self.model.img_final_adapter.parameters(), "lr": self.config.lr}],
        #                       lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = None
        if len(self.config.device_ids.split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)
        if len(self.config.device_ids.split(",")) > 1:
            self.model = self.model.module

        # if task_id > 0:
        #     self.model.param_retention()

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses = 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs, text_tokens=self.cur_text_tokens.cuda())
            logits_per_image = out["logits"]
            # features = out["features"]
            assert logits_per_image.shape[1] == self.cur_classes, "epoch train error"

            # ce loss version implementation
            ce_loss = hard_loss(logits_per_image, targets)
            # ce_loss = F.cross_entropy(logits[:, :self.cur_classes], targets)
            ce_losses += ce_loss.item()
            loss = ce_loss
            preds = torch.max(logits_per_image, dim=1)[1]

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
                out = model(inputs, text_tokens=self.cur_text_tokens.cuda())
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

    def after_task(self, task_id):
        super().after_task(task_id)
        self.model.save_old_param()

