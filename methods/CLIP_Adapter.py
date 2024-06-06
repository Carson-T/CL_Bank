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


class CLIP_Adapter(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.class_covs = None
        self.class_to_idx = None
        self.current_class_names = []
        self.new_class_names = []
        self.cur_text_tokens = None
        self.new_text_tokens = None
        self.prompt_template = config.prompt_template if config.prompt_template is not None else "a bad photo of a {}."

        if config.increment_type != 'CIL':
            raise ValueError('CLIP_Adapter is a class incremental method!')

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
            self.model = CLIP_Adapter_Net(self.config, self.logger)
            self.model.model_init()
        self.model.freeze_fe()
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
        if len(self.config.device_ids.split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        # self.logger.info("new feature extractor requires_grad=True")
        # optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        optimizer = optim.SGD([{"params": self.model.img_adapter_list.parameters(), "lr": self.config.lr*0.01},
                               {"params": self.model.img_final_adapter.parameters(), "lr": self.config.lr}],
                              lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = None
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)
        if self.config.ca_epoch > 0:
            self.compute_mean_cov(data_manager)
            self.logger.info("class means and covs computed!")
            if task_id > 0:
                self.stage2_training(task_id)
                self.logger.info("stage 2 training finished!")
        if len(self.config.device_ids.split(",")) > 1:
            self.model = self.model.module

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses = 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs, text_tokens=self.new_text_tokens.cuda())
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

    def compute_mean_cov(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, 'class_means') and self.class_means is not None and not check_diff:
            ori_classes = self.class_means.shape[0]
            assert ori_classes == self.known_classes
            cur_class_means = torch.zeros((self.cur_classes, self.model.output_dim))
            cur_class_means[:self.known_classes] = self.class_means
            self.class_means = cur_class_means
            cur_class_cov = torch.zeros((self.cur_classes, self.model.output_dim, self.model.output_dim))
            cur_class_cov[:self.known_classes] = self.class_covs
            self.class_covs = cur_class_cov
        elif not check_diff:
            self.class_means = torch.zeros((self.cur_classes, self.model.output_dim))
            self.class_covs = torch.zeros((self.cur_classes, self.model.output_dim, self.model.output_dim))

        if check_diff or oracle:
            old_class_dataset = data_manager.get_dataset(source='train', mode='test', indices=np.arange(0, self.known_classes))
            for class_idx in range(0, self.known_classes):
                vectors, _, _ = extract_vectors(self.config, self.model, old_class_dataset, class_idx)
                vectors = vectors.type(torch.float64)
                old_class_mean = torch.mean(vectors, dim=0)
                old_class_cov = torch.cov(vectors.T).detach().cpu() + torch.eye(old_class_mean.shape[-1]) * 1e-5
                if oracle:
                    self.class_means[class_idx, :] = old_class_mean
                    self.class_covs[class_idx, ...] = old_class_cov
                if check_diff:
                    log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(
                        self.class_means[class_idx, :].unsqueeze(0),
                        old_class_mean.unsqueeze(0)).item())
                    self.logger.info(log_info)

        new_class_dataset = data_manager.get_dataset(source='train', mode='test', indices=np.arange(self.known_classes, self.cur_classes))
        for class_idx in range(self.known_classes, self.cur_classes):
            vectors, _, _ = extract_vectors(self.config, self.model, new_class_dataset, class_idx)
            vectors = vectors.type(torch.float64)
            new_class_mean = torch.mean(vectors, dim=0)
            new_class_cov = torch.cov(vectors.T).detach().cpu() + torch.eye(new_class_mean.shape[-1]) * 1e-4
            self.class_means[class_idx, :] = new_class_mean
            self.class_covs[class_idx, ...] = new_class_cov

    def stage2_training(self, task_id):
        self.model.freeze_adapter()
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.ca_lr, momentum=0.9, weight_decay=self.config.weight_decay)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer2, T_max=self.config.ca_epoch)

        self.model.eval()
        for epoch in range(self.config.ca_epoch):
            losses = 0.
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.config.num_sampled_pcls

            for c_id in range(self.cur_classes):
                t_id = c_id // self.config.increment_steps[0]
                decay = (t_id + 1) / (task_id + 1) * 0.1
                cls_mean = self.class_means[c_id].cuda() * (0.9 + decay)  # torch.from_numpy(self._class_means[c_id]).to(self._device)
                cls_cov = self.class_covs[c_id].cuda()

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_pcls = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_pcls)
                sampled_label.extend([c_id] * num_sampled_pcls)

            inputs = torch.cat(sampled_data, dim=0).float().cuda()
            targets = torch.tensor(sampled_label).long().cuda()

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for i in range(self.cur_classes):
                inp = inputs[i * num_sampled_pcls:(i + 1) * num_sampled_pcls]
                tgt = targets[i * num_sampled_pcls:(i + 1) * num_sampled_pcls]
                outputs = self.model.forward_with_vectors(inp, self.cur_text_tokens.cuda())
                logits = outputs['logits']

                # if self.logit_norm is not None:
                #     per_task_norm = []
                #     prev_t_size = 0
                #     cur_t_size = 0
                #     for _ti in range(self._cur_task + 1):
                #         cur_t_size += self.task_sizes[_ti]
                #         temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                #         per_task_norm.append(temp_norm)
                #         prev_t_size += self.task_sizes[_ti]
                #     per_task_norm = torch.cat(per_task_norm, dim=-1)
                #     norms = per_task_norm.mean(dim=-1, keepdim=True)
                #
                #     norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                #     decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                #     loss = F.cross_entropy(decoupled_logits, tgt)

                loss = F.cross_entropy(logits[:, :self.cur_classes], tgt)

                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                losses += loss.item()

            scheduler2.step()

