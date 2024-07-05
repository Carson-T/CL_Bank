import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.Inc_Net import Inc_Net
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *
from utils.plots import *


class SLCA(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        # self.memory_bank = ReplayBank(config, logger)
        self.class_covs = None
        self.ca_logit_norm = config.ca_logit_norm
        self.num_sampled_pcls = config.num_sampled_pcls

        if config.increment_type != 'CIL':
            raise ValueError('SLCA is a class incremental method!')

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = Inc_Net(self.config, self.logger)
            self.model.model_init()
        self.model.update_fc(task_id)
        if task_id > 0:
            self.model.freeze_fe()
        # self.model.unfreeze()
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        self.model.show_trainable_params()
        self.model = self.model.cuda()

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        if task_id == 0:
            optimizer = optim.SGD([{"params": self.model.backbone.parameters(), "lr": self.config.lr * 0.01},
                                   {"params": self.model.fc.parameters(), "lr": self.config.lr}],
                                  lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = None
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)

        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = self.model.module
        if self.config.ca_epoch > 0:
            self.model.fc_backup()
            self.compute_mean_cov(data_manager, task_id)
            self.logger.info("class means and covs computed!")
            if task_id > 0:
                self.stage2_training(task_id)
                self.logger.info("stage 2 training finished!")


    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses, kd_losses = 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs)
            logits = out["logits"]
            assert logits.shape[1] == self.cur_classes, "epoch train error"
            preds = torch.max(logits[:, self.known_classes:self.cur_classes], dim=1)[1] + self.known_classes

            ce_loss = hard_loss(logits[:, self.known_classes:self.cur_classes], targets-self.known_classes)
            ce_losses += ce_loss.item()
            loss = ce_loss
            # losses += loss.item()

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
                      'loss_kd': kd_losses / len(train_loader)}

        return all_preds, all_targets, train_loss

    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        losses = 0.
        ce_losses, kd_losses = 0., 0.
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs)
                logits = out["logits"]
                outputs = F.softmax(logits, dim=-1)
                preds = torch.max(outputs[:, :self.cur_classes], dim=1)[1]
                ce_loss = hard_loss(logits[:, :self.cur_classes], targets)
                # ce_loss = F.cross_entropy(logits[:, :self.cur_classes], targets)
                ce_losses += ce_loss.item()
                loss = ce_loss
                losses += loss.item()

                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))

            test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader),
                         'loss_kd': kd_losses / len(test_loader)}
            return all_preds, all_targets, test_loss

    def compute_mean_cov(self, data_manager, task_id, check_diff=False, oracle=False):
        if hasattr(self, 'class_means') and self.class_means is not None:
            ori_classes = self.class_means.shape[0]
            assert ori_classes == self.known_classes
            cur_class_means = torch.zeros((self.cur_classes, self.model.feature_dim))
            cur_class_means[:self.known_classes] = self.class_means
            self.class_means = cur_class_means
            cur_class_cov = torch.zeros((self.cur_classes, self.model.feature_dim, self.model.feature_dim))
            cur_class_cov[:self.known_classes] = self.class_covs
            self.class_covs = cur_class_cov
        else:
            self.class_means = torch.zeros((self.cur_classes, self.model.feature_dim))
            self.class_covs = torch.zeros((self.cur_classes, self.model.feature_dim, self.model.feature_dim))
        old_class_vectors = []
        if (check_diff or oracle) and task_id > 0:
            old_class_dataset = data_manager.get_dataset(source='train', mode='test', indices=np.arange(0, self.known_classes))
            for class_idx in range(0, self.known_classes):
                vectors, _, _ = extract_vectors(self.config, self.model, old_class_dataset, class_idx)
                vectors = vectors.type(torch.float64)
                old_class_vectors.append(vectors)
                old_class_mean = torch.mean(vectors, dim=0)
                old_class_cov = torch.cov(vectors.T).detach().cpu() + torch.eye(old_class_mean.shape[-1]) * 1e-5
                if oracle:
                    self.class_means[class_idx, :] = old_class_mean
                    self.class_covs[class_idx, ...] = old_class_cov
                if check_diff:
                    log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(
                        self.class_means[class_idx, :].unsqueeze(0),
                        old_class_mean.unsqueeze(0).detach().cpu()).item())
                    self.logger.info(log_info)
        new_class_vectors = []
        new_class_dataset = data_manager.get_dataset(source='train', mode='test', indices=np.arange(self.known_classes, self.cur_classes))
        for class_idx in range(self.known_classes, self.cur_classes):
            vectors, _, _ = extract_vectors(self.config, self.model, new_class_dataset, class_idx)
            vectors = vectors.type(torch.float64)
            new_class_vectors.append(vectors)
            new_class_mean = torch.mean(vectors, dim=0)
            new_class_cov = torch.cov(vectors.T).detach().cpu() + torch.eye(new_class_mean.shape[-1]) * 1e-4
            self.class_means[class_idx, :] = new_class_mean
            self.class_covs[class_idx, ...] = new_class_cov
        t_sne(self.class_means.numpy(), old_class_vectors, new_class_vectors, task_id, self.config.run_dir+f"/{task_id}.jpg")

    def stage2_training(self, task_id):
        self.model.freeze_fe()
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.ca_lr, momentum=0.9, weight_decay=self.config.weight_decay)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer2, T_max=self.config.ca_epoch)

        self.model.eval()
        for epoch in range(self.config.ca_epoch):
            losses = 0.
            sampled_data = []
            sampled_label = []

            for c_id in range(self.cur_classes):
                t_id = c_id // self.config.increment_steps[0]
                decay = (t_id + 1) / (task_id + 1) * 0.1
                cls_mean = self.class_means[c_id].cuda() * (0.9 + decay)  # torch.from_numpy(self._class_means[c_id]).to(self._device)
                cls_cov = self.class_covs[c_id].cuda()

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_pcls = m.sample(sample_shape=(self.num_sampled_pcls,))
                sampled_data.append(sampled_data_pcls)
                sampled_label.extend([c_id] * self.num_sampled_pcls)

            inputs = torch.cat(sampled_data, dim=0).float().cuda()
            targets = torch.tensor(sampled_label).long().cuda()

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for i in range(self.cur_classes):
                inp = inputs[i * self.num_sampled_pcls:(i + 1) * self.num_sampled_pcls]
                tgt = targets[i * self.num_sampled_pcls:(i + 1) * self.num_sampled_pcls]
                outputs = self.model(inp, fc_only=True)
                logits = outputs['logits']

                if self.ca_logit_norm > 0:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(task_id + 1):
                        cur_t_size += self.config.increment_steps[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.config.increment_steps[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)

                    # norms_all = torch.norm(logits[:, :self.cur_classes], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :self.cur_classes], norms) / self.ca_logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :self.cur_classes], tgt)

                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                losses += loss.item()

            scheduler2.step()

    def after_task(self, task_id):
        super().after_task(task_id)
        self.model.fc_recall()

