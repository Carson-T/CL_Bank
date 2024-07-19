import os
import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import wandb
from tqdm import tqdm
from random import sample
from methods.Base import Base
from model.CLIP_local_fe_Net import CLIP_local_fe_Net
from model.backbone.clip import clip
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class CLIP_local_fe(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.memory_bank = ReplayBank(config, logger) if self.config.memory_size else None

        self.use_addi_desc = config.use_addi_desc
        self.desc_num = config.desc_num
        self.class_covs = None
        self.class_to_idx = None
        self.cur_class_names = []
        self.new_class_names = []
        self.cur_text_tokens = None
        self.new_text_tokens = None
        self.prompt_template = config.prompt_template if config.prompt_template is not None else "a photo of a {}."

        if config.increment_type != 'CIL':
            raise ValueError('CLIP_Adapter is a class incremental method!')

    def get_desc(self, class_names, descs):
        # descs = []
        # for idx, class_name in enumerate(class_names):
        #     assert class_name in self.all_descs, "Class_name not in prompt_json!"
        #     cls_desc = sample(self.all_descs[class_name], self.desc_num)
        #     # cls_desc_tokens = torch.cat([clip.tokenize(c) for c in cls_desc])  # [prompt_num, 77]
        #     descs.append(cls_desc)
        # # descs_tokens = torch.stack(descs_tokens, dim=0)  # [classes_num, prompt_num, 77]
        #
        # return descs

        result = []
        for a in self.config.attr_list:
            attr_values = []
            for cls in class_names:
                attr_values.append("The {} of the object in the photo is {}".format(a, descs[cls][a]))
            result.append(attr_values)

        return result

    def prepare_task_data(self, data_manager, task_id, is_train=True):
        if self.class_to_idx is None:
            self.class_to_idx = data_manager.class_to_idx
            self.all_descs = data_manager.class_descs
            self.idx_to_class = dict((value, key) for key, value in self.class_to_idx.items())
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
        self.openset_test_dataset = data_manager.get_openset_dataset(source='test', mode='test',
                                                                     known_indices=np.arange(0, self.cur_classes))
        self.openset_test_loader = DataLoader(self.openset_test_dataset, batch_size=self.config.batch_size,
                                              shuffle=False,
                                              num_workers=self.config.num_workers)

        self.new_class_names = [self.idx_to_class[i] for i in range(self.known_classes, self.cur_classes)]
        self.cur_class_names = [self.idx_to_class[i] for i in range(0, self.cur_classes)]
        if self.use_addi_desc:
            self.new_descs = self.get_desc(self.new_class_names, self.all_descs)
            self.cur_descs = self.get_desc(self.cur_class_names, self.all_descs)
        self.logger.info('Cur Task classnames: {}'.format(self.cur_class_names))
        self.logger.info("test data num of task {}: {}".format(task_id + 1, len(self.test_dataset.samples)))

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = CLIP_local_fe_Net(self.config, self.logger)
            self.model.model_init()
        self.model.update_model(task_id)
        self.model.freeze_fe()
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        self.model.show_trainable_params()

        if self.use_addi_desc:
            self.new_desc_tokens = torch.stack([clip.tokenize(i) for i in self.new_descs])
            self.cur_desc_tokens = torch.stack([clip.tokenize(i) for i in self.cur_descs])
        self.new_text_tokens = self.model.text_tokenize(self.new_class_names, self.prompt_template, descs=self.new_descs if self.use_addi_desc else None)
        self.cur_text_tokens = self.model.text_tokenize(self.cur_class_names, self.prompt_template, descs=self.cur_descs if self.use_addi_desc else None)
        self.model = self.model.cuda()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         self.logger.info('{} requires grad!'.format(name))

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        # optimizer = optim.SGD([{"params": self.model.img_adapter_list.parameters(), "lr": self.config.lr*0.01},
        #                        {"params": self.model.img_final_adapter.parameters(), "lr": self.config.lr}],
        #                       lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = None
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = self.model.module

        # new_class_dataset = data_manager.get_dataset(source='train', mode='test',
        #                                              indices=np.arange(self.known_classes, self.cur_classes))
        #
        # self.logger.info("calculate class means")
        # if self.class_means is not None:
        #     ori_classes = self.class_means.shape[0]
        #     assert ori_classes == self.known_classes
        #     cur_class_means = torch.zeros((self.cur_classes, self.model.output_dim))
        #     cur_class_means[:self.known_classes] = self.class_means
        #     self.class_means = cur_class_means
        # else:
        #     self.class_means = torch.zeros((self.cur_classes, self.model.output_dim))
        #
        # for class_idx in range(self.known_classes, self.cur_classes):
        #     vectors, _, _ = extract_vectors(self.config, self.model, new_class_dataset, class_idx)
        #     new_class_mean = torch.mean(vectors, dim=0)
        #     self.class_means[class_idx, :] = new_class_mean

    def train_model(self, train_loader, test_loader, hard_loss, soft_loss, optimizer, scheduler, task_id, epochs):
        wandb.define_metric("task " + str(task_id + 1) + "/" + "epoch")
        wandb.define_metric("task " + str(task_id + 1) + "/*",
                            step_metric="task" + str(task_id + 1) + "/" + "epoch")

        for epoch in range(epochs):
            train_preds, train_targets, train_loss = self.epoch_train(self.model, train_loader, hard_loss, soft_loss,
                                                                      optimizer,
                                                                      task_id)
            if scheduler is not None:
                scheduler.step()

            test_preds, test_targets, test_loss = self.epoch_test(self.model, test_loader, hard_loss, soft_loss,
                                                                  task_id)

            train_overall_acc, _ = calculate_acc(train_preds.cpu().detach().numpy(),
                                                                   train_targets.cpu().detach().numpy(),
                                                                   self.cur_classes, self.config.increment_steps)
            test_overall_acc, _ = calculate_acc(test_preds.cpu().detach().numpy(),
                                                                 test_targets.cpu().detach().numpy(),
                                                                 self.cur_classes, self.config.increment_steps)

            wandb.log({
                "task " + str(task_id + 1) + "/" + "epoch": epoch + 1,
                "task " + str(task_id + 1) + "/" + "train_overall_acc": train_overall_acc,
                "task " + str(task_id + 1) + "/" + "test_overall_acc": test_overall_acc,
                "task " + str(task_id + 1) + "/" + "train_loss": train_loss["all_loss"],
                "task " + str(task_id + 1) + "/" + "test_loss": test_loss["all_loss"]
            })

            self.logger.info("task_id: {}, epoch: {}/{}".format(task_id + 1, epoch + 1, epochs))
            self.logger.info(
                "train_overall_acc: {:.2f}, test_overall_acc: {:.2f}".format(train_overall_acc, test_overall_acc))
            self.logger.info("train_losses: {}".format(train_loss))
            self.logger.info("test_losses: {}".format(test_loss))

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses, local_losses, text_losses, = 0., 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs, text_tokens=self.new_text_tokens.cuda(), desc_tokens=self.new_text_tokens.cuda(), train=True, task_id=task_id)
            logits_global = out["logits"]
            ce_loss = hard_loss(logits_global, targets-self.known_classes)
            # ce_loss = hard_loss(logits_global, targets)
            ce_losses += ce_loss.item()
            loss = ce_loss

            if self.config.alpha>0:
                text_loss = out["text_loss"]
                text_losses += text_loss.item()
                loss += self.config.alpha * text_loss

            logits_local = out["logits_local"]   # 5, B, cls
            if logits_local is not None:
                local_loss = hard_loss(logits_local, targets-self.known_classes)
                local_losses += local_loss.item()
                loss += local_loss
                preds = torch.max(F.softmax(logits_global,dim=-1)+F.softmax(logits_local,dim=-1), dim=1)[1]+self.known_classes
            else:
                preds = torch.max(logits_global, dim=1)[1]+self.known_classes

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
                      "loss_local": local_losses/len(train_loader), "loss_text": text_losses/len(train_loader)}

        return all_preds, all_targets, train_loss

    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        losses = 0.
        ce_losses, local_losses, text_losses = 0., 0., 0.
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs, text_tokens=self.cur_text_tokens.cuda(), desc_tokens=self.cur_text_tokens.cuda(), task_id=task_id)
                logits_global = out["logits"]

                # preds = torch.max(logits_global, dim=1)[1]
                ce_loss = hard_loss(logits_global, targets)
                ce_losses += ce_loss.item()
                loss = ce_loss

                # text_loss = out["text_loss"]
                # text_losses += text_loss.item()
                # loss += self.config.alpha*text_loss

                logits_local = out["logits_local"]  # 5, B, cls
                if logits_local is not None:
                    local_loss = hard_loss(logits_local, targets)
                    local_losses += local_loss.item()
                    loss += local_loss
                    preds = torch.max(F.softmax(logits_global,dim=-1)+F.softmax(logits_local,dim=-1), dim=1)[1]
                else:
                    preds = torch.max(logits_global, dim=1)[1]

                # losses += loss.item()
                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))

            test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader),
                         "loss_local": local_losses/len(test_loader), "loss_text": text_losses/len(test_loader)}
            return all_preds, all_targets, test_loss

    def predict(self, model, test_loader, task_id):
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs, text_tokens=self.cur_text_tokens.cuda(), desc_tokens=self.cur_text_tokens.cuda(), task_id=task_id)
                logits_global = out["logits"]

                # qk_loss = out["qk_loss"]
                # loss += qk_loss
                # qk_losses += qk_loss.item()
                logits_local = out["logits_local"]
                if logits_local is not None:
                    scores, preds = torch.max(F.softmax(logits_global,dim=-1)+F.softmax(logits_local,dim=-1), dim=1)
                else:
                    scores, preds = torch.max(F.softmax(logits_global, dim=-1), dim=1)

                if idx == 0:
                    all_preds = preds
                    all_scores = scores
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_scores = torch.cat((all_scores, scores))
                    all_targets = torch.cat((all_targets, targets))

            return all_preds, all_scores, all_targets

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

