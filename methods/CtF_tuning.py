import os
import copy
import math
import yaml
import argparse
import importlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import wandb
from tqdm import tqdm
from random import sample
from methods.Dual_Prompt import Dual_Prompt
from methods.Base import Base
from model.backbone.clip import clip
from model.Dual_prompt_Net import Dual_prompt_Net
from model.CtF_tuning_Net import CtF_tuning_Net
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


def get_coarse_config(coarse_yaml):
    yaml_args = {}
    with open(coarse_yaml, 'r') as f:
        cfg = yaml.safe_load(f)

    yaml_args.update(cfg["basic"])
    yaml_args.update(cfg["usual"])
    scheduler = yaml_args["scheduler"]
    if "options" in cfg:
        yaml_args.update(cfg["options"][scheduler])
    if "special" in cfg:
        yaml_args.update(cfg["special"])
    coarse_configs = argparse.Namespace(**yaml_args)

    return coarse_configs


class CtF_tuning(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        coarse_config = get_coarse_config(config.coarse_yaml)
        coarse_method = getattr(importlib.import_module("methods."+coarse_config.method), coarse_config.method)
        self.coarse_trainer = coarse_method(coarse_config, logger)
        if coarse_config.memory_size is not None or coarse_config.memory_per_class is not None:
            self.memory_bank = ReplayBank(config, logger)

        self.use_desc = config.use_desc
        self.desc_num = config.desc_num
        self.class_covs = None
        self.class_to_idx = None
        self.topk_acc_list = []
        self.train_class_names = []
        self.test_class_names = []
        self.train_text_tokens = None
        self.test_text_tokens = None
        self.prompt_template = config.prompt_template if config.prompt_template is not None else "a bad photo of a {}."

        if config.increment_type != 'CIL':
            raise ValueError('ctf_tuning is a class incremental method!')

    def update_class_num(self, task_id):
        self.new_classes = self.config.increment_steps[task_id]
        self.known_classes = sum(self.config.increment_steps[:task_id])
        self.cur_classes = self.new_classes + self.known_classes
        self.logger.info("known classes: {}, new classes: {}, current classes: {}".format(self.known_classes, self.new_classes, self.cur_classes))
        self.coarse_trainer.update_class_num(task_id)

    def clip_text_tokenize(self, classes_name):
        if self.use_desc:
            assert self.desc_num != 0, "Template num is zero!"
            descs_tokens = []
            for idx, class_name in enumerate(classes_name):
                assert class_name in self.all_descs, "Class_name not in prompt_json!"
                cls_desc = sample(self.all_descs[class_name], self.desc_num)
                cls_desc_tokens = torch.cat([clip.tokenize(c) for c in cls_desc])  # [prompt_num, 77]
                descs_tokens.append(cls_desc_tokens)
                # text_inputs[idx] = text_tokenize
            descs_tokens = torch.stack(descs_tokens, dim=0)  # [classes_num, prompt_num, 77]

            return descs_tokens
        else:
            texts_tokens = torch.cat([clip.tokenize(self.prompt_template.format(c)) for c in classes_name])

            return texts_tokens

    def prepare_task_data(self, data_manager, task_id, is_train=True):
        if self.class_to_idx is None:
            self.class_to_idx = data_manager.class_to_idx
            self.all_descs = data_manager.class_descs
            self.idx_to_class = dict((value, key) for key, value in self.class_to_idx.items())
        if is_train:
            if task_id > 0 and self.memory_bank is not None:
                self.train_dataset = data_manager.get_dataset(source='train', mode='train',
                                                              indices=np.arange(self.known_classes, self.cur_classes),
                                                              appendent=self.memory_bank.get_memory(), ret_clip_img=True)
                self.train_class_names = [self.idx_to_class[i] for i in range(0, self.cur_classes)]

            else:
                self.train_dataset = data_manager.get_dataset(source='train', mode='train',
                                                              indices=np.arange(self.known_classes, self.cur_classes), ret_clip_img=True)
                self.train_class_names = [self.idx_to_class[i] for i in range(self.known_classes, self.cur_classes)]

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.num_workers)
            self.logger.info("train data num of task {}: {}".format(task_id + 1, len(self.train_dataset.samples)))

        self.test_dataset = data_manager.get_dataset(source='test', mode='test', indices=np.arange(0, self.cur_classes), ret_clip_img=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                      num_workers=self.config.num_workers)
        self.logger.info("test data num of task {}: {}".format(task_id + 1, len(self.test_dataset.samples)))

        # self.new_class_names = [self.idx_to_class[i] for i in range(self.known_classes, self.cur_classes)]
        self.test_class_names = [self.idx_to_class[i] for i in range(0, self.cur_classes)]
        self.train_text_tokens = self.clip_text_tokenize(self.train_class_names)
        self.test_text_tokens = self.clip_text_tokenize(self.test_class_names)

        self.logger.info("train classnames: {}".format(self.train_class_names))

    def prepare_model(self, task_id, checkpoint=None):
        ckpt_path = self.coarse_trainer.config.save_path + "/" + self.coarse_trainer.config.method + "/" + self.coarse_trainer.config.version_name
        task_ckpt = torch.load(os.path.join(ckpt_path, f"checkpoint_task{task_id}.pkl"))
        self.coarse_trainer.prepare_model(task_id, task_ckpt)
        self.coarse_trainer.model.freeze()
        self.logger.info("coarse model prepared!")
        if self.model is None:
            self.model = CtF_tuning_Net(self.config, self.logger)
            self.model.model_init()

        self.model.freeze_fe()
        # self.model.freeze_text()
        # self.model.use_lora()

        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        self.model.show_trainable_params()
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
        if len(self.config.device_ids.split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)
        if len(self.config.device_ids.split(",")) > 1:
            self.model = self.model.module
        # if self.config.ca_epoch > 0:
        #     self.compute_mean_cov(data_manager)
        #     self.logger.info("class means and covs computed!")
        #     if task_id > 0:
        #         self.stage2_training(task_id)
        #         self.logger.info("stage 2 training finished!")

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
            test_overall_acc, _ = calculate_acc(test_preds[0].cpu().detach().numpy(),
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
        ce_losses = 0.
        model.train()
        for idx, (_, targets, _, clip_inputs) in enumerate(train_loader):
            targets, clip_inputs = targets.cuda(), clip_inputs.cuda()
            text_inputs = self.train_text_tokens.cuda()
            out = model(clip_inputs, text_tokens=text_inputs, train=True)
            logits = out["logits"]

            if self.memory_bank is not None:
                ce_loss = hard_loss(logits, targets)
                preds = torch.max(logits, dim=1)[1]
            else:
                ce_loss = hard_loss(logits, targets - self.known_classes)
                preds = torch.max(logits, dim=1)[1] + self.known_classes
            # ce_loss = F.cross_entropy(logits[:, :self.cur_classes], targets)
            ce_losses += ce_loss.item()
            loss = ce_loss
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
            for idx, (inputs, targets, _, clip_inputs) in enumerate(test_loader):
                inputs, targets, clip_inputs = inputs.cuda(), targets.cuda(), clip_inputs.cuda()
                preds_topk = self.stage1_batch_infer(inputs, task_id)
                text_inputs = torch.cat([self.test_text_tokens[i] for i in preds_topk]).cuda()  # 从template中挑选出top5对应的文本输入 # [320,77]

                out = model(clip_inputs, text_tokens=text_inputs, train=False)
                logits = out["logits"]
                # features = out["features"]
                values_top1, indices_top1 = logits.softmax(dim=-1).topk(1)  # indices_top1 的shape为[1]， 而predicts_topk的shape为[1,5]
                preds = torch.gather(preds_topk, 1, indices_top1).squeeze()

                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                    all_preds_topk = preds_topk
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))
                    all_preds_topk = torch.cat((all_preds_topk, preds_topk))

            test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader)}
            return [all_preds, all_preds_topk], all_targets, test_loss

    def eval_task(self, task_id):
        hard_loss = get_loss_func(self.config)
        soft_loss = None
        preds, cnn_all_targets, _ = self.epoch_test(self.model, self.test_loader, hard_loss, soft_loss, task_id)
        cnn_all_preds, cnn_all_preds_topk = preds[0].cpu().detach().numpy(), preds[1].cpu().detach().numpy()
        cnn_all_targets = cnn_all_targets.cpu().detach().numpy()
        topk_acc = round(np.array([cnn_all_targets[i] in cnn_all_preds_topk[i] for i in range(len(cnn_all_targets))]).sum() * 100/len(cnn_all_targets), 2)
        cnn_overall_acc, cnn_task_acc = calculate_acc(cnn_all_preds,
                                                        cnn_all_targets, self.cur_classes,
                                                        self.config.increment_steps, cal_task_acc=True)
        self.topk_acc_list.append(topk_acc)
        self.cnn_overall_acc_list.append(cnn_overall_acc)
        self.logger.info("=" * 100)
        self.logger.info("topk acc: {}".format(self.topk_acc_list))
        self.logger.info("CNN ACC results:")
        self.logger.info("overall acc at each increment step: {}".format(self.cnn_overall_acc_list))
        self.logger.info(
            "average of all overall acc until current increment step: {}".format(np.mean(self.cnn_overall_acc_list)))
        for i in range(task_id + 1):
            self.cnn_task_acc_list[i][task_id] = cnn_task_acc[i]
            self.logger.info("acc of task {} at each increment step (row is task, column is step): {}".format(i + 1, self.cnn_task_acc_list[i]))
        self.logger.info(
            "average of task acc at current increment step: {}".format(np.mean(self.cnn_task_acc_list[:task_id + 1, task_id])))
        self.logger.info("backward transfer: {}".format(calculate_bwf(self.cnn_task_acc_list, task_id)))
        self.logger.info("average forgetting: {}".format(cal_avg_forgetting(self.cnn_task_acc_list, task_id)))

        cnn_overall_mcr, cnn_task_mcr = cal_mean_class_recall(cnn_all_preds,
                                                                cnn_all_targets,
                                                                self.cur_classes,
                                                                self.config.increment_steps, cal_task_mcr=True)

        self.cnn_overall_mcr_list.append(cnn_overall_mcr)
        self.logger.info("=" * 100)
        self.logger.info("CNN MCR results:")
        self.logger.info(
            "overall mcr at each increment step: {}".format(self.cnn_overall_mcr_list))
        self.logger.info(
            "average of all overall mcr until current increment step: {}".format(np.mean(
                self.cnn_overall_mcr_list)))
        for i in range(task_id + 1):
            self.cnn_task_mcr_list[i][task_id] = cnn_task_mcr[i]
            self.logger.info("mcr of task {} at each increment step (row is task, column is step): {}".format(i + 1, self.cnn_task_mcr_list[i]))
        self.logger.info(
            "average of task mcr at current increment step: {}".format(np.mean(self.cnn_task_mcr_list[:task_id + 1, task_id])))
        self.logger.info("backward transfer: {}".format(calculate_bwf(self.cnn_task_mcr_list, task_id)))
        self.logger.info("average forgetting: {}".format(cal_avg_forgetting(self.cnn_task_mcr_list, task_id)))
        if not os.environ["WANDB_DISABLED"]:
            wandb.log({
                "overall/task_id": task_id + 1,
                "overall/test_overall_acc": cnn_overall_acc,
                "overall/test_overall_mcr": cnn_overall_mcr
            })

    def stage1_batch_infer(self, input_batch, task_id):
        out = self.coarse_trainer.model(input_batch, train=False, task_id=task_id)
        logits = out["logits"]
        features = out["features"]
        if self.coarse_trainer.memory_bank is not None and self.coarse_trainer.config.apply_nme:  # 例如icarl等nme方法时，使用nme的top-k
            _, dists = self.memory_bank.KNN_classify(features, self.coarse_trainer.class_means, ret_logits=True)
            predicts_topk = torch.topk(dists, k=self.config.topk, dim=1, largest=False, sorted=True)[1]  # [bs, topk]
            # predicts_topk = torch.topk(dist, k=self.topk, dim=1)[1] # [bs, topk]
        else:
            predicts_topk = torch.topk(logits[:, :self.cur_classes], k=self.config.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

        return predicts_topk

    def update_memory(self, data_manager):
        if self.memory_bank:
            new_classes_dataset = data_manager.get_dataset(indices=np.arange(self.known_classes, self.cur_classes),
                                                           source='train', mode='test')
            self.memory_bank.update_param(new_classes_dataset)
            self.memory_bank.store_examplars(new_classes_dataset, self.coarse_trainer.model)

        else:
            pass