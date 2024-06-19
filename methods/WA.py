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


class WA(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.memory_bank = ReplayBank(config, logger)
        self.kd_lambda = None

        if config.increment_type != 'CIL':
            raise ValueError('WA is a class incremental method!')

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = Inc_Net(self.config, self.logger)
            self.model.model_init()
        self.model.update_fc(task_id)
        self.model = self.model.cuda()
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        if self.old_model is not None:
            self.old_model = self.old_model.cuda()

        self.kd_lambda = self.known_classes / self.cur_classes

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        # self.logger.info("new feature extractor requires_grad=True")
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = KD_loss
        if len(self.config.device_ids.split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)
        if len(self.config.device_ids.split(",")) > 1:
            self.model = self.model.module

        if task_id > 0:
            self.model.weight_align(self.new_classes)


    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses, kd_losses = 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs)
            logits = out["logits"]
            # outputs = F.softmax(logits, dim=-1)
            preds = torch.max(logits[:, :self.cur_classes], dim=1)[1]
            assert logits.shape[1] == self.cur_classes, "epoch train error"

            # ce loss version implementation
            ce_loss = hard_loss(logits[:, :self.cur_classes], targets) * (1 - self.kd_lambda)
            ce_losses += ce_loss.item()
            loss = ce_loss
            if self.old_model is not None:
                kd_loss = soft_loss(logits[:, :self.known_classes], self.old_model(inputs)["logits"],
                                    self.config.T) * self.kd_lambda
                loss += kd_loss
                kd_losses += kd_loss.item()

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
                # outputs = F.softmax(logits, dim=-1)
                preds = torch.max(logits[:, :self.cur_classes], dim=1)[1]

                ce_loss = hard_loss(logits[:, :self.cur_classes], targets) * (1 - self.kd_lambda)
                ce_losses += ce_loss.item()
                loss = ce_loss
                if self.old_model is not None:
                    kd_loss = soft_loss(logits[:, :self.known_classes], self.old_model(inputs)["logits"],
                                        self.config.T) * self.kd_lambda
                    loss += kd_loss
                    kd_losses += kd_loss.item()
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

    def eval_task(self, task_id):
        hard_loss = get_loss_func(self.config)
        soft_loss = KD_loss
        cnn_all_preds, cnn_all_targets, _ = self.epoch_test(self.model, self.test_loader, hard_loss, soft_loss, task_id)

        cnn_overall_acc, cnn_task_acc = calculate_acc(cnn_all_preds.cpu().detach().numpy(),
                                                      cnn_all_targets.cpu().detach().numpy(), self.cur_classes,
                                                      self.config.increment_steps, cal_task_acc=True)

        self.cnn_overall_acc_list.append(cnn_overall_acc)
        self.logger.info("=" * 100)
        self.logger.info("CNN ACC results:")
        self.logger.info("overall acc at each increment step: {}".format(self.cnn_overall_acc_list))
        self.logger.info(
            "average of all overall acc until current increment step: {}".format(np.mean(self.cnn_overall_acc_list)))
        for i in range(task_id + 1):
            self.cnn_task_acc_list[i][task_id] = cnn_task_acc[i]
            self.logger.info("acc of task {} at each increment step (row is task, column is step): {}".format(i + 1,
                                                                                                              self.cnn_task_acc_list[
                                                                                                                  i]))
        self.logger.info(
            "average of task acc at current increment step: {}".format(
                np.mean(self.cnn_task_acc_list[:task_id + 1, task_id])))
        self.logger.info("backward transfer: {}".format(calculate_bwf(self.cnn_task_acc_list, task_id)))
        self.logger.info("average forgetting: {}".format(cal_avg_forgetting(self.cnn_task_acc_list, task_id)))

        cnn_overall_mcr, cnn_task_mcr = cal_mean_class_recall(cnn_all_preds.cpu().detach().numpy(),
                                                              cnn_all_targets.cpu().detach().numpy(),
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
            self.logger.info("mcr of task {} at each increment step (row is task, column is step): {}".format(i + 1,
                                                                                                              self.cnn_task_mcr_list[
                                                                                                                  i]))
        self.logger.info(
            "average of task mcr at current increment step: {}".format(
                np.mean(self.cnn_task_mcr_list[:task_id + 1, task_id])))
        self.logger.info("backward transfer: {}".format(calculate_bwf(self.cnn_task_mcr_list, task_id)))
        self.logger.info("average forgetting: {}".format(cal_avg_forgetting(self.cnn_task_mcr_list, task_id)))
        if not os.environ["WANDB_DISABLED"]:
            wandb.log({
                "overall/task_id": task_id + 1,
                "overall/test_overall_acc": cnn_overall_acc,
                "overall/test_overall_mcr": cnn_overall_mcr
            })

    def after_task(self, task_id):
        super().after_task(task_id)
        self.old_model = copy.deepcopy(self.model).freeze()

