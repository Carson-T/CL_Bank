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


class iCarL(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.memory_bank = ReplayBank(config, logger)

        if config.increment_type != 'CIL':
            raise ValueError('iCaRL is a class incremental method!')

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = Inc_Net(self.config, self.logger)
            self.model.model_init()
        self.model.update_fc(task_id)
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        self.model = self.model.cuda()
        if self.old_model is not None:
            self.old_model = self.old_model.cuda()

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        # self.logger.info("new feature extractor requires_grad=True")
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = KD_loss
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)

        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = self.model.module

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses, kd_losses = 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs)
            logits = out["logits"]
            outputs = F.softmax(logits, dim=-1)
            preds = torch.max(outputs[:, :self.cur_classes], dim=1)[1]
            assert logits.shape[1] == self.cur_classes, "epoch train error"

            # ce loss version implementation
            ce_loss = hard_loss(logits[:, :self.cur_classes], targets)
            # ce_loss = F.cross_entropy(logits[:, :self.cur_classes], targets)
            ce_losses += ce_loss.item()
            if task_id == 0:
                loss = ce_loss
            else:
                # kd_loss = soft_loss(F.softmax(logits[:, :self.known_classes] / self.config.T, dim=1),
                #                     F.softmax(self.old_model(inputs)["logits"] / self.config.T, dim=1))
                kd_loss = soft_loss(logits[:, :self.known_classes], self.old_model(inputs)["logits"], self.config.T)
                kd_losses += kd_loss.item()
                loss = ce_loss + kd_loss
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
                if task_id == 0:
                    loss = ce_loss
                else:
                    # kd_loss = soft_loss(F.softmax(logits[:, :self.known_classes] / self.config.T, dim=1),
                    #                     F.softmax(self.old_model(inputs)["logits"] / self.config.T, dim=1))
                    kd_loss = soft_loss(logits[:, :self.known_classes], self.old_model(inputs)["logits"], self.config.T)
                    kd_losses += kd_loss.item()
                    loss = ce_loss + kd_loss
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

    def predict(self, model, test_loader, task_id):
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = self.model(inputs)
                logits = out["logits"]
                features = out["features"]
                cnn_scores, cnn_preds = torch.max(F.softmax(logits[:, :self.cur_classes], dim=-1), dim=1)
                nme_preds, nme_scores = self.memory_bank.KNN_classify(features, self.class_means)
                if idx == 0:
                    all_cnn_preds = cnn_preds
                    all_nme_preds = nme_preds
                    all_cnn_scores = cnn_scores
                    all_nme_scores = nme_scores
                    all_targets = targets
                else:
                    all_cnn_preds = torch.cat((all_cnn_preds, cnn_preds))
                    all_nme_preds = torch.cat((all_nme_preds, nme_preds))
                    all_cnn_scores = torch.cat((all_cnn_scores, cnn_scores))
                    all_nme_scores = torch.cat((all_nme_scores, nme_scores))
                    all_targets = torch.cat((all_targets, targets))

            return all_cnn_preds, all_nme_preds, all_cnn_scores, all_nme_scores, all_targets


    def eval_task(self, task_id):
        all_cnn_preds, all_nme_preds, all_cnn_scores, all_nme_scores, all_targets = self.predict(self.model, self.openset_test_loader, task_id)
        if self.is_openset_test and task_id < len(self.config.increment_steps):
            openset_target = np.ones_like(all_nme_preds)
            openset_idx = np.where(all_targets == sum(self.config.increment_steps))[0]
            openset_target[openset_idx] = 0
            nme_openset_scores = all_nme_scores.copy()

            all_targets = np.delete(all_targets, openset_idx)
            all_cnn_preds = np.delete(all_cnn_preds, openset_idx)
            all_cnn_scores = np.delete(all_cnn_scores, openset_idx)
            all_nme_preds = np.delete(all_nme_preds, openset_idx)
            all_nme_scores = np.delete(all_nme_scores, openset_idx)

            roc_auc, fpr95, ap = cal_openset_metrics(nme_openset_scores, openset_target)
            self.cnn_auc_curve.append(roc_auc)
            self.cnn_fpr95_curve.append(fpr95)
            self.cnn_AP_curve.append(ap)
            self.logger.info("=" * 100)
            self.logger.info(
                "CNN : openset AUC curve at each increment step: [\t" + ("{:2.2f}\t" * len(self.cnn_auc_curve)).format(
                    *self.cnn_auc_curve) + ']')
            self.logger.info("CNN : Average AUC of all steps: {:.2f}".format(np.mean(self.cnn_auc_curve)))
            self.logger.info(
                "CNN : openset fpr95 curve at each increment step: [\t" + ("{:2.2f}\t" * len(self.cnn_fpr95_curve)).format(
                    *self.cnn_fpr95_curve) + ']')
            self.logger.info("CNN : Average fpr95 of all steps: {:.2f}".format(np.mean(self.cnn_fpr95_curve)))
            self.logger.info(
                "CNN : openset AP curve at each increment step: [\t" + ("{:2.2f}\t" * len(self.cnn_AP_curve)).format(
                    *self.cnn_AP_curve) + ']')
            self.logger.info("CNN : Average AP of all steps: {:.2f}".format(np.mean(self.cnn_AP_curve)))
        else:
            all_cnn_preds, all_nme_preds, all_cnn_scores, all_nme_scores, all_targets = self.predict(self.model,
                                                                                                     self.test_loader,
                                                                                                     task_id)
        nme_overall_acc, nme_task_acc = calculate_acc(all_nme_preds.cpu().detach().numpy(),
                                                        all_targets.cpu().detach().numpy(), self.cur_classes,
                                                        self.config.increment_steps, cal_task_acc=True)
        cnn_overall_acc, cnn_task_acc = calculate_acc(all_cnn_preds.cpu().detach().numpy(),
                                                       all_targets.cpu().detach().numpy(), self.cur_classes,
                                                        self.config.increment_steps, cal_task_acc=True)
        self.logger.info("=" * 100)
        self.logger.info("CNN results:")
        self.cnn_overall_acc_list.append(cnn_overall_acc)
        self.logger.info("overall acc at each increment step: {}".format(self.cnn_overall_acc_list))
        self.logger.info(
            "average of all overall acc until current increment step: {}".format(np.mean(self.cnn_overall_acc_list)))
        for i in range(task_id + 1):
            self.cnn_task_acc_list[i][task_id] = cnn_task_acc[i]
            self.logger.info("acc of task {} at each increment step (row is task, column is step): {}".format(i+1, self.cnn_task_acc_list[i]))
        self.logger.info(
            "average of task acc at current increment step: {}".format(np.mean(self.cnn_task_acc_list[:task_id+1, task_id])))
        self.logger.info("backward transfer: {}".format(calculate_bwf(self.cnn_task_acc_list, task_id)))
        self.logger.info("average forgetting: {}".format(cal_avg_forgetting(self.cnn_task_acc_list, task_id)))

        # nme
        self.logger.info("=" * 100)
        self.logger.info("NME results:")
        self.nme_overall_acc_list.append(nme_overall_acc)
        self.logger.info("overall acc at each increment step: {}".format(self.nme_overall_acc_list))
        self.logger.info(
            "average of all overall acc until current increment step: {}".format(
                np.mean(self.nme_overall_acc_list)))
        for i in range(task_id + 1):
            self.nme_task_acc_list[i][task_id] = nme_task_acc[i]
            self.logger.info("acc of task {} at each increment step (row is task, column is step): {}".format(i + 1, self.nme_task_acc_list[i]))
        self.logger.info(
            "average of task acc at current increment step: {}".format(
                np.mean(self.nme_task_acc_list[:task_id+1, task_id])))
        self.logger.info("backward transfer: {}".format(calculate_bwf(self.nme_task_acc_list, task_id)))
        self.logger.info("average forgetting: {}".format(cal_avg_forgetting(self.nme_task_acc_list, task_id)))
        if not os.environ["WANDB_DISABLED"]:
            wandb.log({
                "overall/task_id": task_id + 1,
                "overall/cnn_test_overall_acc": cnn_overall_acc,
                "overall/nme_test_overall_acc": nme_overall_acc
            })

    def after_task(self, task_id):
        super().after_task(task_id)
        self.old_model = copy.deepcopy(self.model).freeze()
