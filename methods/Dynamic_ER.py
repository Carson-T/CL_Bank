import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.DER_Net import DERNet
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class Dynamic_ER(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.is_finetuning = False
        self.memory_bank = ReplayBank(config, logger)

        if config.increment_type != 'CIL':
            raise ValueError('Dynamic_ER is a class incremental method!')


    def prepare_model(self, task_id):
        if self.model is None:
            self.model = DERNet()
        self.model.update(self.config, task_id)
        self.logger.info("model updated!")
        if task_id > 0:
            self.model.freeze_old_feature_extractors(task_id)
            self.logger.info("freezed feature extractors 0-{}".format(task_id-1))
        self.model = self.model.cuda()

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        self.model.change_new_feature_extractors_grad(requires_grad=True)
        self.logger.info("new feature extractor requires_grad=True")
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = None
        if len(self.config.device_ids.split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)
        if len(self.config.device_ids.split(",")) > 1:
            self.model = self.model.module

        if task_id > 0:
            ft_train_dataset = self.get_balanced_dataset(data_manager)
            ft_train_loader = DataLoader(ft_train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                         num_workers=self.config.num_workers)
            self.model.change_new_feature_extractors_grad(requires_grad=False)
            # self.model.reset_fc_parameters(self.model.fc)
            self.logger.info("new feature extractor requires_grad=False, reset fc!")
            ft_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=self.config.ft_lr, momentum=0.9)
            ft_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=ft_optimizer, milestones=self.config.ft_milestones,
                                                          gamma=self.config.ft_gamma)
            ft_hard_loss = nn.CrossEntropyLoss()
            ft_soft_loss = None
            self.train_model(ft_train_loader, self.test_loader, ft_hard_loss, ft_soft_loss, ft_optimizer, ft_scheduler,
                             task_id=task_id, epochs=self.config.ft_epochs, stage=2)



    def train_model(self, train_loader, test_loader, hard_loss, soft_loss, optimizer, scheduler, task_id, epochs, stage):
        wandb.define_metric("task " + str(task_id + 1) + "/" + "stage" + str(stage) + "/" + "epoch")
        wandb.define_metric("task " + str(task_id + 1) + "/" + "stage" + str(stage) + "/*",
                            step_metric="task" + str(task_id + 1) + "/" + "stage" + str(stage) + "/" + "epoch")

        for epoch in range(epochs):
            train_preds, train_targets, train_loss = self.epoch_train(self.model, train_loader, hard_loss, soft_loss, optimizer,
                                                                      task_id, stage)
            if scheduler is not None:
                scheduler.step()

            test_preds, test_targets, test_loss = self.epoch_test(self.model, test_loader, hard_loss, soft_loss, task_id)

            train_overall_acc, _ = calculate_acc(train_preds.cpu().detach().numpy(),
                                                                   train_targets.cpu().detach().numpy(),
                                                                   self.cur_classes, self.config.increment_steps)
            test_overall_acc, _ = calculate_acc(test_preds.cpu().detach().numpy(),
                                                                 test_targets.cpu().detach().numpy(),
                                                                 self.cur_classes, self.config.increment_steps)

            wandb.log({
                "task " + str(task_id + 1) + "/" + "stage" + str(stage) + "/" + "epoch": epoch + 1,
                "task " + str(task_id + 1) + "/" + "stage" + str(stage) + "/" + "train_overall_acc": train_overall_acc,
                "task " + str(task_id + 1) + "/" + "stage" + str(stage) + "/" + "test_overall_acc": test_overall_acc,
                "task " + str(task_id + 1) + "/" + "stage" + str(stage) + "/" + "train_loss": train_loss["all_loss"],
                "task " + str(task_id + 1) + "/" + "stage" + str(stage) + "/" + "test_loss": test_loss["all_loss"]
            })

            self.logger.info("task_id: {}, stage: {} epoch: {}/{}".format(task_id+1, stage, epoch + 1, epochs))
            self.logger.info("train_overall_acc: {:.2f}, test_overall_acc: {:.2f}".format(train_overall_acc, test_overall_acc))
            # print("train_task_acc: {}".forat(train_task_acc_list)))
            # print("test_task_acc: {}".format(test_task_acc_list))
            self.logger.info("train_losses: {}".format(train_loss))
            self.logger.info("test_losses: {}".format(test_loss))


    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id, stage):
        losses = 0.
        losses_clf = 0.
        losses_aux = 0.

        if isinstance(model, nn.DataParallel):
            model.module.feature_extractor_list[-1].train()
        else:
            model.feature_extractor_list[-1].train()

        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs)
            logits = out["logits"]
            aux_logits = out["aux_logits"]
            outputs = F.softmax(logits, dim=-1)
            _, preds = torch.max(outputs, dim=1)
            assert logits.shape[1] == self.cur_classes, "epoch train error"
            if stage == 2:
                loss_clf = hard_loss(logits / self.config.T, targets)
                loss = loss_clf
            else:
                loss_clf = hard_loss(logits, targets)
                if task_id == 0:
                    loss = loss_clf
                else:
                    aux_targets = targets.clone()
                    aux_targets = torch.where(aux_targets - self.known_classes + 1 > 0,
                                              aux_targets - self.known_classes + 1, 0)
                    loss_aux = hard_loss(aux_logits, aux_targets)
                    loss = loss_clf + loss_aux
                    losses_aux += loss_aux.item()

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
            losses_clf += loss_clf.item()

        train_loss = {'all_loss': losses / len(train_loader), 'loss_clf': losses_clf / len(train_loader),
                      'loss_aux': losses_aux / len(train_loader)}

        return all_preds, all_targets, train_loss

    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        losses = 0.0
        losses_clf = 0.0
        losses_aux = 0.0
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs)
                logits = out["logits"]
                aux_logits = out["aux_logits"]
                outputs = F.softmax(logits, dim=-1)
                confidences, preds = torch.max(outputs, dim=-1)

                loss_clf = hard_loss(logits, targets)
                if task_id == 0:
                    loss = loss_clf
                else:
                    aux_targets = targets.clone()
                    aux_targets = torch.where(aux_targets - self.known_classes + 1 > 0,
                                              aux_targets - self.known_classes + 1, 0)
                    loss_aux = hard_loss(aux_logits, aux_targets)
                    losses_aux += loss_aux.item()
                    loss = loss_clf + loss_aux
                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))

                losses += loss.item()
                losses_clf += loss_clf.item()

        test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': losses_clf / len(test_loader),
                     'loss_aux': losses_aux / len(test_loader)}

        return all_preds, all_targets, test_loss

    def get_balanced_dataset(self, data_manager):
        new_classes_dataset = data_manager.get_dataset(indices=np.arange(self.known_classes, self.cur_classes),
                                                       source='train', mode='train')
        balanced_dataset = self.memory_bank.get_balanced_data(new_classes_dataset, self.model)
        return balanced_dataset

