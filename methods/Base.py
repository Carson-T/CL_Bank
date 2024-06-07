import os
import copy
from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from model.Inc_Net import Inc_Net
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class Base():
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.old_model = None
        self.memory_bank = None
        self.class_means = None

        self.new_classes = 0
        self.known_classes = 0
        self.cur_classes = 0

        self.cnn_overall_acc_list = []
        self.cnn_task_acc_list = np.zeros((len(config.increment_steps), len(config.increment_steps)))
        self.cnn_overall_mcr_list = []
        self.cnn_task_mcr_list = np.zeros((len(config.increment_steps), len(config.increment_steps)))
        self.nme_overall_acc_list = []
        self.nme_task_acc_list = np.zeros((len(config.increment_steps), len(config.increment_steps)))

    def update_class_num(self, task_id):
        self.new_classes = self.config.increment_steps[task_id]
        self.known_classes = sum(self.config.increment_steps[:task_id])
        self.cur_classes = self.new_classes + self.known_classes
        self.logger.info("known classes: {}, new classes: {}, current classes: {}".format(self.known_classes, self.new_classes, self.cur_classes))

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

        self.logger.info("train data num of task {}: {}".format(task_id + 1, len(self.train_dataset.samples)))
        self.logger.info("test data num of task {}: {}".format(task_id + 1, len(self.test_dataset.samples)))

    @abstractmethod
    def prepare_model(self, task_id):
        pass

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        # self.logger.info("new feature extractor requires_grad=True")
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

    @abstractmethod
    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        pass

    @abstractmethod
    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        pass

    def eval_task(self, task_id):
        hard_loss = get_loss_func(self.config)
        soft_loss = None
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
            self.logger.info("acc of task {} at each increment step (row is task, column is step): {}".format(i + 1, self.cnn_task_acc_list[i]))
        self.logger.info(
            "average of task acc at current increment step: {}".format(np.mean(self.cnn_task_acc_list[:task_id + 1, task_id])))
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
            self.logger.info("mcr of task {} at each increment step (row is task, column is step): {}".format(i + 1, self.cnn_task_mcr_list[i]))
        self.logger.info(
            "average of task mcr at current increment step: {}".format(np.mean(self.cnn_task_mcr_list[:task_id + 1, task_id])))
        self.logger.info("backward transfer: {}".format(calculate_bwf(self.cnn_task_mcr_list, task_id)))
        self.logger.info("average forgetting: {}".format(cal_avg_forgetting(self.cnn_task_mcr_list, task_id)))

        wandb.log({
            "overall/task_id": task_id + 1,
            "overall/test_overall_acc": cnn_overall_acc,
            "overall/test_overall_mcr": cnn_overall_mcr
        })

    def update_memory(self, data_manager):
        if self.memory_bank:
            new_classes_dataset = data_manager.get_dataset(indices=np.arange(self.known_classes, self.cur_classes),
                                                           source='train', mode='test')
            self.memory_bank.update_param(new_classes_dataset)
            self.memory_bank.store_examplars(new_classes_dataset, self.model)
            # examplars dataset
            if self.config.apply_nme:
                samples_memory, targets_memory = self.memory_bank.get_memory()
                examplars_dataset = MyDataset(samples_memory, targets_memory, new_classes_dataset.use_path, new_classes_dataset.transform)
                self.class_means = self.memory_bank.cal_class_means(self.model, examplars_dataset)

        else:
            pass

    def after_task(self, task_id):
        if self.config.save_checkpoint:
            checkpoint_saved_path = self.config.save_path+"/"+self.config.method+"/"+self.config.version_name
            if not os.path.exists(checkpoint_saved_path):
                os.makedirs(checkpoint_saved_path)
            save_dict = {'state_dict': self.model.state_dict(),
                         'task_id': task_id,
                         'class_means': self.class_means if hasattr(self, "class_means") else None}
            torch.save(save_dict, os.path.join(checkpoint_saved_path, f"checkpoint_task{task_id}" + ".pkl"))
            self.logger.info("model saved!")
