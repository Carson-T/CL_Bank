import os
import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from methods.Base import Base
from model.Ease_Net import EaseNet
from ReplayBank import ReplayBank
from utils.functions import *
from utils.train_utils import *


class Ease(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.use_init_ptm = config.use_init_ptm
        self.use_diagonal = config.use_diagonal
        self.recalc_sim = config.recalc_sim

        if config.increment_type != 'CIL':
            raise ValueError('EASE is a class incremental method!')

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

        self.proto_dataset = data_manager.get_dataset(indices=np.arange(self.known_classes, self.cur_classes),
                                                         source='train', mode='test')
        self.proto_loader = DataLoader(self.proto_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        self.logger.info("train data num of task {}: {}".format(task_id + 1, len(self.train_dataset.samples)))
        self.logger.info("test data num of task {}: {}".format(task_id + 1, len(self.test_dataset.samples)))

    def prepare_model(self, task_id):
        if self.model is None:
            self.model = EaseNet(self.config, self.logger)
            self.model.model_init()
        self.model.update_model(task_id)
        self.model.freeze_fe()
        self.model.show_trainable_params()
        self.model = self.model.cuda()

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
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

        self.replace_fc(self.proto_loader, task_id)



    def train_model(self, train_loader, test_loader, hard_loss, soft_loss, optimizer, scheduler, task_id, epochs):
        wandb.define_metric("task " + str(task_id + 1) + "/" + "epoch")
        wandb.define_metric("task " + str(task_id + 1) + "/*",
                            step_metric="task" + str(task_id + 1) + "/" + "epoch")

        for epoch in range(epochs):
            train_preds, train_targets, train_loss = self.epoch_train(self.model, train_loader, hard_loss, soft_loss,
                                                                      optimizer, task_id)
            if scheduler is not None:
                scheduler.step()

            # test_preds, test_targets, test_loss = self.epoch_test(self.model, test_loader, hard_loss, task_id)

            train_overall_acc, _ = calculate_acc(train_preds.cpu().detach().numpy(),
                                                                   train_targets.cpu().detach().numpy(),
                                                                   self.cur_classes, self.config.increment_steps)
            # test_overall_acc, _ = calculate_acc(test_preds.cpu().detach().numpy(),
            #                                                      test_targets.cpu().detach().numpy(),
            #                                                      self.cur_classes, self.config.increment_steps)

            wandb.log({
                "task " + str(task_id + 1) + "/" + "epoch": epoch + 1,
                "task " + str(task_id + 1) + "/" + "train_overall_acc": train_overall_acc,
                # "task " + str(task_id + 1) + "/" + "test_overall_acc": test_overall_acc,
                "task " + str(task_id + 1) + "/" + "train_loss": train_loss["all_loss"],
                # "task " + str(task_id + 1) + "/" + "test_loss": test_loss["all_loss"]
            })

            self.logger.info("task_id: {}, epoch: {}/{}".format(task_id + 1, epoch + 1, epochs))
            self.logger.info("train_overall_acc: {:.2f}".format(train_overall_acc))
            self.logger.info("train_losses: {}".format(train_loss))
            # self.logger.info("test_losses: {}".format(test_loss))


    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id):
        losses = 0.
        ce_losses = 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs, task_id=task_id, train=True)
            logits = out["logits"]
            features = out["features"]
            assert logits.shape[1] == self.new_classes, "epoch train error"
            # aux_targets = targets.clone()
            # aux_targets = torch.where(
            #     aux_targets - self.known_classes >= 0,
            #     aux_targets - self.known_classes,
            #     -1,
            # )
            # outputs = F.softmax(logits, dim=-1)

            # ce loss version implementation
            ce_loss = hard_loss(logits, targets-self.known_classes)
            # ce_loss = F.cross_entropy(logits[:, :self.cur_classes], targets)
            ce_losses += ce_loss.item()
            loss = ce_loss
            preds = torch.max(logits, dim=1)[1]+self.known_classes

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
                out = model(inputs, task_id=task_id, train=False)
                logits = out["logits"]
                features = out["features"]
                assert logits.shape[1] == self.cur_classes
                outputs = F.softmax(logits, dim=-1)
                preds = torch.max(outputs[:, :self.cur_classes], dim=1)[1]

                # ce loss
                ce_loss = hard_loss(logits[:, :self.cur_classes], targets)
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

    def replace_fc(self, train_loader_for_protonet, task_id):
        self.model.eval()

        with torch.no_grad():
            # replace proto for each adapter in the current task
            embedding_list, label_list = [], []
            for i, (inputs, targets, _) in enumerate(train_loader_for_protonet):
                inputs = inputs.cuda()
                targets = targets.cuda()
                out = self.model(inputs, task_id, train=False)
                features = out["features"]
                embedding_list.append(features.cpu())
                label_list.append(targets.cpu())

            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            for index in range(embedding_list.shape[1]//self.model.num_features):
                # only use the diagonal feature, index = -1 denotes using init PTM, index = self._cur_task denotes the last adapter's feature
                if self.use_diagonal:
                    if self.use_init_ptm and index != 0 and index != task_id+1:
                        continue
                    elif not self.use_init_ptm and index != task_id:
                        continue

                class_list = np.unique(train_loader_for_protonet.dataset.targets)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index, index*self.model.num_features:(index+1)*self.model.num_features]
                    proto = embedding.mean(0)
                    self.model.fc.weight.data[class_index, index*self.model.num_features:(index+1)*self.model.num_features] = proto
            # # self.use_exemplars is always False in ours code
            # if self.use_exemplars and self._cur_task > 0:
            #     embedding_list = []
            #     label_list = []
            #     dataset = self.data_manager.get_dataset(np.arange(0, self._known_classes), source="train", mode="test", )
            #     loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
            #     for i, batch in enumerate(loader):
            #         (_, data, label) = batch
            #         data = data.cuda()
            #         label = label.cuda()
            #         embedding = model.feature_extractor.forward_proto(data, adapt_index=self._cur_task)
            #         embedding_list.append(embedding.cpu())
            #         label_list.append(label.cpu())
            #     embedding_list = torch.cat(embedding_list, dim=0)
            #     label_list = torch.cat(label_list, dim=0)

            #     class_list = np.unique(dataset.targets)
            #     for class_index in class_list:
            #         # print('adapter index:{}, Replacing...{}'.format(self._cur_task, class_index))
            #         data_index = (label_list == class_index).nonzero().squeeze(-1)
            #         embedding = embedding_list[data_index]
            #         proto = embedding.mean(0)
            #         model.fc.weight.data[class_index, -self._network.hidden_dim:] = proto

        if self.use_diagonal:
            return

        if self.recalc_sim:
            self.solve_sim_reset(task_id)
        else:
            self.solve_similarity(task_id)


    def get_A_B_Ahat(self, start_cls, end_cls, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self.model.num_features
            end_dim = start_dim + self.model.num_features
        else:
            start_dim = task_id * self.model.num_features
            end_dim = start_dim + self.model.num_features

        # W(Ti)  i is the i-th task index, T is the cur task index, W is a T*T matrix
        A = self.model.fc.weight.data[self.known_classes:, start_dim: end_dim]
        # W(TT)
        B = self.model.fc.weight.data[self.known_classes:, -self.model.num_features:]
        # W(ii)
        A_hat = self.model.fc.weight.data[start_cls: end_cls, start_dim: end_dim]

        return A.cpu(), B.cpu(), A_hat.cpu()

    def solve_similarity(self, task_id):
        for t in range(task_id):
            if t == 0:
                start_cls = 0
                end_cls = self.config.increment_steps[t]
            else:
                start_cls = sum(self.config.increment_steps[:t])
                end_cls = start_cls + self.config.increment_steps[t]

            A, B, A_hat = self.get_A_B_Ahat(start_cls, end_cls, task_id=t)

            # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
            similarity = torch.zeros(A_hat.shape[0], A.shape[0])
            for i in range(A_hat.shape[0]):
                for j in range(A.shape[0]):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

            # softmax the similarity, it will be failed if not use it
            similarity = F.softmax(similarity, dim=1)

            # weight the combination of B(new_cls2)
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(similarity.shape[0]):
                for j in range(similarity.shape[1]):
                    B_hat[i] += similarity[i][j] * B[j]

            # B_hat(old_cls2)
            self.model.fc.weight.data[start_cls: end_cls, -self.model.num_features:] = B_hat.cuda()

    # a ver2 solve_similarity , use only above the diagonal to calculate
    def solve_sim_reset(self, task_id):
        for t in range(task_id):
            if t == 0:
                start_cls_hat = 0
                end_cls_hat = self.config.increment_steps[t]
            else:
                start_cls_hat = sum(self.config.increment_steps[:t])
                end_cls_hat = start_cls_hat + self.config.increment_steps[t]

            if self.use_init_ptm:
                range_dim = range(t + 2, task_id + 2)
            else:
                range_dim = range(t + 1, task_id + 1)
            for dim_id in range_dim:
                # print('Solve_similarity adapter:{}, {}'.format(task_id, dim_id))
                start_dim_B = dim_id * self.model.num_features
                end_dim_B = (dim_id + 1) * self.model.num_features
                start_cls = sum(self.config.increment_steps[:dim_id])
                end_cls = self.cur_classes

                # Use the element above the diagonal to calculate
                if self.use_init_ptm:
                    start_dim_A = (t + 1) * self.model.num_features
                    end_dim_A = (t + 2) * self.model.num_features
                else:
                    start_dim_A = t * self.model.num_features
                    end_dim_A = (t + 1) * self.model.num_features

                A = self.model.fc.weight.data[start_cls:end_cls, start_dim_A:end_dim_A].cpu()
                B = self.model.fc.weight.data[start_cls:end_cls, start_dim_B:end_dim_B].cpu()
                A_hat = self.model.fc.weight.data[start_cls_hat:end_cls_hat, start_dim_A:end_dim_A].cpu()

                # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
                similarity = torch.zeros(A_hat.shape[0], A.shape[0])
                for i in range(A_hat.shape[0]):
                    for j in range(A.shape[0]):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

                # softmax the similarity, it will be failed if not use it
                similarity = F.softmax(similarity, dim=1)  # dim=1, not dim=0

                # weight the combination of B(new_cls2)
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(A_hat.shape[0]):
                    for j in range(A.shape[0]):
                        B_hat[i] += similarity[i][j] * B[j]

                # B_hat(old_cls2)
                self.model.fc.weight.data[start_cls_hat: end_cls_hat, start_dim_B: end_dim_B] = B_hat.cuda()

