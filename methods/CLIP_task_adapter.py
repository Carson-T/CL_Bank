import os
import copy
import math
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.distributions.multivariate_normal import MultivariateNormal
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import wandb
from tqdm import tqdm
from random import sample
from methods.Base import Base
from model.CLIP_task_adapter_Net import CLIP_task_adapter_Net
from model.backbone.clip import clip
from model.backbone.VAE import VariationalAutoEncoderModel
from ReplayBank import ReplayBank
from utils.GMM import MixtureOfGaussiansModel
from utils.functions import *
from utils.train_utils import *


class CLIP_task_adapter(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.memory_bank = ReplayBank(config, logger) if self.config.memory_size else None

        self.use_addi_desc = config.use_addi_desc
        self.desc_num = config.desc_num
        self.class_covs = None
        self.task_means = None
        self.task_covs = None
        self.gmm_list = []
        self.vae_list = []
        self.class_cluster_ratios = []
        self.class_cluster_means = []
        self.class_cluster_covs = []
        self.ood_data = None
        self.class_to_idx = None
        self.cur_class_names = []
        self.new_class_names = []
        self.cur_text_tokens = None
        self.new_text_tokens = None
        self.prompt_template = config.prompt_template if config.prompt_template is not None else "a photo of a {}."

        if config.increment_type != 'CIL':
            raise ValueError('CLIP_Adapter is a class incremental method!')

    def get_desc(self, class_names, all_descs, desc_num):
        descs = []
        for idx, class_name in enumerate(class_names):
            assert class_name in all_descs, "Class_name not in prompt_json!"
            # cls_desc = max(all_descs[class_name], key=len)
            cls_desc = all_descs[class_name]
            # cls_desc = sample(all_descs[class_name], desc_num)
            descs.append(cls_desc)
        # descs_tokens = torch.stack(descs_tokens, dim=0)  # [classes_num, prompt_num, 77]

        return descs

        # result = []
        # for a in self.config.attr_list:
        #     attr_values = []
        #     for cls in class_names:
        #         attr_values.append("The {} of the object in the photo is {}".format(a, descs[cls][a]))
        #     result.append(attr_values)

        # return result

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
        # if self.use_addi_desc:
        #     self.new_descs = self.get_desc(self.new_class_names, self.all_descs, self.desc_num)
        #     self.cur_descs = self.get_desc(self.cur_class_names, self.all_descs, self.desc_num)
        self.logger.info('Cur Task classnames: {}'.format(self.cur_class_names))
        self.logger.info("test data num of task {}: {}".format(task_id + 1, len(self.test_dataset.samples)))

        # new_counts = collections.Counter(self.train_dataset.targets)
        # new_counts = dict(sorted(dict(new_counts).items(), key=lambda x: x[0]))
        # self.new_class_weight = (sum(list(new_counts.values())) / torch.tensor(list(new_counts.values()))).cuda()
        # cur_counts = collections.Counter(self.test_dataset.targets)
        # cur_counts = dict(sorted(dict(cur_counts).items(), key=lambda x: x[0]))
        # self.cur_class_weight = (sum(list(cur_counts.values())) / torch.tensor(list(cur_counts.values()))).cuda()
        self.new_class_weight = None
        self.cur_class_weight = None

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = CLIP_task_adapter_Net(self.config, self.logger)
            self.model.model_init()
        # self.model.unfreeze_prompt()
        self.model.update_model(task_id)
        self.model.freeze_fe()
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            if checkpoint["class_covs"] is not None:
                self.class_means = checkpoint["class_covs"]
            self.logger.info("checkpoint loaded!")
        self.model.show_trainable_params()

        if self.use_addi_desc:
            self.new_text_tokens, self.cur_text_tokens = self.model.select_desc(self.new_class_names, old_desc_tokens=self.cur_text_tokens, all_descs=self.all_descs, class_means=self.class_means)
        else:
            self.new_text_tokens = self.model.text_tokenize(self.new_class_names, self.prompt_template, descs=None)
            self.cur_text_tokens = self.model.text_tokenize(self.cur_class_names, self.prompt_template, descs=None)
            # self.cur_desc_tokens = torch.stack([clip.tokenize(i) for i in self.cur_descs])
            # self.new_desc_tokens = torch.stack([clip.tokenize(i) for i in self.new_descs])
        self.model = self.model.cuda()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         self.logger.info('{} requires grad!'.format(name))

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")
        optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.config)
        # optimizer = optim.AdamW([{"params": self.model.cur_img_adapter.parameters(), "lr": self.config.lr},
        #                        {"params": self.model.img_final_adapter.parameters(), "lr": 0.005},
        #                       {"params": self.model.ood_detector.parameters(), "lr": 0.005}], lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = None
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = self.model.module

        # self.vae_training(data_manager, task_id)
        # self.task_distribution(data_manager, task_id)
        if self.config.ca_epoch > 0:
            self.compute_mean_cov(data_manager)
            self.logger.info("class means and covs computed!")
            # self.get_ood_vectors(self.train_loader, task_id)
            # self.logger.info("ood vectors stored!")
            if task_id > 0:
                self.stage2_training(task_id)
                self.logger.info("stage 2 training finished!")

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
            train_overall_acc, _ = calculate_acc(train_preds.cpu().detach().numpy(),
                                                                   train_targets.cpu().detach().numpy(),
                                                                   self.cur_classes, self.config.increment_steps)
            if epoch > epochs-3:
                test_preds, test_targets, test_loss = self.epoch_test(self.model, test_loader, hard_loss, soft_loss,
                                                                      task_id)
                test_overall_acc, _ = calculate_acc(test_preds.cpu().detach().numpy(),
                                                                     test_targets.cpu().detach().numpy(),
                                                                     self.cur_classes, self.config.increment_steps)
            else:
                test_overall_acc = 0
                test_loss = {"all_loss": 0}

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
        ce_losses, ood_losses, text_losses, = 0., 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # ood_vectors = self.ood_data[torch.randint(len(self.ood_data), (self.config.batch_size,))].cuda()
            # ood_targets = torch.tensor([0] * inputs.shape[0] + [1] * inputs.shape[0]).long().cuda()
            with autocast():
                out = model(inputs, text_tokens=self.new_text_tokens.cuda(), train=True, task_id=task_id)
                logits_global = out["logits"]
                ood_logits = out["ood_logits"]
                # ce_loss = hard_loss(logits_global, targets-self.known_classes)
                ce_loss = F.cross_entropy(logits_global, targets-self.known_classes, weight=self.new_class_weight)
                # ce_loss = hard_loss(logits_global, targets)
                ce_losses += ce_loss.item()
                loss = ce_loss

                # if ood_logits is not None:
                #     ood_loss = hard_loss(ood_logits, ood_targets)
                #     ood_losses += ood_loss.item()
                #     loss += ood_loss
            preds = torch.max(logits_global, dim=1)[1]+self.known_classes
            # preds = torch.max(logits_global[:, self.known_classes:self.cur_classes], dim=1)[1]+self.known_classes

            if idx == 0:
                all_preds = preds
                all_targets = targets
            else:
                all_preds = torch.cat((all_preds, preds))
                all_targets = torch.cat((all_targets, targets))

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # losses += loss.item()

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
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
                out = model(inputs, text_tokens=self.cur_text_tokens.cuda(), train=False, task_id=task_id, labels=targets)
                logits = out["logits"]

                # preds = torch.max(logits_global, dim=1)[1]
                ce_loss = hard_loss(logits, targets)
                ce_losses += ce_loss.item()
                loss = ce_loss

                preds = torch.max(logits, dim=-1)[1]

                pred_tids = out["task_id"].squeeze(-1)
                delta_entropy, entropy1 = out["delta_entropy"]
                # indices = (torch.arange(self.config.increment_steps[0]).reshape(1, -1)).cuda() + tids.unsqueeze(
                #     -1) * self.config.increment_steps[0]
                # logits1 = torch.gather(logits, dim=-1, index=indices)
                # preds = torch.max(logits1, dim=-1)[1] + tids * self.config.increment_steps[0]

                losses += loss.item()
                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                    all_pred_tids = pred_tids
                    all_delta_entropy = delta_entropy
                    all_entropy1 = entropy1
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))
                    all_pred_tids = torch.cat((all_pred_tids, pred_tids))
                    all_delta_entropy = torch.cat((all_delta_entropy, delta_entropy))
                    all_entropy1 = torch.cat((all_entropy1, entropy1))
            true_tids = torch.div(all_targets, self.config.increment_steps[0], rounding_mode="floor")
            self.logger.info("task id acc: {}".format((all_pred_tids == true_tids).sum().item() / len(all_targets)))
            all_delta_entropy = (all_delta_entropy < 0)
            # if task_id == 1:
            #     print("all_delta_entropy:", all_delta_entropy[all_pred_tids != true_tids])
            #     print("all_entropy1:", all_entropy1[all_pred_tids != true_tids])

            # all_delta_entropy = (all_delta_entropy == torch.min(all_delta_entropy, dim=-1, keepdim=True)[0])
            for j in range(task_id+1):
                sum = torch.sum(all_delta_entropy[true_tids == j], dim=0)
                self.logger.info("delta entropy of task {} data: {}".format(j, sum))
                # sum = torch.mean(all_entropy1[true_tids == j], dim=0)
                # self.logger.info("mean entropy1 of task {} data: {}".format(j, sum))
            test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader)}
            return all_preds, all_targets, test_loss

    def predict(self, model, test_loader, task_id):
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs, text_tokens=self.cur_text_tokens.cuda(), train=False, task_id=task_id, labels=targets, task_mean=self.task_means, task_cov=self.task_covs, stage2=True)  # vae_list=self.vae_list
                logits = out["logits"]

                scores, preds = torch.max(F.softmax(logits, dim=-1), dim=-1)

                pred_tids = out["task_id"].squeeze(-1)
                delta_entropy, entropy1 = out["delta_entropy"]
                # indices = (torch.arange(self.config.increment_steps[0]).reshape(1, -1)).cuda() + tids.unsqueeze(
                #     -1) * self.config.increment_steps[0]
                # logits1 = torch.gather(logits, dim=-1, index=indices)
                # scores, preds = torch.max(logits1, dim=-1)
                # preds = preds + tids * self.config.increment_steps[0]
                if idx == 0:
                    all_preds = preds
                    all_scores = scores
                    all_targets = targets
                    all_pred_tids = pred_tids
                    all_delta_entropy = delta_entropy
                    all_entropy1 = entropy1
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_scores = torch.cat((all_scores, scores))
                    all_targets = torch.cat((all_targets, targets))
                    all_pred_tids = torch.cat((all_pred_tids, pred_tids))
                    all_delta_entropy = torch.cat((all_delta_entropy, delta_entropy))
                    all_entropy1 = torch.cat((all_entropy1, entropy1))
            true_tids = torch.div(all_targets, self.config.increment_steps[0], rounding_mode="floor")
            self.logger.info("task id acc: {}".format((all_pred_tids == true_tids).sum().item() / len(all_targets)))
            all_delta_entropy = (all_delta_entropy < 0)
            # if task_id == 1:
            #     print("all_delta_entropy:", all_delta_entropy[all_pred_tids != true_tids])
            #     print("all_entropy1:", all_entropy1[all_pred_tids != true_tids])

            # all_delta_entropy = (all_delta_entropy == torch.min(all_delta_entropy, dim=-1, keepdim=True)[0])
            for j in range(task_id + 1):
                sum = torch.sum(all_delta_entropy[true_tids == j], dim=0)
                self.logger.info("delta entropy of task {} data: {}".format(j, sum))
                # sum = torch.mean(all_entropy1[true_tids == j], dim=0)
                # self.logger.info("mean entropy1 of task {} data: {}".format(j, sum))

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
            if self.config.n_components > 0:
                m = MixtureOfGaussiansModel(vectors.shape[-1], n_components=self.config.n_components).cuda()
                m.fit(vectors)
                self.gmm_list.append(m)
            elif self.config.cluster_num > 0:
                # cluster_idx, class_centers = kmeans(X=vectors, num_clusters=cluster_num, distance="cosine", device=vectors.device)

                # best_cluster_num = cluster_num
                # km = KMeans(n_clusters=cluster_num, init="random", n_init=10, random_state=self.config.random_seed)
                # cluster_idx = km.fit_predict(vectors.detach().cpu().numpy())

                vectors1 = vectors.detach().cpu().numpy()
                gmms = [GaussianMixture(n_components=i, init_params="kmeans", n_init=1, random_state=self.config.random_seed).fit(vectors1) for i in range(1, self.config.cluster_num)]
                bics = [m.bic(vectors1) for m in gmms]
                best_cluster_num = np.argmin(bics)+1
                print(bics, best_cluster_num)
                cluster_idx = gmms[best_cluster_num-1].predict(vectors1)

                cluster_idx = torch.from_numpy(cluster_idx).cuda()
                cluster_ratio = []
                all_cluster_means = []
                all_cluster_covs = []
                for cluster in range(best_cluster_num):
                    cluster_vectors = vectors[(cluster_idx == cluster).nonzero(as_tuple=True)[0]]
                    if cluster_vectors.shape[0] > 1:
                        cluster_mean = torch.mean(cluster_vectors, dim=0)
                        cluster_cov = torch.cov(cluster_vectors.T).detach().cpu() + torch.eye(cluster_mean.shape[-1]) * 1e-4
                        all_cluster_means.append(cluster_mean)
                        all_cluster_covs.append(cluster_cov)
                        cluster_ratio.append(cluster_vectors.shape[0] / vectors.shape[0])
                self.class_cluster_means.append(torch.stack(all_cluster_means, dim=0))
                self.class_cluster_covs.append(torch.stack(all_cluster_covs, dim=0))
                self.class_cluster_ratios.append(cluster_ratio)

            new_class_mean = torch.mean(vectors, dim=0)
            new_class_cov = torch.cov(vectors.T).detach().cpu() + torch.eye(new_class_mean.shape[-1]) * 1e-4
            self.class_means[class_idx, :] = new_class_mean
            self.class_covs[class_idx, ...] = new_class_cov

    def get_ood_vectors(self, train_loader, task_id):
        all_ood_vectors = torch.tensor([])
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            ood_vectors = self.model.get_ood_vectors(inputs, task_id)
            ood_vectors = ood_vectors.detach().cpu()
            all_ood_vectors = torch.cat([all_ood_vectors, ood_vectors], dim=0)

        if self.ood_data is not None:
            self.ood_data = torch.cat([self.ood_data, all_ood_vectors], dim=0)
        else:
            self.ood_data = all_ood_vectors

    def task_distribution(self, data_manager, task_id):
        # self.model.freeze_adapter()
        new_class_dataset = data_manager.get_dataset(source='train', mode='test',
                                                     indices=np.arange(self.known_classes, self.cur_classes))
        new_class_dataloader = DataLoader(new_class_dataset, batch_size=64, shuffle=False, num_workers=self.config.num_workers)
        self.model.eval()
        all_vectors = []
        # all_logits = []
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(new_class_dataloader):
                output = self.model(inputs.cuda())
                all_vectors.append(output["features"])
        all_vectors = torch.cat(all_vectors, dim=0).type(torch.float32)

        mean = all_vectors.mean(0)
        cov = torch.cov(all_vectors.T) + torch.eye(mean.shape[-1]).cuda() * 1e-4
        if self.task_means is None:
            self.task_means = torch.tensor([]).cuda()
            self.task_covs = torch.tensor([]).cuda()
        self.task_means = torch.cat([self.task_means, mean.unsqueeze(0)], dim=0)
        self.task_covs = torch.cat([self.task_covs, cov.unsqueeze(0)], dim=0)

        # vae = VariationalAutoEncoderModel(input_dim=all_vectors.shape[-1], hidden_dim=512, latent_dim=256, lr=0.0002, n_iters=100).cuda()
        # vae.fit(all_vectors)
        # self.vae_list.append(vae)


    def stage2_training(self, task_id):
        self.model.freeze_adapter()
        self.model.show_trainable_params()
        optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.ca_lr, weight_decay=self.config.weight_decay)
        # optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.ca_lr, momentum=0.9, weight_decay=self.config.weight_decay)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer2, T_max=self.config.ca_epoch)

        self.model.eval()
        for epoch in range(self.config.ca_epoch):
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.config.num_sampled_pcls

            for c_id in range(self.cur_classes):
                # t_id = c_id // self.config.increment_steps[0]
                # decay = (t_id + 1) / (task_id + 1) * 0.1

                if self.config.n_components > 0:
                    m = self.gmm_list[c_id]
                    sampled_data_pcls = m.sample(num_sampled_pcls)
                    sampled_data.append(sampled_data_pcls)
                    sampled_label.extend([c_id] * num_sampled_pcls)
                elif self.config.cluster_num > 0:
                    cls_cluster_means = self.class_cluster_means[c_id].cuda()
                    cls_cluster_covs = self.class_cluster_covs[c_id].cuda()
                    cls_cluster_ratio = self.class_cluster_ratios[c_id]
                    sampled_nums = [int(num_sampled_pcls*i) for i in cls_cluster_ratio]
                    sampled_nums[0] += num_sampled_pcls-sum(sampled_nums)
                    for cluster in range(cls_cluster_means.shape[0]):
                        m = MultivariateNormal(cls_cluster_means[cluster].float(), cls_cluster_covs[cluster].float())
                        sampled_data.append(m.sample(sample_shape=(sampled_nums[cluster], )))
                        sampled_label.extend([c_id] * sampled_nums[cluster])
                else:
                    cls_mean = self.class_means[c_id].cuda()  # torch.from_numpy(self._class_means[c_id]).to(self._device)
                    cls_cov = self.class_covs[c_id].cuda()

                    m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                    sampled_data_pcls = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_pcls)
                    sampled_label.extend([c_id] * num_sampled_pcls)

            ood_num = num_sampled_pcls
            # ood_vectors = self.ood_data[torch.randint(len(self.ood_data), (ood_num, ))].cuda()
            # sampled_data.append(ood_vectors)
            # sampled_label.extend([self.cur_classes]*ood_num)

            inputs = torch.cat(sampled_data, dim=0).float().cuda()
            targets = torch.tensor(sampled_label).long().cuda()

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]
            assert inputs.shape[0] % num_sampled_pcls == 0
            for i in range(inputs.shape[0]//num_sampled_pcls):
                inp = inputs[i * num_sampled_pcls:(i + 1) * num_sampled_pcls]
                tgt = targets[i * num_sampled_pcls:(i + 1) * num_sampled_pcls]
                # ood_vectors = self.ood_data[torch.randint(len(self.ood_data), (ood_num,))].cuda()
                ood_targets = torch.tensor([0]*num_sampled_pcls+[1]*ood_num).long().cuda()
                ood_vectors = None
                with autocast():
                    outputs = self.model.forward_with_vectors(inp, self.cur_text_tokens.cuda(), ood_vectors=ood_vectors)
                    logits = outputs['logits']
                    ood_logits = outputs["ood_logits"]

                if self.config.ca_logit_norm > 0:
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
                    decoupled_logits = torch.div(logits[:, :self.cur_classes], norms) / self.config.ca_logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits, tgt, weight=self.cur_class_weight)
                    if ood_logits is not None:
                        ood_loss = F.cross_entropy(ood_logits, ood_targets)
                        loss = loss+ood_loss

                # optimizer2.zero_grad()
                # loss.backward()
                # optimizer2.step()
                # losses += loss.item()

                optimizer2.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer2)
                self.scaler.update()

                preds = torch.max(logits, dim=-1)[1]
                if i == 0:
                    all_preds = preds
                    all_targets = tgt
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, tgt))

            train_overall_acc, _ = calculate_acc(all_preds.cpu().detach().numpy(), all_targets.cpu().detach().numpy(),
                                      self.cur_classes, self.config.increment_steps)
            self.logger.info("stage2 train acc: {}".format(train_overall_acc))

            scheduler2.step()
            if epoch == 0:
                test_preds, _, test_targets = self.predict(self.model, self.test_loader, task_id=task_id)
                test_overall_acc, _ = calculate_acc(test_preds.cpu().detach().numpy(),
                                                    test_targets.cpu().detach().numpy(),
                                                    self.cur_classes, self.config.increment_steps)
                self.logger.info("stage2 test acc: {}".format(test_overall_acc))

    def after_task(self, task_id):
        if self.config.save_checkpoint:
            checkpoint_saved_path = self.config.save_path+"/"+self.config.method+"/"+self.config.version_name
            if not os.path.exists(checkpoint_saved_path):
                os.makedirs(checkpoint_saved_path)
            save_dict = {'config': self.config,
                         'state_dict': self.model.state_dict(),
                         'task_id': task_id,
                         'class_means': self.class_means if hasattr(self, "class_means") else None,
                         'class_covs': self.class_covs if hasattr(self, "class_covs") else None }
            torch.save(save_dict, os.path.join(checkpoint_saved_path, f"checkpoint_task{task_id}" + ".pkl"))
            self.logger.info("model saved!")
