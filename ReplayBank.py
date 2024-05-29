from PIL import Image
import torch.nn.functional as F
from utils.functions import *
from utils.dataset import MyDataset
from utils.functions import pil_loader

class ReplayBank:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        assert config.memory_size is not None or config.memory_per_class is not None, "ReplayBank Error"
        self.memory_size = config.memory_size
        self.sampling_method = config.sampling_method

        self.memory_per_class = 0
        self.samples_memory = np.array([])
        self.targets_memory = np.array([])
        self.soft_targets_memory = np.array([])
        self.class_examplar_info = []  # 列表中保存了每个类实际保存的样本数
        self.num_seen_examplars = 0

    def update_param(self, new_class_dataset):
        new_classes = np.unique(new_class_dataset.targets)
        assert min(new_classes) + 1 > len(
            self.class_examplar_info), "Store_samples's dataset should not overlap with buffer"
        if self.config.memory_per_class is not None:
            self.memory_per_class = self.config.memory_per_class
        else:
            self.memory_per_class = self.memory_size // (len(self.class_examplar_info) + len(new_classes))
        self.logger.info("current memory per class: {}".format(self.memory_per_class))

    def get_memory(self):
        return self.samples_memory, self.targets_memory

    def reduce_examplars(self, memory_per_class):
        samples_memory, targets_memory = [], []
        for i in range(len(self.class_examplar_info)):
            if self.class_examplar_info[i] > memory_per_class:
                store_sample_size = memory_per_class
            else:
                store_sample_size = self.class_examplar_info[i]

            mask = np.where(self.targets_memory == i)[0]
            samples_memory.append(self.samples_memory[mask[:store_sample_size]])
            targets_memory.append(self.targets_memory[mask[:store_sample_size]])

            self.class_examplar_info[i] = store_sample_size
            # self.logger.info("类别 {} 存储样本数为: {}".format(i, len(self.samples_memory[i])))
        self.samples_memory = np.concatenate(samples_memory)
        self.targets_memory = np.concatenate(targets_memory)

    def store_examplars(self, new_class_dataset, model):
        if any([i > self.memory_per_class for i in self.class_examplar_info]):
            self.reduce_examplars(self.memory_per_class)

        samples_memory, targets_memory = [], []
        if len(self.class_examplar_info) > 0:
            samples_memory.append(self.samples_memory)
            targets_memory.append(self.targets_memory)

        new_class_samples, new_class_targets, new_class_examplar_info = self.select_class_examplar(new_class_dataset, model)
        self.class_examplar_info += new_class_examplar_info
        self.samples_memory = np.concatenate(samples_memory + new_class_samples)
        self.targets_memory = np.concatenate(targets_memory + new_class_targets)
        self.logger.info("examplar num of each class: {}".format(self.class_examplar_info))

    def get_balanced_data(self, new_class_dataset, model):
        balanced_samples = []
        balanced_targets = []
        if len(self.class_examplar_info) > 0:
            balanced_samples.append(self.samples_memory)
            balanced_targets.append(self.targets_memory)

        # balanced new task data and targets
        new_class_balanced_samples, new_class_balanced_targets, _ = self.select_class_examplar(new_class_dataset, model)

        balanced_samples, balanced_targets = (np.concatenate(balanced_samples + new_class_balanced_samples),
                                              np.concatenate(balanced_targets + new_class_balanced_targets))
        self.logger.info("balanced data num: {}".format(len(balanced_samples)))
        return MyDataset(balanced_samples, balanced_targets, new_class_dataset.use_path, new_class_dataset.transform)

    def select_class_examplar(self, dataset, model):
        samples = []
        targets = []
        class_examplar_info = []
        for class_idx in np.unique(dataset.targets):
            class_vectors, class_samples, class_targets = extract_vectors(self.config, model, dataset, class_idx)
            if self.sampling_method == 'herding':
                selected_idx = herding_select(class_vectors, self.memory_per_class)
            else:
                raise ValueError('Unknown sample select strategy: {}'.format(self.sampling_method))
            samples.append(class_samples[selected_idx])
            targets.append(class_targets[selected_idx])
            class_examplar_info.append(len(class_samples[selected_idx]))

        return samples, targets, class_examplar_info

    def cal_class_means(self, model, dataset):
        class_means = []
        # self.logger.info('Calculating class means for stored classes...')
        for class_idx in np.unique(self.targets_memory):
            # class_samples, class_targets, class_data_idx = select(self.samples_memory, self.targets_memory, class_idx,
            #                                                       class_idx + 1)
            # class_dataset = MyDataset(class_samples, class_targets, use_path, transform)
            class_vectors, _, _ = extract_vectors(self.config, model, dataset, class_idx)

            class_vectors = F.normalize(class_vectors, dim=1)  # 对特征向量做归一化
            mean = torch.mean(class_vectors, dim=0)
            mean = F.normalize(mean, dim=0)
            class_means.append(mean)
        self.logger.info('Calculated class mean of classes {}'.format(np.unique(self.targets_memory)))

        class_means = torch.stack(class_means, dim=0) if len(class_means) > 0 else None
        return class_means

    def KNN_classify(self, vectors, class_means, ret_logits=False):
        assert class_means is not None, 'class means is None'
        vectors = F.normalize(vectors, dim=1)  # 对特征向量做归一化
        dists = torch.cdist(vectors, class_means, p=2)
        min_scores, nme_predicts = torch.min(dists, dim=1)
        if ret_logits:
            return nme_predicts, dists
        else:
            return nme_predicts, 1 - min_scores

    def store_samples_reservoir(self, samples, logits, labels):
        """ This function is for DarkER and DarkER++ """
        init_size = 0
        if len(self.samples_memory) == 0:
            init_size = min(len(samples), self.memory_size)
            self.samples_memory = samples[:init_size]
            self.targets_memory = labels[:init_size]
            self.soft_targets_memory = logits[:init_size]
            self.num_seen_examplars += init_size
        elif len(self.samples_memory) < self.memory_size:
            init_size = min(len(samples), self.memory_size - len(self.samples_memory))
            self.samples_memory = np.concatenate([self.samples_memory, samples[:init_size]])
            self.targets_memory = np.concatenate([self.targets_memory, labels[:init_size]])
            self.soft_targets_memory = np.concatenate([self.soft_targets_memory, logits[:init_size]])
            self.num_seen_examplars += init_size

        for i in range(init_size, len(samples)):
            index = np.random.randint(0, self.num_seen_examplars + 1)
            self.num_seen_examplars += 1
            if index < self.memory_size:
                self.samples_memory[index] = samples[i]
                self.targets_memory[index] = labels[i]
                self.soft_targets_memory[index] = logits[i]

    def get_memory_reservoir(self, size, use_path, transform=None, ret_idx=False):
        if size > min(self.num_seen_examplars, self.memory_size):
            size = min(self.num_seen_examplars, self.memory_size)

        choice = np.random.choice(min(self.num_seen_examplars, self.memory_size), size=size, replace=False)

        data_all = []
        for sample in self.samples_memory[choice]:
            if transform is None:
                data_all.append(torch.from_numpy(sample))  # [h, w, c]
            elif use_path:
                data_all.append(transform(pil_loader(sample)))  # [c, h, w]
            else:
                data_all.append(transform(Image.fromarray(sample)))  # [c, h, w]
        data_all = torch.stack(data_all)

        targets_all = torch.from_numpy(self.targets_memory[choice])
        soft_targets_all = torch.from_numpy(self.soft_targets_memory[choice])

        ret = (data_all, targets_all, soft_targets_all)
        if ret_idx:
            ret = (torch.tensor(choice),) + ret

        return ret