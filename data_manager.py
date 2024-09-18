import numpy as np
import albumentations
from torchvision import transforms
from datasets import *
from utils.functions import *
from utils.dataset import MyDataset


class DataManager(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        if config.dataset_name == "cifar100":
            self.data = CIFAR100(shuffle=config.data_shuffle, img_size=config.img_size)
        elif config.dataset_name == "imagenet-r":
            self.data = ImageNet_R(shuffle=config.data_shuffle, img_size=config.img_size)
        elif config.dataset_name == "imagenet100":
            self.data = ImageNet100(shuffle=config.data_shuffle, img_size=config.img_size)
        elif config.dataset_name == "Skin40":
            self.data = Skin40(shuffle=config.data_shuffle, img_size=config.img_size)
        elif config.dataset_name == "CUB200":
            self.data = CUB200(shuffle=config.data_shuffle, img_size=config.img_size)
        elif config.dataset_name == "Cars196":
            self.data = Cars196(shuffle=config.data_shuffle, img_size=config.img_size)
        self.img_size = self.data.img_size
        self.use_valid = False

        self.data.download_data()
        if hasattr(self.data, "class_descs"):
            self.class_descs = self.data.class_descs
        else:
            self.class_descs = None
        self.class_order = self.data.class_order
        self.logger.info("class_order: {}".format(self.class_order))
        self.total_class_num = len(self.data.class_order)
        self.class_to_idx = self.data.class_to_idx
        self.train_data, self.train_targets = self.data.train_data, self.data.train_targets
        self.test_data, self.test_targets = self.data.test_data, self.data.test_targets
        self.logger.info("train data num: {}".format(len(self.train_data)))
        self.logger.info("test data num: {}".format(len(self.test_data)))

        # Transforms
        self.train_transform = self.data.train_transform
        self.test_transform = self.data.test_transform
        self.common_transform = self.data.common_transform

        # Map indices
        self.train_targets = self.map_index(self.train_targets, self.class_order)
        self.test_targets = self.map_index(self.test_targets, self.class_order)
        # Map class_to_idx
        for key in self.class_to_idx.keys():
            self.class_to_idx[key] = self.class_order.index(self.class_to_idx[key])

    def map_index(self, y, order):
        # map class y to its index of order
        # y = [0, 1, 2, 3, 4]
        # order = [1, 3, 0, 2, 4]
        # result = [2, 0, 3, 1, 4] : 0 -> 2, 1 -> 0, 2 -> 3, 3 -> 1, 4 -> 4
        return np.array(list(map(lambda x: order.index(x), y)))

    @property
    def num_tasks(self):
        return len(self.config.increment_steps)

    @property
    def total_classes(self):
        return self.total_class_num

    def get_task_size(self, task_id):
        return self.config.increment_steps[task_id]

    def get_dataset(self, source, mode, indices, appendent=None, ret_clip_img=False):
        if source == 'train':
            x, y = self.train_data, self.train_targets
        elif source == 'test':
            x, y = self.test_data, self.test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            transform = transforms.Compose([*self.train_transform, *self.common_transform])
            clip_transform = transforms.Compose([
                *self.train_transform,
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ])
        elif mode == 'test' or mode == 'valid':
            transform = transforms.Compose([*self.test_transform, *self.common_transform])
            clip_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets, _ = select(x, y, low_range=idx, high_range=idx + 1)
            data.append(class_data)
            targets.append(class_targets)
        if appendent is not None:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)

        return MyDataset(data, targets, self.data.use_path, transform, clip_transform=clip_transform if ret_clip_img else None)

    def get_openset_dataset(self, source, mode, known_indices, ret_clip_img=False):
        """
        known_indices 必须是连续的, 从0开始的
        """
        self.logger.info(
            'getting openset dataset: {}-{} classes have been learned.'.format(known_indices[0], known_indices[-1]))
        if source == 'train':
            x, y = self.train_data, self.train_targets
        elif source == 'test':
            x, y = self.test_data, self.test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            transform = transforms.Compose([*self.train_transform, *self.common_transform])
            clip_transform = transforms.Compose([
                *self.train_transform,
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ])
        elif mode == 'test' or mode == 'valid':
            transform = transforms.Compose([*self.test_transform, *self.common_transform])
            clip_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in known_indices:
            class_data, class_targets, _ = select(x, y, low_range=idx, high_range=idx + 1)
            data.append(class_data)
            targets.append(class_targets)
        unknown_class_data, unknown_class_targets, _ = select(x, y, low_range=max(known_indices) + 1, high_range=self.total_classes)
        data.append(unknown_class_data)
        targets.append(np.full_like(unknown_class_targets, self.total_classes))

        data, targets = np.concatenate(data), np.concatenate(targets)

        return MyDataset(data, targets, self.data.use_path, transform, clip_transform=clip_transform if ret_clip_img else None)
