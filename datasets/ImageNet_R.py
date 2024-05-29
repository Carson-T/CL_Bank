from torchvision import datasets, transforms
import albumentations
from albumentations import pytorch as AT
import numpy as np


class ImageNet_R:
    '''
    Dataset Name:   ImageNet_R dataset
    Source:         A subset of the Tiny Images dataset.
    Task:           Classification Task
    Data Format:    32x32 color images.
    Data Amount:    60000 (500 training images and 100 testing images per class)
    Class Num:      100 (grouped into 20 superclass).
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    '''

    def __init__(self, shuffle=False, img_size=224) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size

        self.train_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            # transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.test_transform = []
        self.common_transform = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.230, 0.227, 0.226]),
        ]
        self.class_order = np.arange(200).tolist()
        if shuffle:
            self.class_order = np.random.permutation(len(self.class_order)).tolist()

    def download_data(self):
        train_dataset = datasets.ImageFolder(root="/data/jiantao/Data/imagenet_r_coda/train")
        test_dataset = datasets.ImageFolder(root="/data/jiantao/Data/imagenet_r_coda/test")
        assert train_dataset.class_to_idx == test_dataset.class_to_idx

        with open("/data/jiantao/Data/imagenet_r_coda/class_name.txt", "r") as f:
            lines = f.readlines()
        labels = [item[0:9] for item in lines]
        class_names = [item[10:-1] for item in lines]

        self.class_to_idx = {}
        for i in range(len(labels)):
            self.class_to_idx[class_names[i]] = train_dataset.class_to_idx[labels[i]]
        train_dataset.class_to_idx = self.class_to_idx
        test_dataset.class_to_idx = self.class_to_idx

        self.train_data, self.train_targets = np.array([]), np.array([])
        for (img_path, label) in train_dataset.imgs:
            self.train_data = np.append(self.train_data, img_path)
            self.train_targets = np.append(self.train_targets, label)

        self.test_data, self.test_targets = np.array([]), np.array([])
        for (img_path, label) in test_dataset.imgs:
            self.test_data = np.append(self.test_data, img_path)
            self.test_targets = np.append(self.test_targets, label)

