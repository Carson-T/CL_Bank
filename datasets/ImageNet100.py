import os
from torchvision import datasets, transforms
import albumentations
from albumentations import pytorch as AT
import numpy as np


class ImageNet100:
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
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([myTransforms.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]
        self.strong_transform = [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.test_transform = []
        self.common_transform = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(100).tolist()
        if shuffle:
            self.class_order = np.random.permutation(len(self.class_order)).tolist()

    def getdata(self, root_dir, txt):
        file = open(os.path.join(root_dir, txt))
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(root_dir + temp[0])
                targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        train_dataset = datasets.ImageFolder(root=os.environ["HOME"]+"/Datasets/Datasets/miniImageNet/train")
        test_dataset = datasets.ImageFolder(root=os.environ["HOME"]+"/Datasets/Datasets/miniImageNet/val")
        assert train_dataset.class_to_idx == test_dataset.class_to_idx
        self.class_to_idx = train_dataset.class_to_idx
        self.train_data, self.train_targets = [], []
        for (img_path, label) in train_dataset.imgs:
            self.train_data.append(img_path)
            self.train_targets.append(label)
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)

        self.test_data, self.test_targets = [], []
        for (img_path, label) in train_dataset.imgs:
            self.test_data.append(img_path)
            self.test_targets.append(label)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

        # self.train_data, self.train_targets = self.getdata(os.environ["HOME"]+"/Datasets/Datasets/miniImageNet", "train.txt")
        # self.test_data, self.test_targets = self.getdata(os.environ["HOME"]+"/Datasets/Datasets/miniImageNet", "val.txt")
        # self.class_to_idx = {}
        self.class_descs = {}

