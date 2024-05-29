from torchvision import datasets, transforms
import albumentations
from albumentations import pytorch as AT
import numpy as np


class CIFAR100:
    """
    Dataset Name:   CIFAR-100 dataset (Canadian Institute for Advanced Research, 100 classes)
    Source:         A subset of the Tiny Images dataset.
    Task:           Classification Task
    Data Format:    32x32 color images.
    Data Amount:    60000 (500 training images and 100 testing images per class)
    Class Num:      100 (grouped into 20 superclass).
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self, shuffle=False, img_size=32) -> None:
        super().__init__()
        self.use_path = False
        self.img_size = img_size
        self.train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255)
        ]
        # self.train_transform = [
        #     albumentations.HorizontalFlip(p=0.5),
        #     albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        #     # albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        #     # albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=0, p=0.5),
        # ]
        self.test_transform = []
        self.common_transform = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]
        # self.common_transform = [
        #     albumentations.Resize(self.img_size, self.img_size),
        #     albumentations.Normalize([0.5071, 0.4867, 0.4408],
        #                              [0.2675, 0.2565, 0.2761]),
        #     AT.ToTensorV2()
        # ]

        self.class_order = np.arange(100).tolist()
        if shuffle:
            self.class_order = np.random.permutation(len(self.class_order)).tolist()
        # self.class_order = [
        #         87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45,
        #         88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6,
        #         46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
        #         40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39
        #     ]

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/data/jiantao/Data/cifar100", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("/data/jiantao/Data/cifar100", train=False, download=True)

        # self.class_to_idx = train_dataset.class_to_idx
        self.class_to_idx = {}
        for key, value in train_dataset.class_to_idx.items():
            if key == 'aquarium_fish':
                print('class {}: {} => {}'.format(value, key, 'goldfish'))
                key = 'goldfish'
            elif key == 'lawn_mower':
                print('class {}: {} => {}'.format(value, key, 'mower'))
                key = 'mower'
            elif key == 'maple_tree':
                print('class {}: {} => {}'.format(value, key, 'maple'))
                key = 'maple'
            elif key == 'oak_tree':
                print('class {}: {} => {}'.format(value, key, 'oak'))
                key = 'oak'
            elif key == 'palm_tree':
                print('class {}: {} => {}'.format(value, key, 'palm'))
                key = 'palm'
            elif key == 'pickup_truck':
                print('class {}: {} => {}'.format(value, key, 'truck'))
                key = 'truck'
            elif key == 'pine_tree':
                print('class {}: {} => {}'.format(value, key, 'pine'))
                key = 'pine'
            elif key == 'sweet_pepper':
                print('class {}: {} => {}'.format(value, key, 'pepper'))
                key = 'pepper'
            elif key == 'willow_tree':
                print('class {}: {} => {}'.format(value, key, 'willow'))
                key = 'willow'
            else:
                print('class {}: {}'.format(value, key))
            self.class_to_idx[key] = value

        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)
