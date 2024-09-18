import os
import json
from torchvision import datasets, transforms
import albumentations
from albumentations import pytorch as AT
import numpy as np


class Cars196:

    def __init__(self, shuffle=False, img_size=224) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size
        self.train_transform = [
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ]
        self.strong_transform = [
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(196).tolist()
        if shuffle:
            self.class_order = np.random.permutation(len(self.class_order)).tolist()


    def getdata(self, train:bool, root_dir):
        data, targets = [], []
        class_to_idx = {}
        for item in self.info["train" if train else "test"]:
            data.append(os.path.join(root_dir, item[0]))
            targets.append(int(item[1]))
            class_to_idx[item[2]] = int(item[1])

        return np.array(data), np.array(targets), class_to_idx

    def download_data(self):
        root_dir = os.path.join(os.environ["HOME"], 'Data/stanford_cars')

        with open(root_dir+"/split_StanfordCars.json", "r") as f:
            self.info = json.load(f)

        self.train_data, self.train_targets, self.class_to_idx = self.getdata(True, root_dir)
        self.test_data, self.test_targets, self.class_to_idx = self.getdata(False, root_dir)
