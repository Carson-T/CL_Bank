import os
import json
from torchvision import datasets, transforms
import albumentations
from albumentations import pytorch as AT
import numpy as np

class CUB200:
    '''
    Dataset Name:   CUB200-2011
    Task:           fine-grain birds classification
    Data Format:    224x224 color images. (origin imgs have different w,h)
    Data Amount:    5,994 images for training and 5,794 for validationg/testing
    Class Num:      200
    Label:

    Reference:      https://opendatalab.com/CUB-200-2011
    '''

    def __init__(self, shuffle=False, img_size=224) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size
        self.train_transform = [
            # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
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

        self.class_order = np.arange(200).tolist()
        if shuffle:
            self.class_order = np.random.permutation(len(self.class_order)).tolist()


    def getdata(self, train:bool, root_dir, img_dir):
        data, targets = [], []
        with open(os.path.join(root_dir, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_train = line.split()
                if int(is_train) == int(train):
                    data.append(os.path.join(img_dir, self.images_path[image_id]))
                    targets.append(self.class_ids[image_id])

        return np.array(data), np.array(targets)

    def download_data(self):
        root_dir = os.path.join(os.environ["HOME"], 'Data/CUB_200_2011')
        img_dir = os.path.join(root_dir, 'images')

        self.images_path = {}
        with open(os.path.join(root_dir, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(root_dir, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = int(class_id) - 1

        self.class_to_idx = {}
        with open(os.path.join(root_dir, 'classes.txt')) as f:
            for line in f:
                class_id, class_name = line.split()
                # self.class_to_idx[class_name] = int(class_id) - 1
                self.class_to_idx[class_name[4:]] = int(class_id) - 1

        self.train_data, self.train_targets = self.getdata(True, root_dir, img_dir)
        self.test_data, self.test_targets = self.getdata(False, root_dir, img_dir)

        # print(len(np.unique(self.train_targets))) # output: 200
        # print(len(np.unique(self.test_targets))) # output: 200