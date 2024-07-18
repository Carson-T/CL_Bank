from torchvision import transforms
import os
import numpy as np

class Skin40():
    '''
    Dataset Name:   Skin40 (Subset of SD-198)
    Task:           Skin disease classification
    Data Format:    224x224 color images. (origin imgs have different w,h)
    Data Amount:    2000 imgs for training and 400 for validationg/testing
    Class Num:      40
    Label:

    Reference:      https://link.springer.com/chapter/10.1007/978-3-319-46466-4_13
    '''
    def __init__(self, shuffle=False, img_size=224, seed=1993) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size
        self.train_transform = [
            transforms.RandomHorizontalFlip(),
            ]

        self.test_transform = []

        self.common_transform = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.60298395, 0.4887822, 0.46266827], std=[0.25993535, 0.24081337, 0.24418062]),
        ]

        self.class_order = np.arange(40).tolist()
        if shuffle:
            self.class_order = np.random.permutation(len(self.class_order)).tolist()

    def getdata(self, fn, img_dir):
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(os.path.join(img_dir, temp[0]))
                targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        train_dir = os.path.join(os.environ["HOME"], "Datasets/Datasets/SD-198/main_classes_split/train_1.txt")
        test_dir = os.path.join(os.environ["HOME"], "Datasets/Datasets/SD-198/main_classes_split/val_1.txt")
        img_dir = os.path.join(os.environ["HOME"], 'Datasets/Datasets/SD-198/images')

        self.class_to_idx = {
            "Stasis Ulcer": 0,
            "Actinic Solar Damage(Solar Elastosis)": 1,
            "Congenital Nevus": 2,
            "Inverse Psoriasis": 3,
            "Perioral Dermatitis": 4,
            "Stasis Dermatitis": 5,
            "Pyogenic Granuloma": 6,
            "Malignant Melanoma": 7,
            "Steroid Use/abusemisuse Dermatitis": 8,
            "Sebaceous Gland Hyperplasia": 9,
            "Rhinophyma": 10,
            "Seborrheic Dermatitis": 11,
            "Psoriasis": 12,
            "Onychomycosis": 13,
            "Tinea Versicolor": 14,
            "Tinea Corporis": 15,
            "Alopecia Areata": 16,
            "Actinic Solar Damage(Pigmentation)": 17,
            "Blue Nevus": 18,
            "Keratoacanthoma": 19,
            "Acne Vulgaris": 20,
            "Cutaneous Horn": 21,
            "Tinea Pedis": 22,
            "Stasis Edema": 23,
            "Dyshidrosiform Eczema": 24,
            "Skin Tag": 25,
            "Seborrheic Keratosis": 26,
            "Ichthyosis": 27,
            "Actinic Solar Damage(Actinic Keratosis)": 28,
            "Nevus Incipiens": 29,
            "Tinea Faciale": 30,
            "Epidermoid Cyst": 31,
            "Compound Nevus": 32,
            "Eczema": 33,
            "Pityrosporum Folliculitis": 34,
            "Dysplastic Nevus": 35,
            "Allergic Contact Dermatitis": 36,
            "Basal Cell Carcinoma": 37,
            "Tinea Manus": 38,
            "Dermatofibroma": 39
        }

        self.train_data, self.train_targets = self.getdata(train_dir, img_dir)
        self.test_data, self.test_targets = self.getdata(test_dir, img_dir)