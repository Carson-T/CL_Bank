from torch.utils.data import Dataset
import cv2
from PIL import Image
from torchvision import datasets, transforms


class MyDataset(Dataset):
    def __init__(self, samples, targets, use_path, transform=None, clip_transform=None):
        super(MyDataset, self).__init__()
        assert len(samples) == len(targets), "MyDataset Error"
        self.samples = samples
        self.targets = targets
        self.use_path = use_path
        self.transform = transform
        self.clip_transform = clip_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_path:
            original_img = Image.open(self.samples[idx])
            img = original_img.convert("RGB")
        else:
            original_img = self.samples[idx]
            img = Image.fromarray(original_img)
        target = self.targets[idx]
        if self.transform is not None:
            img1 = self.transform(img)
        else:
            img1 = img

        if self.clip_transform is not None:
            clip_img = self.clip_transform(img)
            return img1, target, idx, clip_img

        return img1, target, idx

