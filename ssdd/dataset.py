import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, data_dir, img_transform, msk_transform, img_resize=(256, 256), msk_resize=(32, 32)):
        self.image_paths = []
        self.masks_paths = []
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.img_resize = img_resize
        self.msk_resize = msk_resize
        data_list = os.listdir(data_dir)

        for name in data_list:
            if name[-4:] == '.jpg' and 'label' not in name:
                label_name = name[:-4] + "_label.bmp"
                img_path = data_dir + "/" + name
                msk_path = data_dir + "/" + label_name
                self.image_paths.append(img_path)
                self.masks_paths.append(msk_path)

    def __getitem__(self, index):
        i = index % len(self)
        img_path = self.image_paths[i]
        msk_path = self.masks_paths[i]

        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, self.img_resize)

        msk = cv2.imread(msk_path, 0)
        msk = cv2.resize(msk, self.msk_resize)
        msk = np.where(msk > 1, 1, 0)

        label = np.sum(msk) == 0

        return self.img_transform(img).float(), \
               self.msk_transform(msk).float(), \
               torch.tensor(label).long()

    def __len__(self):
        return len(self.image_paths)


def test():
    from torchvision import datasets, models, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    test()
