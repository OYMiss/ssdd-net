import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class SegmentationNet(nn.Module):
    def __init__(self, img_channels):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv15x15 = nn.Sequential(
            nn.Conv2d(64, 1024, 15, padding=7, bias=False),
            nn.BatchNorm2d(1024, affine=True),
            nn.ReLU())
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(1024, 1, 1, bias=False),
            nn.BatchNorm2d(1, affine=True))
        for param in self.parameters():
            if param.ndim == 4:
                torch.nn.init.xavier_uniform_(param)

    def requires_grad(self, is_required):
        for param in self.parameters():
            param.requires_grad = is_required

    def forward(self, x):
        """
        :param x: x(N, C, H, W)
        :return: feat(N, 1024, H, W) mask(N, 1, H, W)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.conv15x15(x)
        mask = self.conv1x1(feat)
        mask = torch.sigmoid(mask)
        return feat, mask


class DecisionNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.pre_pool = nn.MaxPool2d(2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1025, 8, 5, padding=2, bias=False),
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, padding=2, bias=False),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU())
        self.fc = nn.Linear(66, 2)
        for param in self.parameters():
            if param.ndim == 4:
                torch.nn.init.kaiming_normal_(param)

    @staticmethod
    def global_avg_pool(x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = F.avg_pool2d(x, (h, w))
        return x.reshape(n, c)

    @staticmethod
    def global_max_pool(x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = F.max_pool2d(x, (h, w))
        return x.reshape(n, c)

    def forward(self, feat, mask):
        x = torch.cat([feat, mask], dim=1)
        x = self.pre_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_max = self.global_max_pool(x)
        x_avg = self.global_avg_pool(x)
        mask_max = self.global_max_pool(mask)
        mask_avg = self.global_avg_pool(mask)
        x = torch.cat([x_max, x_avg, mask_max, mask_avg], dim=1)
        score = self.fc(x)
        return score


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.seg_net = SegmentationNet(img_channels=1)
        self.cls_net = DecisionNet()

    def forward(self, x, mode='seg'):
        """
        :param x: the image N, C, H, W
        :param mode: seg, cls
        :return: seg: mask, cls, score
        """
        self.seg_net.requires_grad(mode == 'seg')
        feat, mask = self.seg_net(x)
        if mode == 'seg':
            return feat, mask
        score = self.cls_net(feat, mask)
        return score


def test():
    img = torch.rand((4, 1, 64, 64))
    seg_net = SegmentationNet(1)
    cls_net = DecisionNet()
    feat, mask = seg_net.forward(img)
    score = cls_net.forward(feat, mask)
    print(feat.shape)
    print(mask.shape)
    print(score.shape)


def check():
    img = torch.ones((1, 1, 64, 64))
    mask_true = torch.ones((1, 8, 8))
    # mask_true[0][3, 3] = 1
    # mask_true[0][3, 4] = 1
    # mask_true[0][4, 4] = 1
    # mask_true[0][4, 3] = 1
    seg = SegmentationNet(1)
    feat, mask = seg.forward(img)
    loss = F.binary_cross_entropy(mask.reshape(1, -1), mask_true.reshape(1, -1))
    print(feat.shape)
    print(mask.shape)
    print(mask)
    print(loss)


if __name__ == '__main__':
    # test()
    check()
