from __future__ import print_function, division

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import sampler

import time
import copy
import torch.nn.functional as F  # useful stateless functions
from torch.utils.data import DataLoader

from ssdd.dataset import MaskDataset

torch.backends.cudnn.benchmark = True

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

msk_transform = transforms.Compose([
    transforms.ToTensor(),
])


dataset = MaskDataset("./data",
                      img_transform=img_transform,
                      msk_transform=msk_transform)

ALL_TOTAL = 400
num_train = 300
num_val = 100
num_test = 0

dataloaders = dict()
dataloaders['train'] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5,
                                  sampler=sampler.SubsetRandomSampler(range(num_train)))

dataloaders['val'] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5,
                                sampler=sampler.SubsetRandomSampler(range(num_train, num_train + num_val)))
dataset_sizes = {
    'train': num_train,
    'val': num_val
}

print(dataset_sizes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train_model_seg(seg_model, optimizer, scheduler, num_epochs=25):
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                seg_model.train()  # Set model to training mode
            else:
                pass
                # model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            filename = 0
            dataset_iter = dataloaders[phase].__iter__()
            for _ in range(len(dataloaders[phase])):
                inputs, masks, labels = dataset_iter.__next__()

                inputs = inputs.to(device)
                masks = masks.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    feat, pred_mask = seg_model.forward(inputs)
                    # pred_mask = torch.sigmoid(pred_mask)
                    loss = F.binary_cross_entropy(pred_mask.reshape(1, -1), masks.reshape(1, -1))
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if epoch % 3 == 0:
                        with torch.set_grad_enabled(False):
                            save_dir = "./visualization/{}_epoch-{}".format(phase, epoch)
                            visualization(save_dir, str(filename) + ".jpg", inputs[0][0], masks[0][0], pred_mask[0][0])
                            filename += 1
                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    return seg_model


def visualization(save_dir, filename, img, mask, pred_mask):
    import os
    import numpy as np
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img = img.cpu()
    mask = mask.cpu()
    pred_mask = pred_mask.cpu()
    image = np.array(img) * 255
    mask = np.array(mask) * 255
    pred_mask = np.array(pred_mask.detach().numpy()) * 255
    img_visual = concat_image([image, mask, pred_mask])
    visualization_path = os.path.join(save_dir, filename)
    img_visual.save(visualization_path)


def concat_image(images, mode="L"):
    from PIL import Image
    if not isinstance(images, list):
        raise Exception('images must be a list  ')
    count = len(images)
    size = Image.fromarray(images[0]).size
    target = Image.new(mode, (size[0] * count, size[1] * 1))
    for i in range(count):
        image = Image.fromarray(images[i]).resize(size, Image.BILINEAR)
        target.paste(image, (i*size[0], 0, (i+1)*size[0], size[1]))
    return target


def run_seg():
    from ssdd.model import SegmentationNet
    seg_model = SegmentationNet(1)
    seg_model = seg_model.double()
    seg_model = seg_model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(seg_model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    seg_model = train_model_seg(seg_model, optimizer, exp_lr_scheduler, num_epochs=7)
    import os
    save_dir = "./saved/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(seg_model, save_dir + 'segmentation')


def train_model_cls(seg_model, cls_model, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(cls_model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cls_model.train()  # Set model to training mode
            else:
                cls_model.train()  # Set model to training mode
                pass
                # model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = torch.tensor(0)
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            dataset_iter = dataloaders[phase].__iter__()
            for _ in range(len(dataset_iter)):
                inputs, masks, labels = dataset_iter.__next__()

                inputs = inputs.to(device)
                # masks = masks.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.no_grad():
                    feats, pred_masks = seg_model.forward(inputs)

                with torch.set_grad_enabled(phase == 'train'):
                    scores = cls_model.forward(feats, pred_masks)
                    cls_loss = F.cross_entropy(scores, labels)

                    _, preds = torch.max(scores, 1)

                    # loss = cls_loss + seg_loss
                    if phase == 'train':
                        cls_loss.backward()
                        optimizer.step()

                # statistics
                if preds[0] == labels[0]:
                    if preds[0] == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if preds[0] == 1:
                        fp += 1
                    else:
                        fn += 1

                running_loss += cls_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            print('tp: {} fn: {} fp: {} tn: {}'.format(tp, fn, fp, tn))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(cls_model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    cls_model.load_state_dict(best_model_wts)
    return cls_model


def run_cls(save_dir):
    from ssdd.model import DecisionNet

    # seg_model = SegmentationNet(1)
    seg_model = torch.load(save_dir)
    seg_model = seg_model.double()
    seg_model = seg_model.to(device)
    cls_model = DecisionNet()
    cls_model = cls_model.double()
    cls_model = cls_model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(cls_model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    cls_model = train_model_cls(seg_model, cls_model, optimizer, exp_lr_scheduler, num_epochs=4)
    import os
    save_dir = "./saved/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(cls_model, save_dir + 'classification')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print()
        print('Train Segmentation Network!')
        run_seg()
        print()
        print('Train Classification Network!')
        run_cls('./saved/segmentation')
    else:
        if sys.argv[1] == 'seg':
            run_seg()
        elif sys.argv[1] == 'cls':
            run_cls('./saved/segmentation')
        else:
            print('run `python train.py seg` or `python train.py cls`')
