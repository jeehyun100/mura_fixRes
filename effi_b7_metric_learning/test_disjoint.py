from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from PIL import Image
import logging
import matplotlib.pyplot as plt
import umap
from cycler import cycler
import record_keeper
import pytorch_metric_learning

# Set the image transforms
train_transform = transforms.Compose([transforms.Resize(64),
                                    transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Download the original datasets
original_train = datasets.CIFAR100(root="CIFAR100_Dataset", train=True, transform=None, download=True)
original_val = datasets.CIFAR100(root="CIFAR100_Dataset", train=False, transform=None, download=True)


# This will be used to create train and val sets that are class-disjoint
class ClassDisjointCIFAR100(torch.utils.data.Dataset):
    def __init__(self, original_train, original_val, train, transform):
        rule = (lambda x: x < 50) if train else (lambda x: x >= 50)
        train_filtered_idx = [i for i, x in enumerate(original_train.targets) if rule(x)]
        val_filtered_idx = [i for i, x in enumerate(original_val.targets) if rule(x)]
        self.data = np.concatenate([original_train.data[train_filtered_idx], original_val.data[val_filtered_idx]],
                                   axis=0)
        self.targets = np.concatenate(
            [np.array(original_train.targets)[train_filtered_idx], np.array(original_val.targets)[val_filtered_idx]],
            axis=0)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# Class disjoint training and validation set
train_dataset = ClassDisjointCIFAR100(original_train, original_val, True, train_transform)
val_dataset = ClassDisjointCIFAR100(original_train, original_val, False, val_transform)
assert set(train_dataset.targets).isdisjoint(set(val_dataset.targets))



# borrowed from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
for i, data in enumerate(train_dataset, 0):
    inputs, labels = data
    print("labels {0}".format(labels))
    print("inputs {0}".format(inputs))
    # zero the parameter gradients
    #optimizer.zero_grad()

    # forward + backward + optimize
    #embeddings = net(inputs)
    #hard_pairs = miner(embeddings, labels)
    #loss = loss_func(embeddings, labels, hard_pairs)
    #loss.backward()
    #optimizer.step()