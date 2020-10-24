""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import torch as t


class UNetFineTune(nn.Module):
    def __init__(self, pretrained_model):
        super(UNetFineTune, self).__init__()
        self.pretrained_model = pretrained_model

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=0), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=0), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 160 160
            nn.Conv2d(64, 128, 3, padding=0), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=0), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 80 80
            nn.Conv2d(128, 256, 3, padding=0), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=0), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=0), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 10,256,35,35
            #313600
            nn.Conv2d(256, 128, 3, padding=0), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=0), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, padding=0), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(12544, 1)
        self.n_classes = pretrained_model.n_classes

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        x = t.sigmoid(x)
        return x
        # logits = self.outc(x)
        # return logits
