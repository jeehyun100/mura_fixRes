""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import torch as t


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        #1,320,320
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

            # nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2, 2),
            # # 512 20 20
            # nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2, 2),
            # #512 10 10
            # nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2, 2)


        )
        self.classifier = nn.Linear(12544, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        #x = t.view(-1)
        #x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        x = t.sigmoid(x)
        return x
        #logits = self.outc(x)
        #return logits
