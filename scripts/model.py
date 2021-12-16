# Unet based on https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from train import DROPOUT


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 3 1 1 = kernel, stride, padding
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                 )
        self.dropout = nn.Dropout2d(p=DROPOUT)

    def forward(self, x):
        x = self.conv(x)
        if DROPOUT > 0 :
            x = self.dropout(x)
        return x


class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #For the down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #For the up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2)
                )
            self.ups.append(DoubleConv(feature*2, feature))

        #For the bottom part of UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        #For the final part of UNET
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        #To reverse the skip connections for the up part of UNET
        skip_connections = skip_connections[::-1]

        #Setp of 2 since we will do "up,doubleconv" at every step of the up part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            #Since we are doing steps of 2 in the loop --> integer div by 2 to get the skip_connections in a linear step of 1 ordering
            skip_connection = skip_connections[idx//2]

            #Resize otherwise we could lose some info if we have an odd number for the shape
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x), dim=1)
            #run it in the double conv
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


