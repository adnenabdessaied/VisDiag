#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

from torchvision.models import vgg16, googlenet
import torch.nn as nn
from torch.nn import Sequential


"""
A class that reads images, extracts their features and saves them into disk. The features will be used in a later stage 
by the dataset. We use a pre-trained VGG16 for the encoding  
"""


class VGG_clipped(nn.Module):
    def __init__(self, net=vgg16):
        """
        Constructor of the class
        :param net: The net to be modified
        """
        super(VGG_clipped, self).__init__()
        net = net(pretrained=True)
        net = net.double()
        self.features = net.features
        self.avgpool = net.avgpool
        self.classifier = Sequential(*list(net.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 25088)
        x = self.classifier(x)
        return x


# TODO: Take a further look at this! This is likely to be wrong!!
class Googlenet_clipped(nn.Module):
    def __init__(self, net=googlenet):
        super(Googlenet_clipped, self).__init__()
        net = net(pretrained=True).eval()
        net = net.double()
        self.net = Sequential(*list(net.children())[:-2])

    def forward(self, x):
        return self.net(x).view(-1, 1024)
