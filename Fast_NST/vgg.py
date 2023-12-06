from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # print(vgg_pretrained)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for i in range(4):
            self.slice1.add_module(str(i), vgg_pretrained[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vgg_pretrained[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vgg_pretrained[i])
        for i in range(16, 23):
            self.slice4.add_module(str(i), vgg_pretrained[i])
        if requires_grad == False:
            for para in vgg_pretrained.parameters():
                para.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
    
# model = Vgg16()
