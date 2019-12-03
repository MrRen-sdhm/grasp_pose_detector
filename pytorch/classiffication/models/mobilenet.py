import torchvision
import torch.nn.functional as F 
from torch import nn
from config import config


def get_mobilenet(input_channels=15):
    class MobileNet(nn.Module):
        def __init__(self, input_channels):
            super(MobileNet, self).__init__()

            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )

            def conv_dw(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),
        
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )

            self.model = nn.Sequential(
                conv_bn(input_channels, 32, 2), 
                conv_dw(32,  64, 1),
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
                nn.AvgPool2d(2),
            )
            self.fc = nn.Linear(1024, 2)

        def forward(self, x):
            x = self.model(x)
            x = x.view(-1, 1024)
            x = self.fc(x)
            return x

    net = MobileNet(input_channels)
    return net
