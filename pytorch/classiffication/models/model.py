import torchvision
import torch.nn.functional as F
from torch import nn
from config import config


def generate_model():
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.classifier = nn.Linear(pretrained_model.classifier.in_features, config.num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

            self.features = pretrained_model.features
            self.layer1 = pretrained_model.features._modules['denseblock1']
            self.layer2 = pretrained_model.features._modules['denseblock2']
            self.layer3 = pretrained_model.features._modules['denseblock3']
            self.layer4 = pretrained_model.features._modules['denseblock4']

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
            out = F.sigmoid(self.classifier(out))
            return out

    return DenseModel(torchvision.models.densenet169(pretrained=True))


def get_net():
    # return MyModel(torchvision.models.resnet101(pretrained = True))
    model = torchvision.models.resnet152(pretrained=True)
    # for param in model.parameters():
    #    param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2048, config.num_classes)
    return model


def get_lenet(input_channels=15):
    class LeNet(nn.Module):
        def __init__(self, input_channels):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 20, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(20, 50, 5)
    #        self.drop2d = nn.Dropout2d(p=0.2)
            self.fc1 = nn.Linear(50 * 12 *12, 500) # 36450 # 50 * 12 *12
            # self.drop1 = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(500, 2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
    #        x = self.pool(F.relu(self.drop2d(self.conv2(x))))
            x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
            x = F.relu(self.fc1(x))
            # x = self.drop1(x)
            x = self.fc2(x)
            return x

    net = LeNet(input_channels)
    return net
