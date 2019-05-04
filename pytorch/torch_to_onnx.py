import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class Net(nn.Module):
    def __init__(self, input_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 12 * 12, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50*12*12)
        #x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_channels = int(sys.argv[3])
net = Net(input_channels)
net.load_state_dict(torch.load(sys.argv[1]))
print(net)

dummy_input = torch.randn(1, input_channels, 60, 60)
torch.onnx.export(net, dummy_input, sys.argv[2], verbose=True)
