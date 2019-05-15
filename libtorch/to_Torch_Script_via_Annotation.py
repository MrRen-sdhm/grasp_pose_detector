# reference:
# https://pytorch.org/tutorials/advanced/cpp_export.html
# https://pytorch.org/docs/stable/jit.html

import h5py
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as torchdata
import torch.nn.functional as F


class H5Dataset(torchdata.Dataset):
    def __init__(self, file_path, start_idx, end_idx):
        super(H5Dataset, self).__init__()
        with h5py.File(file_path, 'r') as h5_file:
            self.data = torch.from_numpy(np.array(h5_file.get('images')[start_idx : end_idx]))
            self.target = torch.from_numpy(np.array(h5_file.get('labels')[start_idx : end_idx])).to(torch.int32) #.astype('int32'))
        print("Loaded data")

    def __getitem__(self, index):
        image = self.data[index,:,:].to(torch.float32) * 1/256.0
        # Pytorch uses NCHW format
        image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))
        target = self.target[index,:][0]
        return (image, target)

    def __len__(self):
        return self.data.shape[0]


class Net(torch.jit.ScriptModule):
    def __init__(self, input_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
#        self.drop2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(50 * 12 * 12, 500)
#        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(500, 2)


    @torch.jit.script_method
    def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
        #        x = self.pool(F.relu(self.drop2d(self.conv2(x))))
      x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
      x = F.relu(self.fc1(x))
        #        x = self.drop1(x)
      x = self.fc2(x)
      return x


use_cuda = True
input_channels = 15
pre_trained_model = '/home/sdhm/Projects/gpd2/pytorch/model_lenet_new_15ch_87.34.pwf'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net(input_channels)
# load the pre trained net work
net.load_state_dict(torch.load(pre_trained_model))
print(net)

if use_cuda:
    # my_script_module = net.to(device) # to不被支持！
    my_script_module = net.cuda()
    my_script_module.save('gpu.pt')
else:
    my_script_module = net
    my_script_module.save('cpu.pt')

# test
test_set = H5Dataset('/home/sdhm/Projects/gpd2/models/new/15channels/test.h5', 0, 1)
test_loader = torchdata.DataLoader(test_set, batch_size=64, shuffle=True)
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        print(inputs)
        print(inputs.shape)
        if use_cuda:
            inputs = inputs.to(device)
        output_script = my_script_module(inputs)
        output = net(inputs)
        print(output_script)
        print(output)
