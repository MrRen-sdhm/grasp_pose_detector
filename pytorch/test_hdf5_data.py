# Use tensors to speed up loading data onto the GPU during training.

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import torch.multiprocessing
import sys

np.set_printoptions(threshold=np.inf)

#torch.multiprocessing.set_start_method('spawn')

class Net(nn.Module):
    def __init__(self, input_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
#        self.drop2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(50 * 12 * 12, 500)
#        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
#        x = self.pool(F.relu(self.drop2d(self.conv2(x))))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
#        x = self.drop1(x)
        x = self.fc2(x)
        return x


def eval(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    print('Testing the network on the test data ...')

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

    accuracy = 100.0 * float(correct) / float(total)
    print('Accuracy of the network on the test set: %.3f%%' % (
        accuracy))

    return accuracy


class H5Dataset(torchdata.Dataset):
    def __init__(self, file_path, start_idx, end_idx):
        super(H5Dataset, self).__init__()
        with h5py.File(file_path, 'r') as h5_file:
            self.data = torch.from_numpy(np.array(h5_file.get('images')[start_idx : end_idx]))
            print('nparray shape:', np.array(h5_file.get('images')[start_idx : end_idx]).shape)
            self.target = torch.from_numpy(np.array(h5_file.get('labels')[start_idx : end_idx])).to(torch.int32) #.astype('int32'))
            # print(self.target)
            # print(self.data)
            # print(self.data.shape)

        print("Loaded data")

    def __getitem__(self, index):
        image = self.data[index,:,:].to(torch.float32) * 1/256.0
        print(image)
        print(image.shape)
        # Pytorch uses NCHW format
        image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))
        print(image.shape)
        target = self.target[index,:][0]
        return (image, target)

    def __len__(self):
        return self.data.shape[0]


with h5py.File(sys.argv[1], 'r') as db:
    num_test = len(db['images']) # 不可用！这样获得的是整个数据库的大小
    num_test = 1
print('Have', num_test, 'total testing examples')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net(15)
net.to(device)

test_set = H5Dataset(sys.argv[1], 0, num_test)
test_loader = torchdata.DataLoader(test_set, batch_size=256, shuffle=True)
accuracy = eval(net, test_loader, device)

