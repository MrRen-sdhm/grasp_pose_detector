import sys
import h5py
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from models.model import *
from models.mobilenet import *
from models.mobilenet_v2 import *
from models.resnet import *
from models.densenet import *
from models.inception import *


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


class DefaultConfigs(object):
    # 1.string parameters
    model_name = "lenet"
    test_data = "/home/sdhm/Projects/gpd2/models/gpd_dataset/12channels/3obj/test.h5"
    pre_trained = "/home/sdhm/Projects/gpd2/pytorch/classiffication/checkpoints/best_model/lenet/3obj/model_best_89.52.pt"
    save_prefix = model_name + "_3obj_"

    # 2.numeric parameters
    img_channels = 12
    img_size = 60

    # 3.bool parameters
    use_cuda = False


config = DefaultConfigs()

device = torch.device("cuda:0" if config.use_cuda else "cpu")

# An instance of your model.
if config.model_name is "lenet":
    model = get_lenet(config.img_channels)
elif config.model_name is "mobilenet":
    model = get_mobilenet(config.img_channels)
elif config.model_name is "mobilenet_v2":
    model = mobilenet_v2(config.img_channels)
elif config.model_name is "resnet":
    model = resnet18()
elif config.model_name is "densenet":
    model = densenet121()
elif config.model_name is "inception_v3":
    model = inception_v3()


print(model)
model.load_state_dict(torch.load(config.pre_trained)["state_dict"])

if config.use_cuda:
    model.to(device)

# An example input you would normally provide to your model's forward() method.

example = torch.rand(1, config.img_channels, config.img_size, config.img_size)
if config.use_cuda:
    example = example.cuda()
    print(example.shape)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
if config.use_cuda:
    traced_script_module.save(config.save_prefix + "gpu.pt")
else:
    traced_script_module.save(config.save_prefix + "cpu.pt")

# test
test_set = H5Dataset(config.test_data, 0, 1)
test_loader = torchdata.DataLoader(test_set, batch_size=64, shuffle=True)
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        print(inputs)
        print(inputs.shape)
        if config.use_cuda:
            inputs = inputs.to(device)
        output_script = traced_script_module(inputs)
        output = model(inputs)
        print(output_script)
        print(output)

