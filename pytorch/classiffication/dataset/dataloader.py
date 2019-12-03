from torch.utils.data import Dataset
import torch.utils.data as torchdata
from torchvision import transforms as T 
from config import config
from PIL import Image 
# from dataset.aug import *
from itertools import chain 
from glob import glob
from tqdm import tqdm
import random 
import numpy as np 
import pandas as pd 
import os
import torch 
import h5py

#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#2.define dataset
class H5Dataset(torchdata.Dataset):
    def __init__(self, file_path, start_idx, end_idx):
        super(H5Dataset, self).__init__()
        with h5py.File(file_path, 'r') as h5_file:
            self.data = torch.from_numpy(np.array(h5_file.get('images')[start_idx : end_idx]))
            self.target = torch.from_numpy(np.array(h5_file.get('labels')[start_idx : end_idx])).to(torch.int32).long()
        print("Loaded dataset:", file_path)

    def __getitem__(self, index):
        image = self.data[index,:,:].to(torch.float32) * 1/256.0
        # Pytorch uses NCHW format
        image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))    
        target = self.target[index,:][0]
        return (image, target)

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

def get_files(root,mode):
    #for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test": 
        #for train and val       
        all_data_path,labels = [],[]
        image_folders = list(map(lambda x:root+x,os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x:glob(x+"/*"),image_folders))))
        print("loading train dataset")
        for file in tqdm(all_images):
            all_data_path.append(file)
            labels.append(int(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    else:
        print("check the mode please!")
    
