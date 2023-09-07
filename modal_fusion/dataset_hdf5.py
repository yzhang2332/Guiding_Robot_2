#!/usr/bin/env python
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import pandas as pd
import numpy as np
import pandas as pd
pd.set_option('display.width',None)#设置数据展示宽度(好像没用)
# Ignore warnings
import warnings
import csv
import cv2
import ast
from sklearn import preprocessing
import h5py

warnings.filterwarnings("ignore")


# 定义数据集类
class DogDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.group_names = None  # 将数据集的初始化延迟到__getitem__中

    def __len__(self):
        if self.group_names is None:
            # 打开HDF5文件并获取组名列表
            with h5py.File(self.hdf5_file, 'r') as file:
                self.group_names = list(file.keys())
        return len(self.group_names)


    def __getitem__(self, index):
        if self.group_names is None:
            # 打开HDF5文件并获取组名列表
            with h5py.File(self.hdf5_file, 'r') as file:
                self.group_names = list(file.keys())

        group_name = self.group_names[index]
        with h5py.File(self.hdf5_file, 'r') as file:
            group = file[group_name]

            bev = torch.tensor(group['bev'][()])
            video = torch.tensor(group['video'][()])
            imu = torch.tensor(group['imu'][()])
            sensor = torch.tensor(group['sensor'][()])
            motor = torch.tensor(group['motor'][()])
            label = torch.tensor(group['label'][()])
            class_value = torch.tensor(group['class'][()])

        sample = {
            'bev': bev,
            'video': video,
            'imu': imu,
            'sensor': sensor,
            'motor': motor,
            'label': label,
            'class': class_value
        }

        return sample

if __name__ == "__main__":
    hdf5_file = "data.hdf5"  # 将文件名替换为你的HDF5文件路径
    dataset = DogDataset(hdf5_file)

    # test：打印第一个样本
    sample = dataset[0]
    print(sample)