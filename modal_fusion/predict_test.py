#!/usr/bin/env python

import torch
import multi_swin
from dataset_hdf5 import DogDataset
from torch.utils.data import DataLoader
from collections import Counter

class Args:
    def __init__(self) -> None:
        self.batch_size = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = "./model/model.pth"
args = Args()


if __name__ == "__main__":
    # 创建数据集和数据加载器
    dataset = DogDataset("data.hdf5")
    data_loader = DataLoader(dataset, args.batch_size, shuffle=False)

    # 加载预训练模型
    model = torch.load(args.model_path)
    model.eval()

    # 遍历数据集并进行预测
    predictions = []
    for sample_batch in data_loader:
        with torch.no_grad():
            bev = sample_batch['bev']
            video = sample_batch['video']
            imu = sample_batch['imu']
            sensor = sample_batch['sensor']
            motor = sample_batch['motor']

            # 进行预测
            outputs = model(sample_batch['bev'].to(args.device), sample_batch['video'].to(args.device),
                            sample_batch['imu'].to(args.device), sample_batch['sensor'].to(args.device),
                            sample_batch['motor'].to(args.device))
            


            print(f"output : {outputs}")
            print(f"label : {sample_batch['label']}")