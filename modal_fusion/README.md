# getdata说明   
本文件夹中的程序用于从录制的rosbag中提取原始数据  
注意：认为rosbag的存放地址和提取原始的存放地址都是固定的  
  rosbag存放：../dataset/rosbag_raw/  
  提取数据存放： ../dataset/raw/{rosbagname}/  
  BEV数据存放： ../dataset/preprocess/{rosbagname}/  
## 1. dataset.py 
### 说明  
有两个功能。  
1.如果是使用从rosbag中提取的图片和文件作为数据集（需按照指定文件夹目录）  
可在train_batch16.py中引用该模块创建数据集  
2.可将rosbag提取数据创建为hpf5文件  
注意：生成的hpf5针对类别不平衡问题引入了欠采样和过采样，使用前需修改

### 使用
from dataset import DogDataset
python dataset.py #生成hpf5文件


## 2. dataset_simulator.py
### 说明  
生成符合形状的随机tensor数据

### 使用
from dataset_simulator import CreateDataset

## 3. dataset_hdf5.py
### 说明  
引用hdf5文件创建pytorch的dataset

### 使用
from dataset_hdf5 import DogDataset

### 4. train_batch16.py
### 说明  
训模型  
训的是五分类任务，将小车分为停止、前进、后退、右转、左转5类，label是一个(batch,frame,classes_num)的tensor，是一个one-hot矩阵 
loss使用的是focal loss  
learning rate 会自动调整，若触发early stop则将learning rate变为原来的1/5但不stop，重复直到小于某个值停止  
epoch 100  
batch size 16  
tensorboard记录训练集的loss和accuracy和测试集的loss和accuracy，tensorboard文件存在./modal_fusion/logs/train_log  
model存在./model里，整个模型的model和checkpoint都存，每训完一个epoch都存一次  
模型结构是resnet18提取csv文件数据特征，resnet3d50提取BEV图特征，video_swim_transfomer提取腿部rgb视频特征，然后把所有特征直接拼接投入transfomer，预测，具体模型结构可以从该文件回溯看
### 使用
python train_batch16.py

### 5. predict_test.py
### 说明  
拿模型预测，print模型的output和label
### 使用
python predict_test.py