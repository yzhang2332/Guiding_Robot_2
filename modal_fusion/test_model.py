# import torch
# from tools.video_swin import swin_tiny_patch4_window7_224

# model = swin_tiny_patch4_window7_224(num_classes=1000)

# input = torch.rand(10, 3, 2, 224, 224)  # [batch size, channl, frames, H, W]
# output = model(input)
# print(f'output_shape: {output.shape}')

import torch
import torch.nn as nn
import tools.resnet18 as resnet18
import numpy as np 
import pandas as pd
from datasets import DogDataset
import torchvision
from torch.utils.data import DataLoader



if __name__ == '__main__':

    model = resnet18.ResNet18(input_channels=1, output_classes=100)
    # input = torch.rand(10, 1, 224, 224)  # [batch size, channl, H, W]
    test_dataset = DogDataset(annotations_file='modal_fusion/datasets/test_data/processed_data/annotations.csv', 
                              audio_dir="modal_fusion/datasets/test_data/processed_data/audio", 
                              video_dir="modal_fusion/datasets/test_data/processed_data/video", 
                              touch_dir="modal_fusion/datasets/test_data/processed_data/touch", 
                              pose_dir="modal_fusion/datasets/test_data/processed_data/pose"
                              )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(test_dataloader):
        output = model(sample_batched['pose'].unsqueeze(1))
        print(f'output_shape: {output}')