"""加载一个预训练模型，用于视频级别的特征提取"""

import torch
import torch.nn as nn
from torchsummary import summary

# 加载video_swin_tiny_patch4_window7_224.pth

class swin_tiny_patch4_window7_224(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 怎么加载本地的预训练模型
        # model = dict(
        # cls_head=dict(
        # type='TSNHead',
        # num_classes=101  # change from 400 to 101
        # ))
        self.checkpoint = torch.load('modal_fusion\model\swin_tiny_patch244_window877_kinetics400_1k.pth')
        model = copy.deepcopy(checkpoint)
        # self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    checkpoint = torch.load('modal_fusion\model\swin_tiny_patch244_window877_kinetics400_1k.pth')

    # model = swin_tiny_patch4_window7_224(num_classes=1000)
    # print(checkpoint['state_dict'])
    # input = torch.rand(10, 3, 2, 224, 224)  # [batch size, channl, frames, H, W]
    # output = checkpoint(input)
    # print(f'output_shape: {output.shape}') 
    for name, param in checkpoint['state_dict'].items():
        if name=='cls_head.fc_cls.weight':
            param.data = torch.ones(101, 2048)
            checkpoint['state_dict'][name] = param
        if name=='cls_head.fc_cls.bias':
            param.data = torch.ones(101)
            checkpoint['state_dict'][name] = param
        print('{},size:{}'.format(name, param.data.size()))
    # input = torch.rand(10, 3, 2, 224, 224)  # [batch size, channl, frames, H, W]
    # output = model(input)
    # print(f'output_shape: {output.shape}')
    # summary(model, (3, 2, 224, 224))