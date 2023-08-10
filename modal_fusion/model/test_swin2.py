import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from video_swin_transformer import swin_tiny_patch4_window7_224,CustomDataset

from torchvision import transforms
from torchvision.models import video

if __name__ == '__main__':
    # 加载预训练模型
    model = swin_tiny_patch4_window7_224(num_classes=400,pretrained='modal_fusion\model\swin_tiny_patch244_window877_kinetics400_1k.pth')

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # 设置transform用于数据预处理
    transform = transforms.Compose([
  
    ])

    # 加载训练集
    train_dataset = torchvision.datasets.Kinetics(
        root='tiny-Kinetics-400',  # kenetics400-tiny数据集的路径
        frames_per_clip=16,
        step_between_clips=1,
        transform=transform,
        frame_rate=30,
        split='train'  # 使用训练集的数据
    )

    # 加载测试集
    # test_dataset = torchvision.datasets.Kinetics(
    #     root='tiny-Kinetics-400',  # kenetics400-tiny数据集的路径
    #     frames_per_clip=16,
    #     step_between_clips=1,
    #     transform=transform,
    #     frame_rate=30,
    #     split='val'  # 使用验证集的数据
    # )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8
    )

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=8
    # )

    # 将模型设置为训练模式
    model.train()

    num_epochs = 10
    # 训练模型
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, audio, labels = data
            print(inputs.shape)
            torch.permute(inputs, (0, 2, 1, 3, 4))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # 将模型设置为评估模式
    model.eval()

    # 测试模型
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, audio, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('准确率: {:.2f}%'.format(accuracy))