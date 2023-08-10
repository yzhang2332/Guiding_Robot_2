import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from video_swin_transformer import swin_tiny_patch4_window7_224,CustomDataset


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def train(model, train_loader, criterion, optimizer, num_epochs):
    model = model.to(device)
    
    running_loss = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss += criterion(outputs, labels)
        
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss))

    return model
# 在训练函数中，我们传入以下参数：
# model：要训练的模型
# train_loader：用于训练的数据加载器
# criterion：损失函数
# optimizer：优化器
# num_epochs：迭代的总轮数

# 示例用法
if __name__ == '__main__':
    model = swin_tiny_patch4_window7_224(num_classes=2,pretrained='modal_fusion\model\swin_tiny_patch244_window877_kinetics400_1k.pth')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 10

    train_dataset = CustomDataset(file_path='kinetics400_tiny\kinetics_tiny_train_video.txt', root_dir='kinetics400_tiny/train', transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    test_dataset = CustomDataset(file_path='kinetics400_tiny\kinetics_tiny_val_video.txt', root_dir='kinetics400_tiny/val', transform=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    model = train(model, train_loader, criterion, optimizer, num_epochs)

    model.eval()

    # 测试模型
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('准确率: {:.2f}%'.format(accuracy))