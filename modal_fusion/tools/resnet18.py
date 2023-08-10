import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ResBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(outchannel)
#             )
            
#     def forward(self, x):
#         out = self.left(x)
#         out = out + self.shortcut(x)
#         out = nn.functional.relu(out)
        
#         return out

# class ResNet18(nn.Module):
#     def __init__(self, ResidualBlock,input_channels=3 ,output_classes=10):
#         super(ResNet18, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
#         self.fc = nn.Linear(512, output_classes)
        
#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         # print(out.size())
#         out = nn.functional.avg_pool2d(out, 4)
#         # print(out.size(0))
#         out = out.reshape(out.size(0), -1)
#         # out = self.fc(out)
#         return out

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel,kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self,input_channels=3 ,output_classes=100): 
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_classes)

    def make_layer(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    t = ResNet18(input_channels = 1, output_classes = 100).to(device)
    summary(t, (1, 500, 500))