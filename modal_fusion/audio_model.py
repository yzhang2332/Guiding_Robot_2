import torch
import torch.nn as nn
from torchsummary import summary

from tools.resnet101 import ResNet101
from tools.resnet18 import ResNet18,ResBlock

from feature_abstract import audio_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    audiotest = audio_feature.AudioFeatureExtract(audio_addr="modal_fusion/datasets/wav.wav",debug=False)
    mfccFeatures = audiotest.extract_mfcc()

    # print(torch.tensor(mfccFeatures).unsqueeze(0).shape)
    t = ResNet101(input_channels = 1, output_classes = 100).to(device)
    s = ResNet18(ResidualBlock=ResBlock,input_channels=1 ,output_classes=100).to(device)
    
    summary(s, (1, 500, 500))