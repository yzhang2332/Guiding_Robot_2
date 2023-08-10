import tools.resnet101 as resnet101
import tools.resnet18 as resnet18
# import tools.video_swin as video_swin
import feature_abstract.audio_feature as audio_feature

import tools.fusion_transformer as fusion_transformer
from model.video_swin_transformer import swin_tiny_patch4_window7_224
import torch
import torch.nn as nn
from torchsummary import summary


# 这个名字很奇怪
class MultiSwin(nn.Module):
    def __init__(self, input_channels, output_classes, pretrained=None, final_output_classes=17,debug=False):
        super(MultiSwin, self).__init__()
        self.debug = debug
        self.audio_model = resnet18.ResNet18(
            input_channels=input_channels, output_classes=output_classes)
        self.video_model = swin_tiny_patch4_window7_224(num_classes=output_classes, pretrained=pretrained)
        self.touch_model = resnet18.ResNet18(
            input_channels=input_channels, output_classes=output_classes)
        self.pose_model = resnet18.ResNet18(
            input_channels=input_channels, output_classes=output_classes)
        # self.bn = nn.BatchNorm1d(output_classes)
        self.ln = nn.LayerNorm(output_classes)
        self.relu = nn.ReLU(inplace=True)
        # self.fusion_model = resnet18.ResNet18(
        #     input_channels=1, output_classes=final_output_classes)

        self.fusion_model = fusion_transformer.Fusion_transformer(
            input_channels=output_classes*4, output_channels=final_output_classes, tensor_len=1)  

# test only for model not properlly for actual use
    def forward(self, audio, video, touch, pose):
        # audio = audio_feature.AudioFeatureExtract(audio_addr=audio, debug=False)
        audio_out = self.ln(self.audio_model(audio))
        video_out = self.ln(self.video_model(video))
        touch_out = self.ln(self.touch_model(touch))
        pose_out = self.ln(self.pose_model(pose))

        # print(audio_out.shape,video_out.shape,touch_out.shape,pose_out.shape)
        fusion_in = torch.cat(
            (audio_out, video_out, touch_out, pose_out), dim=1) # 各个模态的张量拼接到一起
        fusion_in = self.relu(fusion_in)
        # print(fusion_in.shape)
        fusion_out = self.fusion_model(fusion_in)
        # fusion_out = self.fusion_model(fusion_in.unsqueeze(1).unsqueeze(1))

        if self.debug:
            print(audio_out,'\n',video_out,'\n',touch_out,'\n',pose_out,'\n',fusion_in,'\n',fusion_out)

        return fusion_out
