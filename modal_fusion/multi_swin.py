import tools.resnet18 as resnet18
import tools.fusion_transformer as fusion_transformer
from model.video_swin_transformer import swin_tiny_patch4_window7_224
import torch
import torch.nn as nn
#from torchsummary import summary


#特征提取
class MultiSwin(nn.Module):
    def __init__(self, input_channels, output_classes, final_output_classes=2,pretrained=None, debug=False):
        """
        rgb:resnet
        imu:resnet
        sensor:resnet
        point_cloud:?
        video:swin transformer
        motor:resnet
        """
        super(MultiSwin, self).__init__()
        self.debug = debug
        self.rgb_model= resnet18.ResNet18(input_channels=input_channels, output_classes=output_classes)
        self.imu_model= resnet18.ResNet18(input_channels=input_channels, output_classes=output_classes)
        self.sensor_model= resnet18.ResNet18(input_channels=input_channels, output_classes=output_classes)
        self.point_cloud_model= resnet18.ResNet18(input_channels=input_channels, output_classes=output_classes)
        self.video_model = swin_tiny_patch4_window7_224(num_classes=output_classes, pretrained=pretrained)
        self.motor_model=resnet18.ResNet18(input_channels=input_channels, output_classes=output_classes)
        # self.bn = nn.BatchNorm1d(output_classes)
        self.ln = nn.LayerNorm(output_classes)
        self.relu = nn.ReLU(inplace=True)
        # self.fusion_model = resnet18.ResNet18(
        #     input_channels=1, output_classes=final_output_classes)

        # self.fusion_model = fusion_transformer.Fusion_transformer(
        #     input_channels=output_classes*4, output_channels=final_output_classes, tensor_len=1)
        self.fusion_model = fusion_transformer.Fusion_transformer(
            input_channels=output_classes * 6, output_channels=final_output_classes, tensor_len=1)


    # test only for model not properlly for actual use
#     def forward(self, audio, video, touch, pose):
#         # audio = audio_feature.AudioFeatureExtract(audio_addr=audio, debug=False)
#         audio_out = self.ln(self.audio_model(audio))
#         video_out = self.ln(self.video_model(video))
#         touch_out = self.ln(self.touch_model(touch))
#         pose_out = self.ln(self.pose_model(pose))
#
#         # print(audio_out.shape,video_out.shape,touch_out.shape,pose_out.shape)
#         fusion_in = torch.cat(
#             (audio_out, video_out, touch_out, pose_out), dim=1)
#         fusion_in = self.relu(fusion_in)
#         # print(fusion_in.shape)
#         fusion_out = self.fusion_model(fusion_in)
#         # fusion_out = self.fusion_model(fusion_in.unsqueeze(1).unsqueeze(1))
#
#         if self.debug:
#             print(audio_out,'\n',video_out,'\n',touch_out,'\n',pose_out,'\n',fusion_in,'\n',fusion_out)
#
#         return fusion_out
    def forward(self, rgb, video, imu,point_cloud,sensor,motor):
        rgb_out = self.ln(self.rgb_model(rgb))
        video_out = self.ln(self.video_model(video))
        imu_out = self.ln(self.imu_model(imu))
        point_cloud_out = self.ln(self.point_cloud_model(point_cloud))
        sensor_out=self.ln(self.sensor_model(sensor))
        motor_out=self.ln(self.motor_model(motor))

        fusion_in = torch.cat((rgb_out, video_out, imu_out, point_cloud_out, sensor_out,motor_out),dim=1)
        fusion_in = self.relu(fusion_in)
        fusion_out = self.fusion_model(fusion_in)

        velocity = fusion_out[:, 0]  # 速度的输出
        heading = fusion_out[:, 1]  # 航向角的输出

        return velocity, heading