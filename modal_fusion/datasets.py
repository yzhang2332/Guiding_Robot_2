import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import pandas as pd
import numpy as np
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


# 定义数据集类
class DogDataset(Dataset):
    """
    Attributes:
        annotations：包含输入数据和相应标签的数据帧。
        rgb_dir, video_dir, imu_dir, point_cloud_dir, sensor_dir, motor_dir：原始数据的目录。
    """

    def __init__(self, annotations_file=None, rgb_dir=None, video_dir=None, imu_dir=None, point_cloud_dir=None,
                 sensor_dir=None, motor_dir=None):

        if annotations_file:
            self.annotations = pd.read_csv(annotations_file)
        else:
            print("The annotations_file is unspecified!")

        self.rgb_dir = rgb_dir
        self.video_dir = video_dir
        self.imu_dir = imu_dir
        self.point_cloud_dir = point_cloud_dir
        self.sensor_dir = sensor_dir
        self.motor_dir = motor_dir

    def __len__(self):
        """annotations的行数(数据集的长度)"""

        return len(self.annotations)

    def __getitem__(self, idx):
        """
        从数据集中获取样本.

        Arguments:
            idx (int): The index of the sampled item.
        """
        # 路径记得改！！
        rgb_path = os.path.join(self.rgb_dir, self.annotations.iloc[idx, 0])
        video_path = os.path.join(self.video_dir, self.annotations.iloc[idx, 1])
        imu_path = os.path.join(self.imu_dir, self.annotations.iloc[idx, 2])
        point_cloud_path = os.path.join(self.point_cloud_dir, self.annotations.iloc[idx, 3])
        sensor_path = os.path.join(self.sensor_dir, self.annotations.iloc[idx, 4])
        motor_path = os.path.join(self.motor_dir, self.annotations.iloc[idx, 5])  # label

        # audio = io.read_file(audio_path)
        # print(audio_path)
        #  audio = audio_feature.AudioFeatureExtract(audio_addr=audio_path, debug=False)

        rgb = io.read_image(rgb_path, io.ImageReadMode.GRAY)  # 图像
        video = io.read_video(video_path)[0]  # 腿部video
        imu = io.read_file(imu_path)  # imu
        point_cloud = io.read_file(point_cloud_path)  # 点云
        sensor = io.read_file(sensor_path)
        motor = io.read_file(motor_path)

        # touch_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        # touch = io.read_image(touch_path, io.ImageReadMode.GRAY)
        # touch = touch_trans(touch)
        # pose = torch.tensor(np.array(pd.read_csv(pose_path)))

        # self.video_transform = transforms.ToTensorVideo()

        # 将维度顺序调整为(channels, batch_size, height, width)
        video = torch.permute(video, (3, 0, 1, 2))

        label = self.annotations.iloc[idx, 5]
        label = torch.FloatTensor([float(item) for item in label.split(sep='_')])

        sample = {'rgb': rgb.float() / 255.0,
                  'video': video.float() / 255.0,
                  'imu': imu.float(),
                  'point_cloud': point_cloud.float(),
                  'sensor': sensor.float(),
                  'motor': motor.float(),
                  'label': label.float()}

        return sample