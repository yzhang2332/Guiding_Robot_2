import torch
from torch.utils.data import Dataset, DataLoader

#随机发布符合shape的random数据

class CreateDataset(Dataset):
    def __init__(self, bev, video, imu, sensor, motor, label):
        self.bev = bev
        self.video = video
        self.imu = imu
        self.sensor = sensor
        self.motor = motor
        self.label = label
        self.length = len(bev)

        print(self.bev.shape)
        print(self.bev[0].shape)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {
            'bev': self.bev[idx],
            'video': self.video[idx],
            'imu': self.imu[idx],
            'sensor': self.sensor[idx],
            'motor': self.motor[idx],
            'label': self.label[idx]
        }
        return sample

if __name__ == "__main__":
    bev = torch.rand(114514, 1, 60, 200, 200) # batch, channel, depth(time), height, weight
    video = torch.rand(114514, 60, 480, 640, 3) # batch, channel, depth(time), height, weight
    imu = torch.rand(114514, 1, 60, 10) # batch, 占位, frames(6 sec), imu_data
    sensor = torch.rand(114514, 1, 60, 4) # batch, 占位, frames(6 sec), sensor_data
    motor = torch.rand(114514, 1, 60, 2) # batch, 占位, frames(6 sec), motor_data
    label = torch.rand(114514, 20) # batch, frames(1 sec), label

    test_dataset = CreateDataset(bev, video, imu, sensor, motor, label)