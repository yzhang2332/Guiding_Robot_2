import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import pandas as pd
import numpy as np
import feature_abstract.audio_feature as audio_feature
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DogDataset(Dataset):
    """This is a multimodal dataset including videos, audios, tactile images and pose vectors for a robotic dog project.

    Attributes:
        annotations: The dataframe with each row indicating the quadruple input and the corresponding label.
        audio_dir, video_dir, touch_dir, pose_dir: Directories of the raw data (defined below).
        audio_transform, video_transform, touch_transform, pose_transform: Transforms for the raw data (defined below).
    """

    def __init__(self, annotations_file=None, audio_dir=None, video_dir=None, touch_dir=None, pose_dir=None, audio_transform=None, video_transform=None, touch_transform=None, pose_transform=None):
        """Init dataset with annotations, data locations, transforms.
        
        Arguments:
            annotations_file (string): Path to the csv file with annotations.
            audio_dir (string): Directory with all the audio files.
            video_dir (string): Directory with all the video files.
            touch_dir (string): Directory with all the touchgraph files.
            pose_dir (string): Directory with all the pose vector files.
            audio_transform (callable, optional): Optional transform to be applied on the audio.
            video_transform (callable, optional): Optional transform to be applied on the video.
            touch_transform (callable, optional): Optional transform to be applied on the touch.
            pose_transform (callable, optional): Optional transform to be applied on the pose.
        """

        if annotations_file:
            self.annotations = pd.read_csv(annotations_file)
        else:
            print("The annotations_file is unspecified!")

        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.touch_dir = touch_dir
        self.pose_dir = pose_dir

        # self.audio_transform = audio_transform
        # self.video_transform = video_transform
        # self.touch_transform = touch_transform
        # self.pose_transform = pose_transform

        # self.audio_exist = False
        # self.video_exist = False
        # self.touch_exist = False
        # self.pose_exist = False
        # if audio_dir: 
        #     self.audio_exist = True
        # if video_dir: 
        #     self.video_exist = True
        # if touch_dir: 
        #     self.touch_exist = True
        # if pose_dir: 
        #     self.pose_exist = True
        # self.state = {  'audio': self.audio_exist,
        #                 'video': self.video_exist,
        #                 'touch': self.touch_exist,
        #                 'pose':  self.pose_exist    }


    def __len__(self):
        """Return the length of the dataset."""

        return len(self.annotations)


    def __getitem__(self, idx):
        """Sample the dataset.

        Arguments:
            idx (int): The index of the sampled item.
        """

        audio_path = os.path.join(self.audio_dir, self.annotations.iloc[idx, 0])
        video_path = os.path.join(self.video_dir, self.annotations.iloc[idx, 1])
        touch_path = os.path.join(self.touch_dir, self.annotations.iloc[idx, 2])
        pose_path = os.path.join(self.pose_dir, self.annotations.iloc[idx, 3])

        # audio = io.read_file(audio_path)
        # print(audio_path)
        audio = audio_feature.AudioFeatureExtract(audio_addr=audio_path, debug=False)
        video = io.read_video(video_path)[0] # 读取视频文件，取结果的第一个element (张量 num_frames, height, width, num_channels)

        # touch_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        touch = io.read_image(touch_path, io.ImageReadMode.GRAY) # 转为灰度图
        # touch = touch_trans(touch)
        pose = torch.tensor(np.array(pd.read_csv(pose_path))) # 转为tensor张量

        # self.video_transform = transforms.ToTensorVideo()
        
        video = torch.permute(video, (3, 0, 1, 2)) # 对张量维度重新排序
        # if self.audio_transform:
        #     audio = self.audio_transform(audio)
        # if self.video_transform:
        #     video = self.video_transform(video)
        # if self.touch_transform:
        #     touch = self.touch_transform(touch)
        # if self.pose_transform:
        #     pose = self.pose_transform(pose)

        label = self.annotations.iloc[idx, 4]
        label = torch.FloatTensor([float(item) for item in label.split(sep='_')])
        
        sample = {  'audio': audio.extract_mfcc(),
                    'video': video.float()/ 255.0,
                    'touch': touch.float()/ 255.0,
                    'pose':  pose.float(), 
                    'label': label.float()  }

        # audio_dataset = DogAudio(self.audio_dir)
        # video_dataset = DogVideo(self.video_dir)
        # touch_dataset = DogTouch(self.touch_dir)
        # pose_dataset = DogPose(self.pose_dir)
        # audio = audio_dataset.__getitem__(idx)
        # video = video_dataset.__getitem__(idx)
        # touch = touch_dataset.__getitem__(idx)
        # pose = pose_dataset.__getitem__(idx)

        return sample



# class DogDataLoader(DataLoader):
#     def __init__(self, input_audio_dir, input_video_dir, input_pose_dir, audio_exist = False, video_exist = False, pose_exist = False):

#         self.input_audio_dir = input_audio_dir
#         self.input_video_dir = input_video_dir
#         self.input_psoe_dir = input_pose_dir

#         self.audio_exist = audio_exist
#         self.video_exist = video_exist
#         self.pose_exist = pose_exist

#     def __getitem__(self, i):
#         pass



# class DogAudio(Dataset):
#     def __init__(self, audio_path):
#         self.audio_path = audio_path
#         self.filelist = os.listdir(audio_path)

#     def __len__(self):
#         return len(self.filelist)

#     def __getitem__(self, idx):
#         path = os.path.join(self.audio_path, self.filelist[idx])
#         audio = io.read_file(path)
#         return audio


# class DogVideo(Dataset):
#     def __init__(self, video_path):
#         self.video_path = video_path
#         self.filelist = os.listdir(video_path)

#     def __len__(self):
#         return len(self.filelist)

#     def __getitem__(self, idx):
#         path = os.path.join(self.video_path, self.filelist[idx])
#         video = io.read_video(path)
#         return video


# class DogTouch(Dataset):
#     def __init__(self, touchgraph_path):
#         self.touchgraph_path = touchgraph_path
#         self.filelist = os.listdir(touchgraph_path)

#     def __len__(self):
#         return len(self.filelist)

#     def __getitem__(self, idx):
#         path = os.path.join(self.touchgraph_path, self.filelist[idx])
#         touchgraph = io.read_image(path)
#         return touchgraph


# class DogPose(Dataset):
#     def __init__(self, pose_path):
#         self.pose_path = pose_path
#         self.filelist = os.listdir(pose_path)

#     def __len__(self):
#         return len(self.filelist)

#     def __getitem__(self, idx):
#         path = os.path.join(self.pose_path, self.filelist[idx])
#         pose = io.read_file(path)
#         return pose

