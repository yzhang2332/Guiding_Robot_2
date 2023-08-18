# 输入：--raw文件夹下的各个模态的文件
# 输出：到--processed文件夹
from moviepy.editor import *
import os
import pandas as pd
import cv2
import open3d


def preprocess(annotations_file=None, original_rgb_dir=None, original_video_dir=None, original_imu_dir=None,
               original_lidar_dir=None, original_sensor_dir=None, original_motor_dir=None,
               rgb_dir=None, video_dir=None, imu_dir=None, lidar_dir=None, sensor_dir=None, motor_dir=None,
               period=6, step=1):
    '''
    Arguments:
        annotations_file (string): Directory of the original annotation file.
        original_rgb_dir (string): Directory of the original RGB image files.
        original_video_dir (string): Directory of the original video files.
        original_imu_dir (string): Directory of the original IMU files.
        original_lidar_dir (string): Directory of the original point cloud files.
        original_sensor_dir (string): Directory of the original sensor files.
        original_motor_dir (string): Directory of the original motor files.
        rgb_dir (string): Directory that new short RGB image files will go to.
        video_dir (string): Directory that new short video files will go to.
        imu_dir (string): Directory that new short IMU files will go to.
        lidar_dir (string): Directory that new short point cloud files will go to.
        sensor_dir (string): Directory that new short sensor files will go to.
        motor_dir (string): Directory that new short motor files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
    '''
    # read original annotations
    init_annotations = pd.read_csv(annotations_file)

    # write new annotation-file head
    with open('modal_fusion/datasets/test_data/processed_data/annotations.csv', 'w') as f:
        f.write('lidar,rgb,video,imu,sensor,motor,label\n')
    f.close()

    # traverse through lines of data instances
    for i in range(init_annotations.shape[0]):

        # a line (an instance)
        data_instance = init_annotations.iloc[i]

        # find indicated files
        lidar_path = os.path.join(original_lidar_dir, data_instance['lidar'])
        rgb_path = os.path.join(original_rgb_dir, data_instance['rgb'])
        video_path = os.path.join(original_video_dir, data_instance['video'])
        imu_path = os.path.join(original_imu_dir, data_instance['imu'])
        sensor_path = os.path.join(original_sensor_dir, data_instance['sensor'])
        motor_path = os.path.join(original_motor_dir, data_instance['motor'])

        # preprocess the original RGB, video, IMU, lidar, sensor ,motor files; generate clips and labels
        lidar_clip_names = preprocessLidar(lidar_path, lidar_dir, period=period, step=step)
        # rgb_clip_names = preprocessRGB(rgb_path, rgb_dir, period=period, step=step)
        video_clip_names = preprocessVideo(video_path, video_dir, period=period, step=step)
        imu_clip_names = preprocessIMU(imu_path, imu_dir, period=period, step=step)
        sensor_clip_names = preprocessSensor(sensor_path, sensor_dir, period=period, step=step)
        motor_clip_names, labels = preprocessMotor(motor_path, motor_dir, period=period, step=step)

        # align the clip_name lists for new annotations  对齐
        clip_name_lists = [lidar_clip_names, video_clip_names, imu_clip_names, sensor_clip_names, motor_clip_names]
        min_length = min([len(lst) for lst in clip_name_lists])
        for j in range(len(clip_name_lists)):
            if len(clip_name_lists[j]) > min_length:
                diff = len(clip_name_lists[j]) - min_length
                del clip_name_lists[j][-diff:]

        # write the new annotation file
        for lidar, rgb, video, imu, sensor, motor, label in zip(*clip_name_lists, labels):
            line = lidar + ',' + rgb + ',' + video + ',' + imu + ',' + sensor + ',' + motor + ',' + '_'.join(
                [str(item) for item in label.values.tolist()]) + '\n'
            with open('modal_fusion/datasets/test_data/processed_data/annotations.csv', 'a') as f:
                f.write(line)
            f.close()


# lidar  ???暂时这么写的，可改，pcd格式
def preprocessLidar(lidar_path=None, to_lidar_dir=None, period=6, step=1, lidar_ext='.pcd'):
    '''Generate subpoint cloud files from original lidar file.

    Arguments:
        lidar_path (string): Directory of the original lidar file.
        to_lidar_dir (string): Directory that new short lidar files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
        lidar_ext (string): The lidar file extension.
    '''
    # read lidar file
    lidar_data = open3d.io.read_lidar(lidar_path)
    # read the lidar file name without extension
    name = os.path.basename(lidar_path).rpartition('.')[0]
    # use a window to slide on the lidar data and separate the lidar parts
    t_start = 0
    t_end = period
    if t_end > len(lidar_data.points):
        print(f'{lidar_path} is too short to slide on!')
        return
    lidar_clip_names = []
    while t_end <= len(lidar_data.points):
        lidar_clip = lidar_data.crop(t_start, t_end)
        name_clip = name + '_' + str(t_start) + '_' + str(t_end)
    open3d.io.write_lidar(os.path.join(to_lidar_dir, name_clip + lidar_ext), lidar_clip)
    lidar_clip_names.append(name_clip + lidar_ext)
    t_start += step
    t_end += step

    return lidar_clip_names


# rgb  jpg格式
# def preprocessRGB(rgb_path=None, to_rgb_dir=None, period=6, step=1, image_ext='.jpg'):
#     '''Generate subRGB images from original RGB image.
#
#     Arguments:
#         rgb_path (string): Directory of the original RGB image.
#         to_rgb_dir (string): Directory that new short RGB image files will go to.
#         period (int): Time interval that the new files span in seconds.
#         step (int): Time shift between two adjacent clips in seconds.
#         image_ext (string): The image file extension.
#     '''
#     # read RGB image
#     rgb_image = cv2.imread(rgb_path)
#     # read the image name without extension
#     name = os.path.basename(rgb_path).rpartition('.')[0]
#     # use a window to slide on the image and separate the image parts
#     t_start = 0
#     t_end = period
#     if t_end > rgb_image.shape[0]:
#         print(f'{rgb_path} is too short to slide on!')
#         return
#     rgb_clip_names = []
#     while t_end <= rgb_image.shape[0]:
#         rgb_clip = rgb_image[t_start:t_end, :]
#         name_clip = name + '_' + str(t_start) + '_' + str(t_end)
#         cv2.imwrite(os.path.join(to_rgb_dir, name_clip + image_ext), rgb_clip)
#         rgb_clip_names.append(name_clip + image_ext)
#         t_start += step
#         t_end += step
#
#     return rgb_clip_names


# video MP4格式
def preprocessVideo(video_path=None, to_video_dir=None, period=6, step=1, video_ext='.mp4'):
    '''Generate subvideos from original video.

    Arguments:
        video_path (string): Directory of the original video.
        to_video_dir (string): Directory that new short video files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
        video_ext (string): The video file extension.
    '''
    # read video
    video = VideoFileClip(video_path)
    # read the video name without extension
    name = os.path.basename(video_path).rpartition('.')[0]
    # use a window to slide on the video and separate the video parts
    t_start = 0
    t_end = period
    if t_end > video.duration:
        print(f'{video_path} is too short to slide on!')
        return
    video_clip_names = []
    while t_end <= video.duration:
        video_clip = video.subclip(t_start, t_end).set_fps(1)
        name_clip = name + '_' + str(t_start) + '_' + str(t_end)
        video_clip.write_videofile(os.path.join(to_video_dir, name_clip + video_ext))
        video_clip_names.append(name_clip + video_ext)
        t_start += step
        t_end += step

    return video_clip_names


# imu csv格式
def preprocessIMU(imu_path=None, to_imu_dir=None, period=6, step=1, imu_ext='.csv'):
    '''Generate subIMU files from original IMU data.

    Arguments:
        imu_path (string): Directory of the original IMU data file.
        to_imu_dir (string): Directory that new subIMU files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
        imu_ext (string): The IMU data file extension.
    '''
    # read IMU data
    imu_data = pd.read_csv(imu_path)
    # read the file name without extension
    name = os.path.basename(imu_path).rpartition('.')[0]
    # use a window to slide on the IMU data and separate the data parts
    t_start = 0
    t_end = period
    if t_end > imu_data.shape[0]:
        print(f'{imu_path} is too short to slide on!')
        return
    imu_clip_names = []
    while t_end <= imu_data.shape[0]:
        imu_clip = imu_data.iloc[t_start:t_end, :]
        name_clip = name + '_' + str(t_start + 1) + '_' + str(t_end)
        imu_clip.to_csv(os.path.join(to_imu_dir, name_clip + imu_ext), index=False)
        imu_clip_names.append(name_clip + imu_ext)
        t_start += step
        t_end += step

    return imu_clip_names


# sensor csv格式
def preprocessSensor(sensor_path=None, to_sensor_dir=None, period=6, step=1, sensor_ext='.csv'):
    '''Generate subSensor files from original sensor data.

    Arguments:
        sensor_path (string): Directory of the original sensor data file.
        to_sensor_dir (string): Directory that new subSensor files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
        sensor_ext (string): The sensor data file extension.
    '''
    # read sensor data
    sensor_data = pd.read_csv(sensor_path)
    # read the file name without extension
    name = os.path.basename(sensor_path).rpartition('.')[0]
    # use a window to slide on the sensor data and separate the data parts
    t_start = 0
    t_end = period
    if t_end > sensor_data.shape[0]:
        print(f'{sensor_path} is too short to slide on!')
        return
    sensor_clip_names = []
    while t_end <= sensor_data.shape[0]:
        sensor_clip = sensor_data.iloc[t_start:t_end, :]
        name_clip = name + '_' + str(t_start + 1) + '_' + str(t_end)
        sensor_clip.to_csv(os.path.join(to_sensor_dir, name_clip + sensor_ext), index=False)
        sensor_clip_names.append(name_clip + sensor_ext)
        t_start += step
        t_end += step

    return sensor_clip_names


# motor、label  csv格式
# 原始文件是p1_motor_raw.csv中第一列是frame，假设现在切割出来的是p1_1_m_motor.csv，标签是p1_motor_raw.csv中frame是m+1时候的motor值
def preprocessMotor(motor_path=None, to_motor_dir=None, period=6, step=1):
    '''Generate submotor data from original motor csv file.

    Arguments:
        motor_path (string): Directory of the original motor file.
        to_motor_dir (string): Directory that new short motor files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
    '''
    # Read the motor file
    df = pd.read_csv(motor_path)

    # Get the name of the motor file without extension
    name = os.path.basename(motor_path).rpartition('.')[0]

    # Use a window to slide on the motor file and separate out submotor data
    t_start = 0
    t_end = period
    if t_end > df.shape[0]:
        print(f'{motor_path} is too short to slide on!')
        return
    motor_clip_names, labels = [], []
    while t_end <= df.shape[0]:
        motor_clip = df.loc[t_start:t_end]
        name_clip = name + '_' + str(t_start) + '_' + str(t_end) + '.csv'
        motor_clip.to_csv(os.path.join(to_motor_dir, name_clip), index=False)
        motor_clip_names.append(name_clip)

        # label为当前子motor数据的结束帧号加1
        label_index = t_end + 1
        if label_index >= df.shape[0]:
            label = df.iloc[-1]['motor']
        else:
            label = df.loc[label_index]['motor']
        labels.append(label)

        t_start += step
        t_end += step

    return motor_clip_names, labels


if __name__ == "__main__":
    preprocess(annotations_file='modal_fusion/datasets/test_data/raw_data/original_annotations.csv',
               original_lidar_dir='modal_fusion/datasets/test_data/raw_data/raw_lidar',
               # original_rgb_dir='modal_fusion/datasets/test_data/raw_rgb',
               original_video_dir='modal_fusion/datasets/test_data/raw_data/raw_video',
               original_imu_dir='modal_fusion/datasets/test_data/raw_data/raw_imu',
               original_sensor_dir='modal_fusion/datasets/test_data/raw_data/raw_sensor',
               original_motor_dir='modal_fusion/datasets/test_data/raw_data/raw_motor',
               lidar_dir='modal_fusion/datasets/test_data/processed_data/lidar',
               # rgb_dir='modal_fusion/datasets/test_data/processed_data/rgb',
               video_dir='modal_fusion/datasets/test_data/processed_data/video',
               imu_dir='modal_fusion/datasets/test_data/processed_data/imu',
               sensor_dir='modal_fusion/datasets/test_data/processed_data/sensor',
               motor_dir='modal_fusion/datasets/test_data/processed_data/motor',
               )
