#!/usr/bin/env python

import os
import cv2
import pandas as pd

# 自定义排序函数，按照浮点数值排序
def custom_sort_key(filename):
    return float(filename[:-4])  # 去掉文件扩展名后转换为浮点数



def change_jpg(bag_time, id):
    # 指定包含图片文件的文件夹路径
    if id == "leg":
        folder_path = f'../dataset/raw/{bag_time}/video/leg/'
    else:
        folder_path = f'../dataset/raw/{bag_time}/video/{id}/'
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    # 对文件名进行排序，以确保按数值大小重命名
    image_files.sort(key=custom_sort_key)
    # 重命名图片文件
    for i, old_name in enumerate(image_files, start=1):
        new_name = f"{i}.jpg"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    print(f"{bag_time} : {id} 重命名完成！")

def change_pcd(bag_time):
    # 指定包含图片文件的文件夹路径
    folder_path = f'../dataset/raw/{bag_time}/cloudpoints/'
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.pcd')]
    # 对文件名进行排序，以确保按数值大小重命名
    image_files.sort(key=custom_sort_key)
    # 重命名图片文件
    for i, old_name in enumerate(image_files, start=1):
        new_name = f"{i}.pcd"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    print(f"{bag_time} : lidar 重命名完成！")

def change_csv(bag_time, topic):
    # 读取CSV文件
    if topic == "imu":
        csv_file = f'../dataset/raw/{bag_time}/imu/imu_raw.csv'
    elif topic == "motor":
        csv_file = f'../dataset/raw/{bag_time}/ecparm/motor/motor_raw.csv'
    elif topic == "joystick":
        csv_file = f'../dataset/raw/{bag_time}/ecparm/joystick/joystick_raw.csv'
    else:
        csv_file = f'../dataset/raw/{bag_time}/ecparm/sensor/sensor_raw.csv'


    df = pd.read_csv(csv_file)

    df['stamp'] = df.index+1

    # 保存修改后的数据到新的CSV文件
    output_csv_file = csv_file  # 指定新的CSV文件路径
    df.to_csv(output_csv_file, index=False)

    print("已完成！")

def change_bev(bag_time):
    # 指定包含图片文件的文件夹路径
    folder_path = f'../dataset/preprocess/{bag_time}/BEV/'
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    # 对文件名进行排序，以确保按数值大小重命名
    image_files.sort(key=custom_sort_key)
    # 重命名图片文件
    for i, old_name in enumerate(image_files, start=1):
        new_name = f"{i}.jpg"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    print(f"{bag_time} : 重命名完成！")

if __name__ == "__main__":
    for file_name in os.listdir("../dataset/raw"):
        name_without_extension = os.path.splitext(file_name)[0]
        bag_time = name_without_extension

        change_jpg(bag_time, 0)
        change_jpg(bag_time, 2)
        change_jpg(bag_time, 4)
        change_jpg(bag_time, 6)
        change_jpg(bag_time, "leg")
        change_pcd(bag_time)
        change_csv(bag_time,"imu")
        change_csv(bag_time,"motor")
        change_csv(bag_time,"joystick")
        change_csv(bag_time,"sensor")
        change_bev(bag_time)

