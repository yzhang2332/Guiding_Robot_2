# -*- coding: utf-8 -*-
#!/usr/bin/env python3

#用来提取原始数据的
#需要修改的是bag_time，即rosbag的名称不带.bag，它可以不是日期，但建议是
#它会在上级目录中生成文件夹../dataset/raw/your_bag_name，存放处理后的文件
#它的处理是：
#将图像转成jpg
#将点云转成pcd
#将imu转成csv
#将ecparm(电控参数)转成csv
#所有文件的命名是统一的，按照时间戳对齐，时间间隔是0.1s，即每秒10帧
#去掉了rosbag的头30s和尾30s的数据

import numpy as np
from sensor_msgs.msg import CompressedImage, Imu, PointCloud2
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import ecparm
import cv2
import os
import rospy
from rospy import Time
import csv
import rosbag

for file_name in os.listdir("../dataset/rosbag_raw"):
    if os.path.isfile(os.path.join("../dataset/rosbag_raw", file_name)):
        name_without_extension = os.path.splitext(file_name)[0]
        print(name_without_extension)
        bag_time = name_without_extension

        # 输入rosbag文件路径
        bag_path = '../dataset/rosbag_raw/'+bag_time+".bag"
        output_folder = '../dataset/raw/'+bag_time+'/'

        # 打开rosbag文件
        bag = rosbag.Bag(bag_path)

        # 获取所有消息的时间戳和数据
        image_msgs_leg = []
        image_msgs_0 = []
        image_msgs_2 = []
        image_msgs_4 = []
        image_msgs_6 = []
        imu_msgs = []
        motor_msgs = []
        joystick_msgs = []
        sensor_msgs = []
        # 以0.1秒的间隔提取图像
        interval = 0.1  # seconds
        current_time = bag.get_start_time()+30
        end_time = bag.get_end_time()-30
        cloudpoints_msgs = []


        #rgb0
        output = output_folder + "video/0"
        if not os.path.exists(output):
            os.makedirs(output)
        for topic, msg, t in bag.read_messages(topics=['/usb_cam0/image_raw0/compressed']):
            image_msgs_0.append((t, msg))
        while current_time <= end_time:
            closest_msg = None
            min_time_diff = float('inf')
            for t, msg in image_msgs_0:
                time_diff = abs(t.to_sec() - current_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_msg = msg
            if closest_msg is not None:
                np_arr = np.frombuffer(closest_msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                image_filename = os.path.join(output, f"{(current_time-bag.get_start_time()):.6f}.jpg")
                cv2.imwrite(image_filename, image)
            current_time += interval
        current_time = bag.get_start_time()+30


        # 关闭rosbag文件
        bag.close()