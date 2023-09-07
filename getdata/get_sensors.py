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

        #rgbd
        output = output_folder + "video/leg"
        if not os.path.exists(output):
            os.makedirs(output)
        for topic, msg, t in bag.read_messages(topics=['/usb_cam/image_raw/compressed']):
            image_msgs_leg.append((t, msg))
        while current_time <= end_time:
            closest_msg = None
            min_time_diff = float('inf')
            for t, msg in image_msgs_leg:
                time_diff = abs(t.to_sec() - current_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_msg = msg
            if closest_msg is not None:
                np_arr = np.frombuffer(closest_msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                image_filename = os.path.join(output, f"{(current_time-bag.get_start_time()):.6f}.jpg") #filp
                image = cv2.flip(image, 0)  # 参数0表示垂直翻转
                cv2.imwrite(image_filename, image)
            current_time += interval
        current_time = bag.get_start_time()+30

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

        #rgb2
        output = output_folder + "video/2"
        if not os.path.exists(output):
            os.makedirs(output)
        for topic, msg, t in bag.read_messages(topics=['/usb_cam2/image_raw2/compressed']):
            image_msgs_2.append((t, msg))
        while current_time <= end_time:
            closest_msg = None
            min_time_diff = float('inf')
            for t, msg in image_msgs_2:
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

        #rgb4
        output = output_folder + "video/4"
        if not os.path.exists(output):
            os.makedirs(output)
        for topic, msg, t in bag.read_messages(topics=['/usb_cam4/image_raw4/compressed']):
            image_msgs_4.append((t, msg))
        while current_time <= end_time:
            closest_msg = None
            min_time_diff = float('inf')
            for t, msg in image_msgs_4:
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

        #rgb6
        output = output_folder + "video/6"
        if not os.path.exists(output):
            os.makedirs(output)
        for topic, msg, t in bag.read_messages(topics=['/usb_cam6/image_raw6/compressed']):
            image_msgs_6.append((t, msg))
        while current_time <= end_time:
            closest_msg = None
            min_time_diff = float('inf')
            for t, msg in image_msgs_6:
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

        #imu
        output = output_folder+'imu/'
        csv_path = output + 'imu_raw.csv'
        if not os.path.exists(output):
            os.makedirs(output)
        with open(csv_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['stamp', 'x', 'y', 'z', 'w', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
            for topic, msg, t in bag.read_messages(topics=['/imu/data']):
                imu_msgs.append((t, msg))
            while current_time <= end_time:
                closest_msg = None
                min_time_diff = float('inf')
                for t, msg in imu_msgs:
                    time_diff = abs(t.to_sec() - current_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_msg = msg
                if closest_msg is not None:
                    row = [
                        f"{(current_time-bag.get_start_time()):.6f}",
                        closest_msg.orientation.x, closest_msg.orientation.y, closest_msg.orientation.z, closest_msg.orientation.w,
                        closest_msg.angular_velocity.x, closest_msg.angular_velocity.y, closest_msg.angular_velocity.z,
                        closest_msg.linear_acceleration.x, closest_msg.linear_acceleration.y, closest_msg.linear_acceleration.z
                    ]
                    csv_writer.writerow(row)
                current_time += interval
        current_time = bag.get_start_time()+30

        #motor
        output = output_folder+'ecparm/motor/'
        csv_path = output + 'motor_raw.csv'
        if not os.path.exists(output):
            os.makedirs(output)
        with open(csv_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['stamp', 'm1', 'm2'])
            for topic, msg, t in bag.read_messages(topics=['/ecparm']):
                motor_msgs.append((t, msg))
            while current_time <= end_time:
                closest_msg = None
                min_time_diff = float('inf')
                for t, msg in motor_msgs:
                    time_diff = abs(t.to_sec() - current_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_msg = msg
                if closest_msg is not None:
                    row = [
                        f"{(current_time-bag.get_start_time()):.6f}",
                        closest_msg.motor.speed1, closest_msg.motor.speed2
                    ]
                    csv_writer.writerow(row)
                current_time += interval
        current_time = bag.get_start_time()+30

        #joystick
        output = output_folder+'ecparm/joystick/'
        csv_path = output + 'joystick_raw.csv'
        if not os.path.exists(output):
            os.makedirs(output)
        with open(csv_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['stamp', 'x', 'y'])
            for topic, msg, t in bag.read_messages(topics=['/ecparm']):
                joystick_msgs.append((t, msg))
            while current_time <= end_time:
                closest_msg = None
                min_time_diff = float('inf')
                for t, msg in joystick_msgs:
                    time_diff = abs(t.to_sec() - current_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_msg = msg
                if closest_msg is not None:
                    row = [
                        f"{(current_time-bag.get_start_time()):.6f}",
                        closest_msg.joystick.joystick_x, closest_msg.joystick.joystick_y
                    ]
                    csv_writer.writerow(row)
                current_time += interval
        current_time = bag.get_start_time()+30

        #sensor
        output = output_folder+'ecparm/sensor/'
        csv_path = output + 'sensor_raw.csv'
        if not os.path.exists(output):
            os.makedirs(output)
        with open(csv_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['stamp', 'f1', 'f2', 'f3', 'f4'])
            for topic, msg, t in bag.read_messages(topics=['/ecparm']):
                sensor_msgs.append((t, msg))
            while current_time <= end_time:
                closest_msg = None
                min_time_diff = float('inf')
                for t, msg in sensor_msgs:
                    time_diff = abs(t.to_sec() - current_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_msg = msg
                if closest_msg is not None:
                    row = [
                        f"{(current_time-bag.get_start_time()):.6f}",
                        closest_msg.sensor.sensor_1, closest_msg.sensor.sensor_2, closest_msg.sensor.sensor_3, closest_msg.sensor.sensor_4
                    ]
                    csv_writer.writerow(row)
                current_time += interval
        current_time = bag.get_start_time()+30

        #cloud_points
        output = output_folder+'cloudpoints/'
        if not os.path.exists(output):
            os.makedirs(output)
        for topic, msg, t in bag.read_messages(topics=['/points_raw']):
            cloudpoints_msgs.append((t, msg))
        while current_time <= end_time:
            closest_msg = None
            min_time_diff = float('inf')
            for t, msg in cloudpoints_msgs:
                time_diff = abs(t.to_sec() - current_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_msg = msg
            if closest_msg is not None:
                points = []
                for data in pc2.read_points(closest_msg,field_names=("x", "y", "z"), skip_nans=True):
                    points.append([data[0], data[1], data[2]])
                point_cloud = np.array(points)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                o3d.io.write_point_cloud(f"{output}{(current_time-bag.get_start_time()):.6f}.pcd", pcd)
            current_time += interval

        # 关闭rosbag文件
        bag.close()