import os
import cv2

# 自定义排序函数，按照浮点数值排序
def custom_sort_key(filename):
    return float(filename[:-4])  # 去掉文件扩展名后转换为浮点数

for file_name in os.listdir("../dataset/rosbag_raw"):
    if os.path.isfile(os.path.join("../dataset/rosbag_raw", file_name)):
        name_without_extension = os.path.splitext(file_name)[0]
        bag_time = name_without_extension

        # 设置图片文件夹路径和输出视频路径
        image_folder = f"../dataset/raw/{bag_time}/video/leg/" 
        output_folder = f"../dataset/raw/{bag_time}/video/leg_video/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_video_path = f"{output_folder}/leg_video.mp4"  # 输出视频路径

        # 获取文件夹中的所有图片文件
        image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]  # 根据需要修改图片文件扩展名

        # 排序图片文件
        image_files.sort(key=custom_sort_key)

        # 获取第一张图片的宽度和高度
        first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
        height, width, layers = first_image.shape

        # 设置视频编码器和帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可能需要根据系统支持的编码器进行更改
        frame_rate = 10.0  # 帧率

        # 创建视频写入对象
        video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        # 逐帧写入图片到视频
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            img = cv2.imread(image_path)
            video.write(img)

        # 释放视频写入对象
        video.release()

        print("视频生成完成：", output_video_path)
