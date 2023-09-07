#!/usr/bin/env python

import os
import cv2
import pandas as pd
from PIL import Image

def change_jpg(bag_time, code):
    # 指定包含图片文件的文件夹路径
    folder_path = f'../dataset/raw/{bag_time}/video/{code}/'
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # 确保只处理.jpg文件
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, filename)
            # 打开原始图像
            original_image = Image.open(input_path)
            # 将原始图像调整为目标大小 (56x56)
            resized_image = original_image.resize((56, 56), Image.ANTIALIAS)
            # 保存压缩后的图像
            resized_image.save(output_path)
            # 关闭图像
            original_image.close()
            resized_image.close()

    print("图像已成功压缩并保存到输出文件夹。")

if __name__ == "__main__":
    for file_name in os.listdir("../dataset/raw"):
        name_without_extension = os.path.splitext(file_name)[0]
        bag_time = name_without_extension
        change_jpg(bag_time, 0)
        change_jpg(bag_time, 2)
        change_jpg(bag_time, 4)
        change_jpg(bag_time, 6)
