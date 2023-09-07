# getdata说明   
本文件夹中的程序用于从录制的rosbag中提取原始数据  
注意：认为rosbag的存放地址和提取原始的存放地址都是固定的  
  rosbag存放：../dataset/rosbag_raw/  
  提取数据存放： ../dataset/raw/{rosbagname}/  

## 1. BEV_map.py  
### 说明  
自动提取所有rosbag。

/usb_cam/image_raw -> jpg图片  
/usb_cam0/image_raw0 -> jpg图片  
/usb_cam2/image_raw2 -> jpg图片  
/usb_cam4/image_raw4 -> jpg图片  
/usb_cam6/image_raw6 -> jpg图片  
/imu_data -> csv文件
/ecparm -> csv文件 (motor/joystick/sensor分开存储)

注：单线程运行，需要对缓慢的运行速度有心里预期。  
一般开多个窗口同时运行多个单摸态的get_data程序，手动实现多线程。
### 使用
python get_sensors.py
### 补充  
如需单独提取各模态数据，可以使用各get_xx.py  
其中get_mpr4_from_jpg.py是通过jpg图片生成mp4视频

## 2. change_file_sort.py
### 说明  
对各数据名进行重排序，按照时间顺序排列为1,2,3...

### 使用
python change_file_sort.py

## 3. change_file_sort.py
### 说明  
resize图片为56x56的分辨率，供video swim transfomer使用

### 使用
python change_video_shape.py