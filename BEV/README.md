# BEV说明   
通过点云pcd和4个RGB相机的jpg图片生成BEV地图。  
  
  pcd点云地址：../dataset/rosbag_raw/{rosbagname}/cloudpoints    

  jpg图像存放：   
  ../dataset/raw/{rosbagname}/video/0  
  ../dataset/raw/{rosbagname}/video/2  
  ../dataset/raw/{rosbagname}/video/4   
  ../dataset/raw/{rosbagname}/video/6     

## 1. BEV_map.py  
### 环境：
  
  你可能需要安装yolov8的环境，请确保你的环境符合  
  Python>=3.8 environment with PyTorch>=1.8

然后即可一键安装：pip install ultralytics

### 说明  
将指定某rosbag包的图像和点云生成BEV图  
输入：某rosbag包生成的点云及4个RGB图片  
输出：每帧对应的BEV图  
### 使用
python BEV_map.py "rosbag_name"  
## 2. BEV_map_total.py  


### 说明  
将所有rosbag包的图像和点云生成BEV图，需按照数据存储路径  
输入：某rosbag包生成的点云及4个RGB图片  
输出：每帧对应的BEV图  
### 使用
python BEV_map.py 