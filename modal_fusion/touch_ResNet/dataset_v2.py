import torch
import os
import numpy as np


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类:MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        txts = []
        labels = []
        
        g = os.walk(root)  
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if path[-2] == '/':  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # labels.append(int(path[-1:])-1)  # start from 1
                    labels.append(int(path[-1:]))    # start from 0
                else:
                    # labels.append(int(path[-2:])-1)  # start from 1
                    labels.append(int(path[-2:]))    # start from 0

                lines = open(os.path.join(path, file_name)).readlines()
                tmpdata = []
                col = []
                for k in lines:
                    k = k.strip('\n')  # 去掉读取中的换行字符
                    k = k.strip('-')   # 去掉读取中的换行字符
                    col.append(k) 
                while '' in col:
                    col.remove('')     # 去掉读取的空格
                for tmp in col:
                    tmp = tmp.split(',')
                    # tmp = tmp[:-1]  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    tmp = [int(x) for x in tmp]
                    tmpdata.append(tmp)
                if len(tmpdata) != 20*32:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    continue
                txts.append(tmpdata)
        
        txts = np.array(txts)
        labels = np.array(labels)
        print(txts.shape)
        print(labels.shape)

        txts = txts.reshape((-1, 640, 32))  # !!!!!!!!!!!!!!!!!!!!!
        txts = txts.reshape(-1, 20, 32, 32).transpose(0, 2, 3, 1)  # !!!!!!!!!!!!!!!!!!!!!

        self.txts = txts
        self.transform = transform
        self.target_transform = target_transform
        self.labels = labels
 
    def __getitem__(self, index):  # 这个方法是必须要有的,用于按照索引读取每个元素的具体内容
        data = self.txts[index]
        label = self.labels[index]
        if self.transform is not None:
            data = data.astype(float)  # numpy强制类型转换
            data = self.transform(data)  # 是否进行transform
        return data, label  # return很关键,return回哪些内容,那么我们在训练时循环读取每个batch时,就能获得哪些内容
 
    def __len__(self):  # 这个函数也必须要写,它返回的是数据集的长度,也就是多少张图片,要和loader的长度作区分
        return len(self.txts)
