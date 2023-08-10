preprocess.py: 把各模态分割、滑窗等，路径写入到annotation.csv中

test_model_train.py：训练

multi_swin.py: 特征提取

* video用transformer
* audio touch pose用resnet
* 特征提取输出过一个nn.LayerNorm标准化
* model fusion的方式：所有张量拼接到一起，过一个relu，再用resnet