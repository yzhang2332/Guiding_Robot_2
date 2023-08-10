import torch
import torch.nn as nn
from collections import OrderedDict
from video_swin_transformer import SwinTransformer3D

model = SwinTransformer3D(embed_dim=96, 
                          depths=(2, 2, 6, 2),
                          num_heads=(3, 6, 12, 24),
                          patch_size=(2,4,4),
                          window_size=(8,7,7), 
                          drop_path_rate=0.4, 
                          patch_norm=True)

# https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py
# checkpoint = torch.load('modal_fusion\model\swin_base_patch244_window1677_sthv2.pth')
checkpoint = torch.load('modal_fusion/model/swin_tiny_patch244_window877_kinetics400_1k.pth')
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    if 'backbone' in k:
        name = k[9:]
        new_state_dict[name] = v 

model.load_state_dict(new_state_dict) 

dummy_x = torch.rand(1, 3, 32, 224, 224)
logits = model(dummy_x)
print(logits.shape)

# print(model)