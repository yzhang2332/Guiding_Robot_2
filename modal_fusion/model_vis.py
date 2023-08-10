from feature_abstract import audio_feature
import multi_swin 

from tensorboardX import SummaryWriter
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audiotest = audio_feature.AudioFeatureExtract(audio_addr="modal_fusion/datasets/wav.wav",debug=False)
mfccFeatures = audiotest.extract_mfcc()

model = multi_swin.MultiSwin(
    input_channels=1, output_classes=100).to(device)

# 读入一个视频的六秒，每秒取一帧转成tensor
def video2tensor(video_addr = None):
    video_tensor = torch.zeros(3, 3, 224, 224)
    for i in range(3):
        video_tensor[i] = torch.rand(3, 224, 224)
    return video_tensor


input = torch.tensor(mfccFeatures).unsqueeze(0).unsqueeze(0).to(device)
input2 = video2tensor().unsqueeze(0).to(device)
# print(input.shape)
output = model(input,input2,input,input)
# print(output.shape)

# with SummaryWriter(logdir="network_visualization") as w:
#     w.add_graph(model, (input,input2,input,input))