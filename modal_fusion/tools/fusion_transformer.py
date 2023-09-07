# import torch.nn as nn
# import torch
# import math
#
# # 定义位置编码模块
# class PositionEmbedding(nn.Module):
#     def __init__(self, input_channels, tensor_len=5000):
#         super(PositionEmbedding, self).__init__()
#
#         pe = torch.zeros(tensor_len, input_channels)
#         position = torch.arange(0, tensor_len, dtype=torch.float).unsqueeze(1)
#
#         div_term = torch.exp(
#             torch.arange(0, input_channels, 2).float() *
#             (-math.log(10000.0) / input_channels))
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(1), :].squeeze(1)
#         return x
#
#
# # 定义Fusion Transformer模型
# class Fusion_transformer(nn.Module):
#     def __init__(self, input_channels=100, output_channels=100, tensor_len=10):
#         super(Fusion_transformer, self).__init__()
#
#         self.fusion_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_channels, nhead=8)
#         self.fusion_encoder = nn.TransformerEncoder(
#             self.fusion_encoder_layer, num_layers=6)
#
#         # Decoder部分，用于自监督训练，这里直接复用了Encoder部分
#         self.fusion_decoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_channels, nhead=8)
#         self.fusion_decoder = nn.TransformerEncoder(
#             self.fusion_decoder_layer, num_layers=6)
#
#         # 注意：output_channels改为input_channels，因为需要预测下一时刻的电机参数
#         self.fusion_position_embedding = PositionEmbedding(
#             input_channels=input_channels, tensor_len=tensor_len)
#         # 注意：output_channels改为input_channels，因为需要预测下一时刻的电机参数
#         self.fusion_linear = nn.Linear(input_channels, input_channels)
#
#     def forward(self, x):
#         # 此处的x表示输入的历史轨迹，需要将它加入到位置编码中
#         x = self.fusion_position_embedding(x)
#         # 使用Transformer进行编码
#         x_encoded = self.fusion_encoder(x, mask=None)
#         # 使用线性层进行预测，将输入的历史轨迹映射到下一时刻的电机参数
#         predicted_params = self.fusion_linear(x_encoded)
#
#         # Decoder部分，用于自监督训练
#         # 此处的输入是预测的电机参数，将其加入到位置编码中
#         predicted_params_encoded = self.fusion_position_embedding(predicted_params)
#         # 使用Decoder进行解码，以自监督训练模型
#         decoded_params = self.fusion_decoder(predicted_params_encoded, mask=None)
#
#         return predicted_params, decoded_params
#
#
# if __name__ == '__main__':
#     # 示例输入，假设历史轨迹是一个1x400的向量
#     input_vector = torch.randn(1, 400)
#     # 创建Fusion Transformer模型，此处的tensor_len仍然是1，因为我们只是处理单个时刻的历史轨迹
#     fusion_transformer = Fusion_transformer(
#         input_channels=400, tensor_len=1, output_channels=400)  # 此处output_channels设置为输入通道数
#     # 输入历史轨迹，模型会预测下一时刻的电机参数，并进行自监督训练
#     predicted_params, decoded_params = fusion_transformer(input_vector)
#     print(predicted_params.shape)  # 输出预测的电机参数的维度
#     print(decoded_params.shape)  # 输出解码后的电机参数的维度
#
#
import torch.nn as nn
import torch
import math
import torchsummary


class PositionEmbedding(nn.Module):
    def __init__(self, input_channels, tensor_len=5000):
        super(PositionEmbedding, self).__init__()

        pe = torch.zeros(tensor_len, input_channels)
        position = torch.arange(0, tensor_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, input_channels, 2).float() *
            (-math.log(10000.0) / input_channels))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class Fusion_transformer(nn.Module):
    def __init__(self, input_channels=100, output_channels=100, tensor_len=60):
        super(Fusion_transformer, self).__init__()
        # encoder
        self.fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_channels, nhead=8)
        self.fusion_encoder = nn.TransformerEncoder(
            self.fusion_encoder_layer, num_layers=6)
        self.fusion_position_embedding = PositionEmbedding(
            input_channels=input_channels, tensor_len=tensor_len)
        # decoder
        self.fusion_linear = nn.Linear(input_channels, output_channels*5)

    def forward(self, x):
        x = self.fusion_position_embedding(x)
        x = self.fusion_encoder(x, mask=None)
        x = self.fusion_linear(x)
        x = x.view(x.size(0),10,5)
        x = nn.functional.softmax(x, dim=2)
        # print(x)
        return x

    def self_supervised_loss(self, predicted, target):
        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted, target)
        return loss


if __name__ == '__main__':
    input_vector = torch.randn(1, 400)
    fusion_transformer = Fusion_transformer(
        input_channels=400, tensor_len=1, output_channels=100)
    output_vector = fusion_transformer(input_vector)
    torchsummary.summary(fusion_transformer, (1, 400))
    print(output_vector.shape)

