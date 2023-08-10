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
    def __init__(self, input_channels=100, output_channels=100, tensor_len=10):
        super(Fusion_transformer, self).__init__()

        self.fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model= input_channels, nhead=8)
        self.fusion_encoder = nn.TransformerEncoder(
            self.fusion_encoder_layer, num_layers=6)
        self.fusion_position_embedding = PositionEmbedding(
            input_channels=input_channels, tensor_len=tensor_len)
        self.fusion_linear = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        x = self.fusion_position_embedding(x)
        x = self.fusion_encoder(x, mask=None)
        x = self.fusion_linear(x)
        return x


if __name__ == '__main__':
    input_vector = torch.randn(1, 400)
    fusion_transformer = Fusion_transformer(
        input_channels=400, tensor_len=1, output_channels=100)
    output_vector = fusion_transformer(input_vector)
    # torchsummary.summary(fusion_transformer, (1, 400))
    print(output_vector.shape)
