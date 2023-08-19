import torch

from torch import nn
from torch.nn import functional as F

from .conv2 import Conv2d


class SyncNet_color(nn.Module):

    def __init__(self):
        #
        super(SyncNet_color, self).__init__()
        #
        # 输入: (N, Cin,  Hin,  Win )
        # 输出: (N, Cout, Hout, Wout)
        #
        self.faces_encoder = nn.Sequential(  # {Tensor: (32, 15, 288//2, 288)}

            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 32, kernel_size=5, stride=1, padding=1),  # TODO 网络调整部分
            Conv2d(32, 32, kernel_size=5, stride=1, padding=1),  # TODO 网络调整部分

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),

            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),

            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),

            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=0),  # TODO 网络调整部分

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),

            Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # TODO 网络调整部分

            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.audio_encoder = nn.Sequential(  # {Tensor: (32,  1, 80, 16)} -- 这个网络结构与原始网络结构一致

            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),

            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),

            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),

            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, audio_sequences, faces_sequences):
        #
        # print(f'audio_sequences: {audio_sequences.size()}') # audio_sequences := (B, dim, T)
        #
        # faces_sequences：{Tensor: (32, 15, 144, 288)} --> [N, C, H/2, W]
        # audio_sequences：{Tensor: (32,  1, 80,   16)}

        faces_embedding = self.faces_encoder(faces_sequences)  # {Tensor: {32, 512, 1, 1}}
        audio_embedding = self.audio_encoder(audio_sequences)  # {Tensor: {32, 512, 1, 1}}

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)  # {Tensor: {32, 512}}
        faces_embedding = faces_embedding.view(faces_embedding.size(0), -1)  # {Tensor: {32, 512}}

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        faces_embedding = F.normalize(faces_embedding, p=2, dim=1)

        return audio_embedding, faces_embedding
